#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train2.py — Дообучение RAG-энкодера на полных данных
✅ Загрузка весов из logs/model.pth
✅ Валидационные книги берутся из logs/val_books.json (не переиспользуются в train)
✅ 3 негативных примера на якорь (InfoNCE loss)
✅ Конфиг: BATCH_SIZE=8, ACCUM=2, VAL_INTERVAL=512, PLOT=256, CKPT=10240
✅ WS=512 (увеличено для лучшего контекста)
✅ ОДНА эпоха на 100% данных
"""

# ==================== ИМПОРТЫ ====================
import os
import sys
import json
import math
import random
import time
import gc
import faulthandler
faulthandler.enable()

from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False

from model import RAGEncoder, VOCAB_SIZE, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN, BOS_TOKEN_ID, PAD_TOKEN_ID
# ====================================================

# ==================== КОНСТАНТЫ ====================
# Данные и модель
WINDOW_SIZE = 1024
TRAIN_DATA_FRACTION = 1.0  # 100% данных для дообучения
NUM_EPOCHS = 1  # Одна полная эпоха

# Батчинг
BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
NUM_WORKERS = 0  # Стабильность с большим dict в RAM

# Оптимизация
BASE_LR = 2e-6
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200

# Контрастивный лосс
N_NEGATIVES = 3  # Количество негативных примеров на якорь
TEMPERATURE = 0.05  # Для InfoNCE
MARGIN = 0.3  # Резерв для triplet

# Пути
PREV_LOG_DIR = "logs6_from_3_using_2_2"  # Откуда брать чекпоинт и валидационные ID
CUR_LOG_DIR = "logs6_from_3_using_2_3"  # Куда писать новые логи
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")
VAL_IDS_PATH = "logs2/val_books.json"

DATA_ROOT = "/mnt/news/llm_ds/fics"

# Логирование
VAL_BOOK_COUNT = 2000
VAL_BATCH_SIZE = 8
LOG_INTERVAL = 10
VAL_INTERVAL = 512
PLOT_INTERVAL = 256
CHECKPOINT_INTERVAL = 2048 * 5  # 10240 для полных данных

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
MASK_TOKEN_ID = 0
# ====================================================

# ==================== НАСТРОЙКА ПУТЕЙ ====================
os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss.jsonl')
METRICS_LOG = os.path.join(CUR_LOG_DIR, 'metrics.jsonl')
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training.png')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model.pth")

for f in [LOSS_LOG, METRICS_LOG]:
    open(f, 'w').close()

def scan_books_paths(data_root: str) -> dict:
    books = defaultdict(list)
    root = Path(data_root)
    for shard in sorted(root.iterdir()):
        if not shard.is_dir() or '-' not in shard.name:
            continue
        for book_folder in sorted(shard.iterdir()):
            if not book_folder.is_dir() or not book_folder.name.startswith('book_'):
                continue
            book_id = book_folder.name.replace('book_', '')
            chapters = sorted(
                [str(p) for p in book_folder.glob('chap*.npy')],
                key=lambda x: int(Path(x).stem.replace('chap', ''))
            )
            if chapters:
                books[book_id] = chapters
    return books

def load_books_to_ram(selected_ids: list, all_paths: dict, window_size: int, pad_id: int) -> dict:
    books = {}
    for book_id in tqdm(selected_ids, desc="Loading to RAM", unit="book", file=sys.stderr, leave=True):
        paths = all_paths[book_id]
        tensors = []
        for chap_path in paths:
            tokens = np.load(chap_path).astype(np.int64)
            tokens = np.concatenate([[BOS_TOKEN_ID], tokens])
            if len(tokens) >= window_size:
                tokens = tokens[:window_size]
            else:
                tokens = np.pad(tokens, (0, window_size - len(tokens)), constant_values=pad_id)
            tensors.append(torch.from_numpy(tokens))
        books[book_id] = tensors
    return books

def augment_chapter(tokens: torch.Tensor, mask_prob=0.15) -> torch.Tensor:
    aug = tokens.clone()
    maskable = (tokens != PAD_TOKEN_ID) & (tokens != BOS_TOKEN_ID)
    candidates = torch.where(maskable)[0]
    if candidates.numel() == 0: return aug
    num_mask = max(1, int(candidates.numel() * mask_prob))
    pos = candidates[torch.randperm(candidates.numel())[:num_mask]]
    rand = torch.rand(num_mask, device=tokens.device)
    m_mask = rand < 0.8
    if m_mask.any(): aug[pos[m_mask]] = MASK_TOKEN_ID
    m_rand = (rand >= 0.8) & (rand < 0.9)
    if m_rand.any():
        n_rand = m_rand.sum().item()
        aug[pos[m_rand]] = torch.randint(2, VOCAB_SIZE, (n_rand,), device=tokens.device)
    return aug

class ContrastiveDataset(Dataset):
    def __init__(self, books: dict, window_size: int, pad_id: int, n_negatives: int, is_val: bool = False):
        self.books = books
        self.book_ids = list(books.keys())
        self.n_negatives = n_negatives
        self.is_val = is_val
        self.n_books = len(self.book_ids)

    def __len__(self):
        return self.n_books * (1 if self.is_val else 2)

    def __getitem__(self, idx):
        if self.is_val:
            book_id = self.book_ids[idx % self.n_books]
            anchor = self.books[book_id][0]
            positive = self.books[book_id][1] if len(self.books[book_id]) > 1 else augment_chapter(anchor, 0.1)
            neg_id = self.book_ids[(idx + 1) % self.n_books]
            negative = self.books[neg_id][0]
            # Для валидации возвращаем один негатив (как раньше)
            return anchor, positive, negative.unsqueeze(0)  # [1, D]
        else:
            book_id = random.choice(self.book_ids)
            chapters = self.books[book_id]
            anchor = random.choice(chapters)
            if len(chapters) > 1:
                positive = random.choice([c for c in chapters if c is not anchor])
            else:
                positive = augment_chapter(anchor, 0.2)
            
            # Выбираем N_NEGATIVES разных книг
            other_ids = [bid for bid in self.book_ids if bid != book_id]
            neg_ids = random.sample(other_ids, min(self.n_negatives, len(other_ids)))
            negatives = torch.stack([random.choice(self.books[nid]) for nid in neg_ids])  # [N, D]
            
            return anchor, positive, negatives

def info_nce_loss(anchor, positive, negatives, temperature: float = 0.05):
    """
    InfoNCE loss с множественными негативами.
    anchor, positive: [B, D]
    negatives: [B, N, D]
    """
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)
    
    # Позитивное сходство: [B]
    pos_sim = (anchor * positive).sum(dim=-1) / temperature
    
    # Негативные сходства: [B, N]
    neg_sim = torch.einsum('bd,bnd->bn', anchor, negatives) / temperature
    
    # Логиты: [B, 1+N], правильный ответ всегда на позиции 0
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    
    loss = F.cross_entropy(logits, labels)
    
    # Для логирования: среднее по позитиву и среднее по негативам
    pos_mean = (anchor * positive).sum(dim=-1).mean().item()
    neg_mean = torch.einsum('bd,bnd->bn', F.normalize(anchor,p=2,dim=-1), F.normalize(negatives,p=2,dim=-1)).mean().item()
    
    return loss, pos_mean, neg_mean

@torch.no_grad()
def validate(model, val_loader, temperature: float, device: str, dtype: torch.dtype):
    model.eval()
    t_loss, t_pos, t_neg, n = 0.0, 0.0, 0.0, 0
    for anchor, pos, negs in val_loader:  # negs: [B, 1, D]
        anchor = anchor.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        negs = negs.to(device, non_blocking=True)
        with autocast('cuda', dtype=dtype):
            za, zp, zn = model(anchor), model(pos), model(negs.squeeze(1))
            # Для валидации используем один негатив (как в triplet)
            loss, ps, ns = info_nce_loss(za, zp, zn.unsqueeze(1), temperature)
        t_loss += loss.item(); t_pos += ps; t_neg += ns; n += 1
    model.train()
    if n == 0: return {'loss': 0.0, 'pos_sim': 0.0, 'neg_sim': 0.0, 'margin': 0.0}
    return {'loss': t_loss/n, 'pos_sim': t_pos/n, 'neg_sim': t_neg/n, 'margin': (t_pos-t_neg)/n}

def draw_plot(steps, losses, pos, neg, margins, lrs):
    if not steps: return
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0,0].plot(steps, losses, 'k-', lw=1.5); ax[0,0].set_title('InfoNCE Loss'); ax[0,0].grid(True, alpha=0.3)
    ax[0,1].plot(steps, pos, 'b-', marker='o', ms=3, label='Pos Sim'); ax[0,1].plot(steps, neg, 'r-', marker='x', ms=3, label='Neg Sim')
    ax[0,1].set_title('Cosine Similarity'); ax[0,1].legend(); ax[0,1].grid(True, alpha=0.3)
    ax[1,0].plot(steps, margins, 'g-', marker='s', ms=3); ax[1,0].axhline(y=0.3, color='orange', ls='--', alpha=0.5)
    ax[1,0].set_title('Margin (Pos - Neg)'); ax[1,0].grid(True, alpha=0.3)
    ax[1,1].plot(steps, lrs, 'm-', marker='d', ms=2); ax[1,1].set_title('LR'); ax[1,1].grid(True, alpha=0.3); ax[1,1].set_yscale('log')
    plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight'); plt.close()

def save_ckpt(path, model, opt, step, info=""):
    tmp = path + '.tmp'
    torch.save({
        'step': step, 'info': info,
        'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
        'opt_state_dict': opt.state_dict()
    }, tmp)
    os.replace(tmp, path)

# ==================== MAIN ====================
def main():
    tqdm.write(f"💾 Устройство: {DEVICE} | {DTYPE} | WS={WINDOW_SIZE}")
    tqdm.write(f"🔄 Fine-tuning: {CHECKPOINT_PATH} → {CUR_LOG_DIR}")
    tqdm.write(f"📊 Negatives: {N_NEGATIVES} | Batch: {BATCH_SIZE}x{ACCUMULATION_STEPS}")
    
    # 1. Загрузка валидационных ID из ПРЕДЫДУЩего запуска
    if not os.path.exists(VAL_IDS_PATH):
        tqdm.write(f"❌ Не найден {VAL_IDS_PATH} — создайте сначала logs/val_books.json")
        sys.exit(1)
    with open(VAL_IDS_PATH, 'r') as f:
        val_ids = json.load(f)
    tqdm.write(f"📋 Валидационные книги загружены: {len(val_ids)} ID из {PREV_LOG_DIR}")
    
    # 2. Сканирование путей
    tqdm.write("📂 Сканирование структуры...")
    all_paths = scan_books_paths(DATA_ROOT)
    all_ids = list(all_paths.keys())
    
    # 3. Исключаем валидационные книги из обучения
    train_ids = [bid for bid in all_ids if bid not in set(val_ids)]
    tqdm.write(f"📚 Train: {len(train_ids):,} книг | Val: {len(val_ids):,} книг")
    
    # 4. Загрузка в RAM (только нужные)
    tqdm.write("📥 Загрузка в RAM...")
    train_books = load_books_to_ram(train_ids, all_paths, WINDOW_SIZE, PAD_TOKEN_ID)
    val_books = load_books_to_ram(val_ids, all_paths, WINDOW_SIZE, PAD_TOKEN_ID)
    del all_paths; gc.collect(); torch.cuda.empty_cache()
    
    # 5. Датасеты
    train_ds = ContrastiveDataset(train_books, WINDOW_SIZE, PAD_TOKEN_ID, N_NEGATIVES, is_val=False)
    val_ds = ContrastiveDataset(val_books, WINDOW_SIZE, PAD_TOKEN_ID, 1, is_val=True)  # 1 негатив для валидации
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 6. Модель + загрузка чекпоинта
    tqdm.write("📦 Загрузка модели...")
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Конвертация в целевую точность
        if DTYPE == torch.bfloat16:
            for k, v in cleaned.items():
                if v.dtype == torch.float32:
                    cleaned[k] = v.bfloat16()
        model.load_state_dict(cleaned, strict=False)
        tqdm.write(f"✅ Веса загружены из {CHECKPOINT_PATH}")
    else:
        tqdm.write(f"⚠️ {CHECKPOINT_PATH} не найден, инициализация с нуля")
    
    model.enable_gradient_checkpointing(False)  # WS=512 влезает без чекпоинтинга
    
    # 7. Оптимизатор
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=(DEVICE=='cuda'))
    scaler = GradScaler('cuda', enabled=(DTYPE == torch.float16))
    
    total_steps = len(train_dl) * NUM_EPOCHS // ACCUMULATION_STEPS
    get_lr = lambda s: BASE_LR*(s+1)/WARMUP_STEPS if s<WARMUP_STEPS else MIN_LR+(BASE_LR-MIN_LR)*0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/max(1,total_steps-WARMUP_STEPS)))
    
    # Логи
    s, l, p, n, m, lr_log = [], [], [], [], [], []
    v_hist = {'step':[], 'loss':[], 'pos_sim':[], 'neg_sim':[], 'margin':[]}
    
    gs, os_ = 0, 0
    t0 = time.time()
    
    # Валидация шаг 0
    tqdm.write("🔍 Валидация (шаг 0)...")
    met = validate(model, val_dl, TEMPERATURE, DEVICE, DTYPE)
    v_hist['step'].append(0)
    for k in met: v_hist[k].append(met[k])
    tqdm.write(f"📊 Step 0: Loss={met['loss']:.3f} | Margin={met['margin']:.3f}")
    
    try:
        model.train()
        tqdm.write(f"🚀 Начало дообучения: {len(train_dl)} итераций, {NUM_EPOCHS} эпох")
        pbar = tqdm(train_dl, desc="Training", dynamic_ncols=True, leave=True, file=sys.stdout)
        opt.zero_grad(set_to_none=True)
        
        for epoch in range(NUM_EPOCHS):
            for anc, pos, negs in pbar:
                anc = anc.to(DEVICE, non_blocking=True)
                pos = pos.to(DEVICE, non_blocking=True)
                negs = negs.to(DEVICE, non_blocking=True)  # [B, N, D]
                
                with autocast('cuda', dtype=DTYPE):
                    za, zp, zn = model(anc), model(pos), model(negs.view(-1, negs.size(-1)))
                    zn = zn.view(BATCH_SIZE, N_NEGATIVES, -1)  # [B, N, D]
                    loss, ps, ns = info_nce_loss(za, zp, zn, TEMPERATURE)
                
                scaler.scale(loss / ACCUMULATION_STEPS).backward()
                
                if (gs + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    cur_lr = get_lr(os_)
                    for pg in opt.param_groups: pg['lr'] = cur_lr
                    os_ += 1
                    if gs % 500 == 0: torch.cuda.empty_cache()
                
                gs += 1
                cl = loss.item()
                s.append(gs); l.append(cl); p.append(ps); n.append(ns); m.append(ps-ns); lr_log.append(opt.param_groups[0]['lr'])
                
                with open(LOSS_LOG, 'a') as f:
                    f.write(json.dumps({'step': gs, 'loss': cl, 'pos': ps, 'neg': ns, 'lr': opt.param_groups[0]['lr']}) + '\n')
                
                if gs % VAL_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    vm = validate(model, val_dl, TEMPERATURE, DEVICE, DTYPE)
                    v_hist['step'].append(gs)
                    for k in vm: v_hist[k].append(vm[k])
                    with open(METRICS_LOG, 'a') as f:
                        f.write(json.dumps({'step': gs, **vm}) + '\n')
                    pbar.set_postfix(val_loss=f"{vm['loss']:.3f}", val_margin=f"{vm['margin']:.3f}")
                    gc.collect()
                
                if gs % PLOT_INTERVAL == 0:
                    draw_plot(s, l, p, n, m, lr_log)
                
                if gs % CHECKPOINT_INTERVAL == 0:
                    save_ckpt(MODEL_PATH, model, opt, gs, f'Epoch{epoch+1}_Step{gs}')
                    tqdm.write(f"💾 Checkpoint: {MODEL_PATH}")
                
                pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
                
                del anc, pos, negs, za, zp, zn, loss
        
        save_ckpt(MODEL_PATH, model, opt, gs, 'Fine-tuning completed')
        tqdm.write(f"\n🎉 Дообучение завершено! Модель: {MODEL_PATH}")
        
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Прервано")
        save_ckpt(os.path.join(CUR_LOG_DIR, 'interrupted.pth'), model, opt, gs, 'Interrupted')
    except Exception as e:
        tqdm.write(f"\n❌ Ошибка: {e}")
        import traceback; traceback.print_exc()
        save_ckpt(os.path.join(CUR_LOG_DIR, 'error.pth'), model, opt, gs, str(e))
    finally:
        draw_plot(s, l, p, n, m, lr_log)
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        tqdm.write(f"⏱️ Время: {(time.time()-t0)/60:.1f} мин | Шагов: {gs}")

if __name__ == '__main__':
    main()