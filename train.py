#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ram_contrastive.py — Контрастивное обучение RAG-энкодера (стабильная версия)
✅ Ограничение данных применяется ДО загрузки в RAM
✅ Отключен gradient checkpointing (WS=256 влезает в 1.2GB, чекпоинтинг вызывал segfault)
✅ PYTORCH_ALLOC_CONF для предотвращения фрагментации памяти
✅ В цикле обучения НЕТ print(). Только tqdm.set_postfix
✅ WS=256, паддинг до 0, аугментация для книг с 1 главой
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

# Стабильность аллокатора (лечит "Аварийный останов" на длинных прогонах)
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False  # Предотвращает cuDNN-краши

from model import RAGEncoder, VOCAB_SIZE, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN, BOS_TOKEN_ID, PAD_TOKEN_ID
# ====================================================

# ==================== КОНСТАНТЫ ====================
WINDOW_SIZE = 256
NUM_EPOCHS = 1
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
BASE_LR = 5e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 500

# Ограничение обучающих данных: 1.0 = 100%, 0.1 = 10% (применяется ДО загрузки в RAM)
TRAIN_DATA_FRACTION = 1

# num_workers=0 критически важен для стабильности с большими dict в RAM
NUM_WORKERS = 0

MARGIN = 0.3
MASK_TOKEN_ID = 0

DATA_ROOT = "/mnt/news/llm_ds/fics"
LOG_DIR = "logs"

VAL_BOOK_COUNT = 2000
VAL_BATCH_SIZE = 8

LOG_INTERVAL = 10
VAL_INTERVAL = 512
PLOT_INTERVAL = 256
CHECKPOINT_INTERVAL = 2048

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
# ====================================================

# ==================== НАСТРОЙКА ПУТЕЙ ====================
os.makedirs(LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(LOG_DIR, 'loss.jsonl')
METRICS_LOG = os.path.join(LOG_DIR, 'metrics.jsonl')
PLOT_PATH = os.path.join(LOG_DIR, 'training.png')
MODEL_PATH = os.path.join(LOG_DIR, "model.pth")
VAL_IDS_PATH = os.path.join(LOG_DIR, "val_books.json")

for f in [LOSS_LOG, METRICS_LOG]:
    open(f, 'w').close()

def scan_books_paths(data_root: str) -> dict:
    """Только сканирует пути, не загружает данные. Возвращает {book_id: [path1, path2, ...]}"""
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
    """Загружает ТОЛЬКО выбранные книги в RAM."""
    books = {}
    for book_id in tqdm(selected_ids, desc="Loading selected to RAM", unit="book", file=sys.stderr, leave=True):
        paths = all_paths[book_id]
        tensors = []
        for chap_path in paths:
            tokens = np.load(chap_path).astype(np.int64)
            tokens = np.concatenate([[BOS_TOKEN_ID], tokens])
            if len(tokens) >= window_size:
                tokens = tokens[:window_size]
            else:
                tokens = np.pad(tokens, (0, window_size - len(tokens)), constant_values=pad_id)
            # torch.from_numpy не копирует данные, работает быстрее
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
    def __init__(self, books: dict, window_size: int, pad_id: int, is_val: bool = False):
        self.books = books
        self.book_ids = list(books.keys())
        self.is_val = is_val
        # Кешируем длину для быстрого доступа
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
        else:
            book_id = random.choice(self.book_ids)
            chapters = self.books[book_id]
            anchor = random.choice(chapters)
            if len(chapters) > 1:
                positive = random.choice([c for c in chapters if c is not anchor])
            else:
                positive = augment_chapter(anchor, 0.2)
            neg_id = random.choice(self.book_ids)
            while neg_id == book_id:
                neg_id = random.choice(self.book_ids)
            negative = random.choice(self.books[neg_id])
        return anchor, positive, negative

def triplet_loss(anchor, positive, negative, margin: float = 0.3):
    a = F.normalize(anchor, p=2, dim=-1)
    p = F.normalize(positive, p=2, dim=-1)
    n = F.normalize(negative, p=2, dim=-1)
    pos_sim = F.cosine_similarity(a, p, dim=-1)
    neg_sim = F.cosine_similarity(a, n, dim=-1)
    return torch.clamp(margin - pos_sim + neg_sim, min=0.0).mean(), pos_sim.mean().item(), neg_sim.mean().item()

@torch.no_grad()
def validate(model, val_loader, margin: float, device: str, dtype: torch.dtype):
    model.eval()
    t_loss, t_pos, t_neg, n = 0.0, 0.0, 0.0, 0
    for anchor, pos, neg in val_loader:
        anchor = anchor.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)
        with autocast('cuda', dtype=dtype):
            za, zp, zn = model(anchor), model(pos), model(neg)
            loss, ps, ns = triplet_loss(za, zp, zn, margin)
        t_loss += loss.item(); t_pos += ps; t_neg += ns; n += 1
    model.train()
    if n == 0: return {'loss': 0.0, 'pos_sim': 0.0, 'neg_sim': 0.0, 'margin': 0.0}
    return {'loss': t_loss/n, 'pos_sim': t_pos/n, 'neg_sim': t_neg/n, 'margin': (t_pos-t_neg)/n}

def draw_plot(steps, losses, pos, neg, margins, lrs):
    if not steps: return
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0,0].plot(steps, losses, 'k-', lw=1.5); ax[0,0].set_title('Loss'); ax[0,0].grid(True, alpha=0.3)
    ax[0,1].plot(steps, pos, 'b-', marker='o', ms=3, label='Pos'); ax[0,1].plot(steps, neg, 'r-', marker='x', ms=3, label='Neg')
    ax[0,1].set_title('Cosine Sim'); ax[0,1].legend(); ax[0,1].grid(True, alpha=0.3)
    ax[1,0].plot(steps, margins, 'g-', marker='s', ms=3); ax[1,0].axhline(y=0.3, color='orange', ls='--', alpha=0.5)
    ax[1,0].set_title('Margin'); ax[1,0].grid(True, alpha=0.3)
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
    tqdm.write(f"📊 fraction={TRAIN_DATA_FRACTION} | workers={NUM_WORKERS}")
    
    # 1. Быстрый скан путей (без загрузки данных)
    tqdm.write("📂 Сканирование структуры дисков...")
    all_paths = scan_books_paths(DATA_ROOT)
    all_ids = list(all_paths.keys())
    random.shuffle(all_ids)
    
    # 2. Разделение и фильтрация ДО загрузки
    val_ids = all_ids[:VAL_BOOK_COUNT]
    train_ids = all_ids[VAL_BOOK_COUNT:]
    
    if TRAIN_DATA_FRACTION < 1.0:
        limit = max(1000, int(len(train_ids) * TRAIN_DATA_FRACTION))
        train_ids = train_ids[:limit]
        tqdm.write(f"⚡ Ограничение: {len(train_ids):,} книг для обучения")
        
    # 3. Загрузка ТОЛЬКО нужных книг в RAM
    tqdm.write("📥 Загрузка выбранных книг в RAM...")
    train_books = load_books_to_ram(train_ids, all_paths, WINDOW_SIZE, PAD_TOKEN_ID)
    val_books = load_books_to_ram(val_ids, all_paths, WINDOW_SIZE, PAD_TOKEN_ID)
    
    # Очистка путей из памяти
    del all_paths; gc.collect()
    torch.cuda.empty_cache()
    
    with open(VAL_IDS_PATH, 'w') as f:
        json.dump(val_ids, f, indent=2)
    tqdm.write(f"📋 Валидационные ID сохранены: {VAL_IDS_PATH}")
    
    train_ds = ContrastiveDataset(train_books, WINDOW_SIZE, PAD_TOKEN_ID, is_val=False)
    val_ds = ContrastiveDataset(val_books, WINDOW_SIZE, PAD_TOKEN_ID, is_val=True)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 4. Модель (чекпоинтинг отключен: при WS=256 модель жрёт ~1.2GB, чекпоинтинг вызывал segfault)
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    # model.enable_gradient_checkpointing(False) # По умолчанию False
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=(DEVICE=='cuda'))
    scaler = GradScaler('cuda', enabled=(DTYPE == torch.float16))
    
    total_steps = len(train_dl) * NUM_EPOCHS // ACCUMULATION_STEPS
    get_lr = lambda s: BASE_LR*(s+1)/WARMUP_STEPS if s<WARMUP_STEPS else MIN_LR+(BASE_LR-MIN_LR)*0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/max(1,total_steps-WARMUP_STEPS)))
    
    s, l, p, n, m, lr_log = [], [], [], [], [], []
    v_hist = {'step':[], 'loss':[], 'pos_sim':[], 'neg_sim':[], 'margin':[]}
    
    gs, os_ = 0, 0
    t0 = time.time()
    
    tqdm.write("🔍 Валидация (шаг 0)...")
    met = validate(model, val_dl, MARGIN, DEVICE, DTYPE)
    v_hist['step'].append(0)
    for k in met: v_hist[k].append(met[k])
    tqdm.write(f"📊 Step 0: Loss={met['loss']:.3f} | Margin={met['margin']:.3f}")
    
    try:
        model.train()
        pbar = tqdm(train_dl, desc="Training", dynamic_ncols=True, leave=True, file=sys.stdout)
        opt.zero_grad(set_to_none=True)
        
        for anc, pos, neg in pbar:
            anc = anc.to(DEVICE, non_blocking=True)
            pos = pos.to(DEVICE, non_blocking=True)
            neg = neg.to(DEVICE, non_blocking=True)
            
            with autocast('cuda', dtype=DTYPE):
                za, zp, zn = model(anc), model(pos), model(neg)
                loss, ps, ns = triplet_loss(za, zp, zn, MARGIN)
            
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
                vm = validate(model, val_dl, MARGIN, DEVICE, DTYPE)
                v_hist['step'].append(gs)
                for k in vm: v_hist[k].append(vm[k])
                with open(METRICS_LOG, 'a') as f:
                    f.write(json.dumps({'step': gs, **vm}) + '\n')
                pbar.set_postfix(val_loss=f"{vm['loss']:.3f}", val_margin=f"{vm['margin']:.3f}")
                gc.collect()
            
            if gs % PLOT_INTERVAL == 0:
                draw_plot(s, l, p, n, m, lr_log)
            
            if gs % CHECKPOINT_INTERVAL == 0:
                save_ckpt(MODEL_PATH, model, opt, gs, f'Step {gs}')
                tqdm.write(f"💾 Checkpoint: {MODEL_PATH}")
            
            pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            
            del anc, pos, neg, za, zp, zn, loss
            
        save_ckpt(MODEL_PATH, model, opt, gs, 'Completed')
        tqdm.write(f"\n🎉 Обучение завершено! Модель: {MODEL_PATH}")
        
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Прервано")
        save_ckpt(os.path.join(LOG_DIR, 'interrupted.pth'), model, opt, gs, 'Interrupted')
    except Exception as e:
        tqdm.write(f"\n❌ Ошибка: {e}")
        import traceback; traceback.print_exc()
        save_ckpt(os.path.join(LOG_DIR, 'error.pth'), model, opt, gs, str(e))
    finally:
        draw_plot(s, l, p, n, m, lr_log)
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        tqdm.write(f"⏱️ Время: {(time.time()-t0)/60:.1f} мин | Шагов: {gs}")

if __name__ == '__main__':
    main()