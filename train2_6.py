#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_new.py — Быстрое дообучение на книгах с большим эффективным батчем (64) и hard негативами.
- WS=256, загружаем только 3 главы на книгу
- Многопоточная загрузка .npy
- In-batch negatives + hard negatives из той же книги
- Сохраняем только финальный чекпоинт (один)
"""

import os
import sys
import json
import math
import random
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from model import RAGEncoder, BOS_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN

# ==================== КОНСТАНТЫ ====================
WINDOW_SIZE = 256                # 256 токенов
MAX_CHUNKS_PER_BOOK = 3          # загружаем только 3 главы на книгу (ускоряет загрузку и снижает память)
NUM_EPOCHS = 1

# Батчинг: эффективный батч = BATCH_SIZE * ACCUMULATION_STEPS
BATCH_SIZE = 1
ACCUMULATION_STEPS = 64          # итого 64 позитивные пары на шаг обновления
NUM_WORKERS = 4                  # DataLoader будет использовать 4 процесса для предзагрузки

# LR
EMBED_LR = 1e-6
BASE_LR = 2e-6
MIN_LR = 5e-7
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200

# Контрастивный лосс
N_NEGATIVES = 15                 # случайные негативы из других книг
N_HARD_NEGATIVES = 5             # жёсткие негативы из той же книги (другие главы)
TEMPERATURE = 0.07
USE_IN_BATCH_NEGATIVES = True

# Пути
PREV_LOG_DIR = "logs6_from_3_using_2_2"
CUR_LOG_DIR = "logs_NEW"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")
VAL_IDS_PATH = "logs2/val_books.json"
DATA_ROOT = "/mnt/news/llm_ds/fics"

# Валидация и логи
VAL_BATCH_SIZE = 8
LOG_INTERVAL = 10
VAL_INTERVAL = 512
PLOT_INTERVAL = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss.jsonl')
METRICS_LOG = os.path.join(CUR_LOG_DIR, 'metrics.jsonl')
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training.png')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model.pth")

for f in [LOSS_LOG, METRICS_LOG]:
    open(f, 'w').close()

# ==================== БЫСТРАЯ ЗАГРУЗКА КНИГ (только 3 главы, многопоточно) ====================
def scan_books_paths(data_root: str) -> dict:
    """Возвращает {book_id: [путь_к_главе1, ...]}"""
    books = defaultdict(list)
    root = Path(data_root)
    shards = [d for d in sorted(root.iterdir()) if d.is_dir() and '-' in d.name]
    for shard in tqdm(shards, desc="Scanning shards", file=sys.stderr):
        for book_folder in sorted(shard.iterdir()):
            if not book_folder.is_dir() or not book_folder.name.startswith('book_'):
                continue
            book_id = book_folder.name.replace('book_', '')
            chapters = sorted(
                [str(p) for p in book_folder.glob('chap*.npy')],
                key=lambda x: int(Path(x).stem.replace('chap', ''))
            )
            if chapters:
                books[book_id] = chapters[:MAX_CHUNKS_PER_BOOK]  # берём только первые N глав
    return books

def load_single_chunk(chunk_path: str, window_size: int, pad_id: int):
    """Загружает один .npy, обрезает/паддит, возвращает тензор."""
    tokens = np.load(chunk_path).astype(np.int64)
    tokens = np.concatenate([[BOS_TOKEN_ID], tokens])
    if len(tokens) >= window_size:
        tokens = tokens[:window_size]
    else:
        tokens = np.pad(tokens, (0, window_size - len(tokens)), constant_values=pad_id)
    return torch.from_numpy(tokens)

def load_books_to_ram_fast(book_ids: list, all_paths: dict, window_size: int, pad_id: int, max_workers: int = 8):
    """Многопоточная загрузка: каждая книга загружает свои главы параллельно."""
    books = {}
    for book_id in tqdm(book_ids, desc="Loading books to RAM", unit="book", file=sys.stderr):
        paths = all_paths[book_id]
        # Загружаем главы параллельно
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(load_single_chunk, p, window_size, pad_id): p for p in paths}
            tensors = []
            for future in as_completed(futures):
                tensors.append(future.result())
        # Сортируем по индексу главы (сохраняем порядок)
        books[book_id] = sorted(tensors, key=lambda t: t[0].item() if t.numel() > 0 else 0)
    return books

# ==================== ДАТАСЕТ С HARD НЕГАТИВАМИ ====================
class ContrastiveDataset(Dataset):
    def __init__(self, books: dict, n_negatives: int, n_hard_negatives: int, is_val: bool = False):
        self.books = books
        self.book_ids = list(books.keys())
        self.n_negatives = n_negatives
        self.n_hard_negatives = n_hard_negatives
        self.is_val = is_val

    def __len__(self):
        return len(self.book_ids) * (1 if self.is_val else 2)

    def __getitem__(self, idx):
        if self.is_val:
            book_id = self.book_ids[idx % len(self.book_ids)]
            chapters = self.books[book_id]
            if len(chapters) < 2:
                # если только одна глава, дублируем её
                anchor = chapters[0]
                positive = anchor.clone()
            else:
                anchor = chapters[0]
                positive = chapters[1]
            # один негатив из другой книги
            neg_id = self.book_ids[(idx + 1) % len(self.book_ids)]
            negative = self.books[neg_id][0]
            return anchor, positive, negative.unsqueeze(0)
        else:
            book_id = random.choice(self.book_ids)
            chapters = self.books[book_id]
            # Выбираем случайные anchor и positive (разные главы)
            if len(chapters) > 1:
                anchor_idx, pos_idx = random.sample(range(len(chapters)), 2)
                anchor = chapters[anchor_idx]
                positive = chapters[pos_idx]
            else:
                anchor = chapters[0]
                positive = anchor.clone()
            # Негативы из других книг
            other_ids = [bid for bid in self.book_ids if bid != book_id]
            neg_ids = random.sample(other_ids, min(self.n_negatives, len(other_ids)))
            negatives = [random.choice(self.books[nid]) for nid in neg_ids]
            # Hard негативы из той же книги (кроме anchor и positive)
            hard_candidates = [ch for i, ch in enumerate(chapters) if i not in (anchor_idx, pos_idx)]
            if self.n_hard_negatives > 0 and hard_candidates:
                n_hard = min(self.n_hard_negatives, len(hard_candidates))
                hard_negs = random.sample(hard_candidates, n_hard)
                negatives.extend(hard_negs)
            # Если недостаточно негативов, повторяем последний
            total_needed = self.n_negatives + self.n_hard_negatives
            while len(negatives) < total_needed:
                negatives.append(negatives[-1])
            negatives = negatives[:total_needed]
            negatives_tensor = torch.stack(negatives)  # [total_needed, L]
            return anchor, positive, negatives_tensor

# ==================== LOSS С IN-BATCH NEGATIVES ====================
def in_batch_infoNCE(anchors, positives, temperature):
    """anchors, positives: [B, D] уже нормализованные. Лосс InfoNCE с in-batch негативами."""
    B = anchors.shape[0]
    # Матрица сходства [B, B]
    sim = anchors @ positives.T / temperature
    labels = torch.arange(B, device=anchors.device)
    loss = F.cross_entropy(sim, labels)
    pos_sim = sim.diag().mean().item()
    neg_sim = (sim.sum() - sim.diag().sum()) / (B * (B - 1)) if B > 1 else 0.0
    return loss, pos_sim, neg_sim

# ==================== ВАЛИДАЦИЯ (с фиксированными негативами) ====================
@torch.no_grad()
def validate(model, val_loader, temperature, device, dtype):
    model.eval()
    total_loss, total_pos, total_neg = 0.0, 0.0, 0.0
    n_batches = 0
    for anchor, pos, negs in val_loader:
        anchor = anchor.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        negs = negs.to(device, non_blocking=True)  # [B, 1, L]
        with autocast('cuda', dtype=dtype):
            za = model(anchor)
            zp = model(pos)
            zn = model(negs.squeeze(1))
            # Стандартный InfoNCE с одним негативом
            za = F.normalize(za, p=2, dim=-1)
            zp = F.normalize(zp, p=2, dim=-1)
            zn = F.normalize(zn, p=2, dim=-1)
            pos_sim = (za * zp).sum(dim=-1) / temperature
            neg_sim = (za * zn).sum(dim=-1) / temperature
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            loss = F.cross_entropy(logits, torch.zeros(anchor.size(0), dtype=torch.long, device=device))
        total_loss += loss.item()
        total_pos += pos_sim.mean().item()
        total_neg += neg_sim.mean().item()
        n_batches += 1
    model.train()
    if n_batches == 0:
        return {'loss': 0.0, 'pos_sim': 0.0, 'neg_sim': 0.0, 'margin': 0.0}
    avg_loss = total_loss / n_batches
    avg_pos = total_pos / n_batches
    avg_neg = total_neg / n_batches
    return {'loss': avg_loss, 'pos_sim': avg_pos, 'neg_sim': avg_neg, 'margin': avg_pos - avg_neg}

def draw_plot(steps, losses, pos, neg, margins, lrs):
    if not steps:
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0,0].plot(steps, losses, 'k-', lw=1.5)
    ax[0,0].set_title('InfoNCE Loss')
    ax[0,0].grid(True, alpha=0.3)
    ax[0,1].plot(steps, pos, 'b-', marker='o', ms=3, label='Pos Sim')
    ax[0,1].plot(steps, neg, 'r-', marker='x', ms=3, label='Neg Sim')
    ax[0,1].set_title('Cosine Similarity')
    ax[0,1].legend()
    ax[0,1].grid(True, alpha=0.3)
    ax[1,0].plot(steps, margins, 'g-', marker='s', ms=3)
    ax[1,0].axhline(y=0.3, color='orange', ls='--', alpha=0.5)
    ax[1,0].set_title('Margin (Pos - Neg)')
    ax[1,0].grid(True, alpha=0.3)
    ax[1,1].plot(steps, lrs, 'm-', marker='d', ms=2)
    ax[1,1].set_title('Learning Rate')
    ax[1,1].grid(True, alpha=0.3)
    ax[1,1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()

def save_checkpoint(path, model, opt, step, info=""):
    tmp = path + '.tmp'
    torch.save({
        'step': step,
        'info': info,
        'model_state_dict': {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()},
        'opt_state_dict': opt.state_dict()
    }, tmp)
    os.replace(tmp, path)
    print(f"💾 Checkpoint saved: {path} (step {step})")

# ==================== MAIN ====================
def main():
    print(f"🔧 Device: {DEVICE}, dtype: {DTYPE}, WS={WINDOW_SIZE}", file=sys.stderr)
    print(f"📁 Logs: {CUR_LOG_DIR}", file=sys.stderr)
    print(f"⚙️ Eff. batch size: {BATCH_SIZE * ACCUMULATION_STEPS}, Negatives: {N_NEGATIVES}+{N_HARD_NEGATIVES}", file=sys.stderr)

    # 1. Загружаем валидационные ID
    if not os.path.exists(VAL_IDS_PATH):
        print(f"❌ {VAL_IDS_PATH} not found", file=sys.stderr)
        sys.exit(1)
    with open(VAL_IDS_PATH, 'r') as f:
        val_ids = set(json.load(f))
    print(f"📋 Val books: {len(val_ids)}", file=sys.stderr)

    # 2. Сканируем пути (только первые MAX_CHUNKS_PER_BOOK глав)
    print("📂 Scanning books...", file=sys.stderr)
    all_paths = scan_books_paths(DATA_ROOT)
    all_ids = set(all_paths.keys())
    train_ids = list(all_ids - val_ids)
    print(f"📚 Train books: {len(train_ids):,}, Val books: {len(val_ids):,}", file=sys.stderr)

    # 3. Загружаем книги в RAM (только train и val, многопоточно)
    print("📥 Loading train books to RAM (multithreaded)...", file=sys.stderr)
    train_books = load_books_to_ram_fast(train_ids, all_paths, WINDOW_SIZE, PAD_TOKEN_ID, max_workers=8)
    print("📥 Loading val books to RAM...", file=sys.stderr)
    val_books = load_books_to_ram_fast(list(val_ids), all_paths, WINDOW_SIZE, PAD_TOKEN_ID, max_workers=8)

    del all_paths
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Датасеты и DataLoader
    train_ds = ContrastiveDataset(train_books, N_NEGATIVES, N_HARD_NEGATIVES, is_val=False)
    val_ds = ContrastiveDataset(val_books, 1, 0, is_val=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 5. Модель (mean pooling, размороженные эмбеддинги)
    model = RAGEncoder(
        dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
        ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
        output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN,
        freeze_embeddings=False,   # размораживаем эмбеддинги
        pooling='mean'
    ).to(DEVICE)

    # Загружаем предобученные эмбеддинги и веса
    if os.path.exists(CHECKPOINT_PATH):
        model.load_embeddings(CHECKPOINT_PATH, freeze=False)
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items() if 'tok_emb' not in k}
        model.load_state_dict(cleaned, strict=False)
        print(f"✅ Model loaded from {CHECKPOINT_PATH}", file=sys.stderr)
    else:
        print(f"⚠️ {CHECKPOINT_PATH} not found, starting from scratch", file=sys.stderr)

    model.enable_gradient_checkpointing(False)
    model.train()

    # 6. Оптимизатор с разными LR
    embed_params = model.tok_emb.parameters()
    other_params = [p for n, p in model.named_parameters() if 'tok_emb' not in n]
    opt = torch.optim.AdamW([
        {'params': embed_params, 'lr': EMBED_LR},
        {'params': other_params, 'lr': BASE_LR}
    ], betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=(DEVICE == 'cuda'))
    scaler = GradScaler('cuda', enabled=(DTYPE == torch.float16))

    total_opt_steps = (len(train_dl) * NUM_EPOCHS) // ACCUMULATION_STEPS
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(1, total_opt_steps - WARMUP_STEPS)
            return 0.5 * (1 + math.cos(math.pi * progress))

    # 7. Цикл обучения
    steps, losses, pos_sims, neg_sims, margins, lrs_log = [], [], [], [], [], []
    v_hist = {'step': [], 'loss': [], 'pos_sim': [], 'neg_sim': [], 'margin': []}

    global_step = 0  # количество обработанных примеров (не шагов оптимизатора)
    opt_step = 0
    accumulated_anchors = []
    accumulated_positives = []
    t0 = time.time()

    # Начальная валидация
    print("🔍 Initial validation...", file=sys.stderr)
    val_metrics = validate(model, val_dl, TEMPERATURE, DEVICE, DTYPE)
    v_hist['step'].append(0)
    for k in val_metrics:
        v_hist[k].append(val_metrics[k])
    print(f"📊 Step 0: Loss={val_metrics['loss']:.3f}, Margin={val_metrics['margin']:.3f}", file=sys.stderr)

    try:
        pbar = tqdm(train_dl, desc="Training", dynamic_ncols=True, file=sys.stdout)
        opt.zero_grad(set_to_none=True)

        for epoch in range(NUM_EPOCHS):
            for anchor, positive, negs in pbar:
                # anchor, positive: [B, L]; negs: [B, N_total, L] – не используем, т.к. используем in-batch
                anchor = anchor.to(DEVICE, non_blocking=True)
                positive = positive.to(DEVICE, non_blocking=True)
                with autocast('cuda', dtype=DTYPE):
                    za = model(anchor)
                    zp = model(positive)
                    za = F.normalize(za, p=2, dim=-1)
                    zp = F.normalize(zp, p=2, dim=-1)
                    accumulated_anchors.append(za)
                    accumulated_positives.append(zp)

                if len(accumulated_anchors) == ACCUMULATION_STEPS:
                    anchors_batch = torch.cat(accumulated_anchors, dim=0)  # [ACC, D]
                    positives_batch = torch.cat(accumulated_positives, dim=0)
                    loss, pos_sim, neg_sim = in_batch_infoNCE(anchors_batch, positives_batch, TEMPERATURE)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                    # LR scheduler
                    lr_factor = lr_schedule(opt_step)
                    for param_group in opt.param_groups:
                        if param_group['params'] == embed_params:
                            param_group['lr'] = EMBED_LR * lr_factor
                        else:
                            param_group['lr'] = BASE_LR * lr_factor
                    opt_step += 1
                    global_step += ACCUMULATION_STEPS

                    steps.append(global_step)
                    losses.append(loss.item())
                    pos_sims.append(pos_sim)
                    neg_sims.append(neg_sim)
                    margins.append(pos_sim - neg_sim)
                    lrs_log.append(opt.param_groups[1]['lr'])  # base LR

                    with open(LOSS_LOG, 'a') as f:
                        f.write(json.dumps({
                            'step': global_step,
                            'loss': loss.item(),
                            'pos_sim': pos_sim,
                            'neg_sim': neg_sim,
                            'lr': opt.param_groups[1]['lr']
                        }) + '\n')

                    accumulated_anchors = []
                    accumulated_positives = []
                    torch.cuda.empty_cache()

                    # Валидация
                    if opt_step % (VAL_INTERVAL // ACCUMULATION_STEPS) == 0:
                        val_metrics = validate(model, val_dl, TEMPERATURE, DEVICE, DTYPE)
                        v_hist['step'].append(global_step)
                        for k in val_metrics:
                            v_hist[k].append(val_metrics[k])
                        with open(METRICS_LOG, 'a') as f:
                            f.write(json.dumps({'step': global_step, **val_metrics}) + '\n')
                        pbar.set_postfix(val_loss=f"{val_metrics['loss']:.3f}", val_margin=f"{val_metrics['margin']:.3f}")

                    # Рисуем график
                    if opt_step % (PLOT_INTERVAL // ACCUMULATION_STEPS) == 0:
                        draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs_log)

                # Ограничение памяти: если занято >14GB, чистим
                if torch.cuda.memory_allocated() > 14e9:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Сохраняем финальную модель
        save_checkpoint(MODEL_PATH, model, opt, global_step, 'Training completed')
        print(f"\n🎉 Training finished. Model saved to {MODEL_PATH}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted, saving checkpoint...", file=sys.stderr)
        save_checkpoint(os.path.join(CUR_LOG_DIR, 'interrupted.pth'), model, opt, global_step, 'Interrupted')
    finally:
        draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs_log)
        elapsed = (time.time() - t0) / 60
        print(f"⏱️ Time: {elapsed:.1f} min | Steps: {global_step}", file=sys.stderr)

if __name__ == '__main__':
    main()