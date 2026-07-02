#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train5.py — Оптимизированная версия: MAX_LEN=512, BS=8, без checkpointing, pin_memory=True
"""

import os, json, random, time, math, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from testoBPE import BPE
from model import RAGEncoder, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN, PAD_TOKEN_ID

# ==================== КОНСТАНТЫ ====================
MAX_LEN = 600
BS = 1
ACCUMULATION_STEPS = 1
N_NEGATIVES = 3
TEMPERATURE = 0.1

BASE_LR = 2e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 300

EPOCHS = 3
VAL_INTERVAL = 300
PLOT_INTERVAL = 150
CHECKPOINT_INTERVAL = 500

PREV_LOG_DIR = "logs4"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")
CUR_LOG_DIR = "logs5"

MY_RAG_DS = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/My_RAG_DS/data.jsonl"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16
torch.backends.cudnn.benchmark = True

# ==================== НАСТРОЙКА ====================
os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss.jsonl')
METRICS_LOG = os.path.join(CUR_LOG_DIR, 'metrics.jsonl')
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training.png')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model.pth")
for f in [LOSS_LOG, METRICS_LOG]:
    open(f, 'w').close()

tok = BPE()

# ==================== DATASET ====================

def load_my_rag_ds(path, max_len, split_ratio=0.9):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q = tok.encode(item['query'])[:max_len]
            c = tok.encode(item['context'])[:max_len]
            q = q + [PAD_TOKEN_ID] * (max_len - len(q))
            c = c + [PAD_TOKEN_ID] * (max_len - len(c))
            pairs.append((torch.tensor(q), torch.tensor(c)))
    
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"✅ Загружено: {len(pairs)} пар | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    return train_pairs, val_pairs

class SimpleContrastiveDataset(Dataset):
    def __init__(self, pairs, n_negatives=3):
        self.pairs = pairs
        self.n_negatives = n_negatives
        self.indices = list(range(len(pairs)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, positive = self.pairs[idx]
        neg_indices = random.sample([i for i in self.indices if i != idx], min(self.n_negatives, len(self.pairs)-1))
        negatives = torch.stack([self.pairs[i][1] for i in neg_indices])
        return anchor, positive, negatives

# ==================== MAIN ====================

def main():
    print(f"🔥 ОБУЧЕНИЕ: только My_RAG_DS → {CUR_LOG_DIR}")
    print(f"💾 Устройство: {DEVICE} | {DTYPE} | MAX_LEN={MAX_LEN} | BS={BS}")
    
    if not os.path.exists(MY_RAG_DS):
        print(f"❌ Файл не найден: {MY_RAG_DS}")
        return
    
    train_pairs, val_pairs = load_my_rag_ds(MY_RAG_DS, MAX_LEN)
    
    train_ds = SimpleContrastiveDataset(train_pairs, N_NEGATIVES)
    val_ds = SimpleContrastiveDataset(val_pairs, 1)
    
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    
    print("📦 Загрузка модели...")
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        if DTYPE == torch.float16:
            for k, v in cleaned.items():
                if v.dtype in [torch.float32, torch.bfloat16]:
                    cleaned[k] = v.half()
        model.load_state_dict(cleaned, strict=False)
        print(f"✅ Веса загружены из {CHECKPOINT_PATH}")
    else:
        print(f"⚠️ Чекпоинт {CHECKPOINT_PATH} не найден, инициализация с нуля")
    
    model.enable_gradient_checkpointing(False)  # ✅ Отключаем для скорости
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=True)
    scaler = GradScaler('cuda', enabled=True)
    
    total_steps = len(train_dl) * EPOCHS // ACCUMULATION_STEPS
    get_lr = lambda s: BASE_LR*(s+1)/WARMUP_STEPS if s<WARMUP_STEPS else MIN_LR+(BASE_LR-MIN_LR)*0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/max(1,total_steps-WARMUP_STEPS)))
    
    steps, losses, pos_sims, neg_sims, margins, lrs = [], [], [], [], [], []
    
    def info_nce_loss(anchor, positive, negatives, temperature=TEMPERATURE):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)
        
        pos_sim = (anchor * positive).sum(dim=-1) / temperature
        neg_sim = torch.einsum('bd,bnd->bn', anchor, negatives) / temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss, (anchor*positive).sum(dim=-1).mean().item(), torch.einsum('bd,bnd->bn', anchor, negatives).mean().item()
    
    @torch.no_grad()
    def validate(model, val_loader):
        model.eval()
        t_loss, t_pos, t_neg, n = 0.0, 0.0, 0.0, 0
        for anchor, pos, negs in val_loader:
            anc = anchor.to(DEVICE, non_blocking=True)
            p = pos.to(DEVICE, non_blocking=True)
            ns = negs.to(DEVICE, non_blocking=True)
            
            anc_mask = (anc != 0).long()
            pos_mask = (p != 0).long()
            
            B, N, L = ns.shape
            ns_flat = ns.view(B * N, L)
            neg_mask = (ns_flat != 0).long()
            
            with autocast('cuda', dtype=DTYPE):
                za = model(anc, attention_mask=anc_mask)
                zp = model(p, attention_mask=pos_mask)
                zn_flat = model(ns_flat, attention_mask=neg_mask)
                zn = zn_flat.view(B, N, -1)
                loss, ps, nsm = info_nce_loss(za, zp, zn)
            
            t_loss += loss.item()
            t_pos += ps
            t_neg += nsm
            n += 1
        model.train()
        if n == 0:
            return {'loss': 0.0, 'pos_sim': 0.0, 'neg_sim': 0.0, 'margin': 0.0}
        return {'loss': t_loss/n, 'pos_sim': t_pos/n, 'neg_sim': t_neg/n, 'margin': (t_pos-t_neg)/n}
    
    def draw_plot(steps, losses, pos, neg, margins, lrs):
        if not steps:
            return
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0,0].plot(steps, losses, 'k-')
        ax[0,0].set_title('Loss')
        ax[0,0].grid(alpha=0.3)
        ax[0,1].plot(steps, pos, 'b-', label='Pos')
        ax[0,1].plot(steps, neg, 'r-', label='Neg')
        ax[0,1].set_title('Similarity')
        ax[0,1].legend()
        ax[0,1].grid(alpha=0.3)
        ax[1,0].plot(steps, margins, 'g-')
        ax[1,0].set_title('Margin')
        ax[1,0].grid(alpha=0.3)
        ax[1,1].plot(steps, lrs, 'm-')
        ax[1,1].set_title('LR')
        ax[1,1].grid(alpha=0.3)
        ax[1,1].set_yscale('log')
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
        plt.close()
    
    gs, os_ = 0, 0
    accum_step = 0
    t0 = time.time()
    plot_saved = False
    
    print(f"\n🚀 Начало: {EPOCHS} эпох, ~{total_steps} шагов")
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    opt.zero_grad(set_to_none=True)
    
    try:
        for epoch in range(EPOCHS):
            print(f"\n🔄 Эпоха {epoch+1}/{EPOCHS}")
            for anchor_cpu, pos_cpu, negs_cpu in train_dl:
                # ✅ Маски создаём ДО autocast
                anc_mask = (anchor_cpu != 0).long().to(DEVICE, non_blocking=True)
                pos_mask = (pos_cpu != 0).long().to(DEVICE, non_blocking=True)
                
                anc = anchor_cpu.to(DEVICE, non_blocking=True)
                pos = pos_cpu.to(DEVICE, non_blocking=True)
                negs = negs_cpu.to(DEVICE, non_blocking=True)
                
                B, N, L = negs.shape
                negs_flat = negs.view(B * N, L)
                neg_mask = (negs_flat != 0).long()
                
                with autocast('cuda', dtype=DTYPE):
                    za = model(anc, attention_mask=anc_mask)
                    zp = model(pos, attention_mask=pos_mask)
                    zn_flat = model(negs_flat, attention_mask=neg_mask)
                    zn = zn_flat.view(B, N, -1)
                    loss, ps, ns = info_nce_loss(za, zp, zn)
                
                loss = loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                accum_step += 1
                
                if accum_step % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    
                    cur_lr = get_lr(os_)
                    for pg in opt.param_groups:
                        pg['lr'] = cur_lr
                    os_ += 1
                    gs += 1
                    
                    cl = loss.item() * ACCUMULATION_STEPS
                    steps.append(gs)
                    losses.append(cl)
                    pos_sims.append(ps)
                    neg_sims.append(ns)
                    margins.append(ps-ns)
                    lrs.append(cur_lr)
                    
                    with open(LOSS_LOG, 'a') as f:
                        f.write(json.dumps({'step': gs, 'loss': cl, 'pos': ps, 'neg': ns, 'lr': cur_lr}) + '\n')
                    
                    if gs % VAL_INTERVAL == 0:
                        vm = validate(model, val_dl)
                        with open(METRICS_LOG, 'a') as f:
                            f.write(json.dumps({'step': gs, **vm}) + '\n')
                        pbar.set_postfix(val_loss=f"{vm['loss']:.3f}", val_margin=f"{vm['margin']:.3f}")
                    
                    if gs % PLOT_INTERVAL == 0:
                        draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
                        plot_saved = True
                    
                    if gs % CHECKPOINT_INTERVAL == 0:
                        torch.save({
                            'step': gs,
                            'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                            'opt_state_dict': opt.state_dict()
                        }, MODEL_PATH)
                    
                    elapsed = time.time() - t0
                    it_s = gs / elapsed if elapsed > 0 else 0
                    pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", it_s=f"{it_s:.2f}")
                    pbar.update(1)
                
                del anc, pos, negs, za, zp, zn, loss, anc_mask, pos_mask, neg_mask
            
            gc.collect()
            torch.cuda.empty_cache()
        
        pbar.close()
        print("\n🏁 Обучение завершено.")
        torch.save({
            'step': gs,
            'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
            'opt_state_dict': opt.state_dict()
        }, MODEL_PATH)
        print(f"💾 Модель: {MODEL_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not plot_saved and steps:
            draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
        elapsed = time.time() - t0
        print(f"⏱️ Время: {elapsed/60:.1f} мин | Шагов: {gs}")
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()