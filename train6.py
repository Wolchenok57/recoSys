# train7.py — MarginMSE + Hard Negatives + Логирование + Графики
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, random, time, math, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

from model import RAGEncoder, VOCAB_SIZE, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN, BOS_TOKEN_ID, PAD_TOKEN_ID
from testoBPE import BPE

# ==================== КОНСТАНТЫ ====================
MAX_LEN = 2048
BS = 1
ACCUMULATION_STEPS = 8
TEMPERATURE = 0.1

BASE_LR = 2e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 200

EPOCHS = 10

PREV_LOG_DIR = "logs5"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")
# Пользователь указал использовать logs6 для логов
CUR_LOG_DIR = "logs6" 

LLM_DATASETS_ROOT = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets"
SYNTHETIC_PATH = os.path.join(LLM_DATASETS_ROOT, "My_RAG_DS/data.jsonl")
PARAPHRASER_PATH = os.path.join(LLM_DATASETS_ROOT, "ru_paraphraser/train.jsonl")

PLOT_INTERVAL = 100
CHECKPOINT_INTERVAL = 500

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16

os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss.jsonl')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model.pth")
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training.png')

open(LOSS_LOG, 'w').close()

tok = BPE()

# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_synthetic(limit=50000):
    """Загружает синтетику: (Query, Context)"""
    pairs = []
    if not os.path.exists(SYNTHETIC_PATH):
        print(f"⚠️ Файл синтетики не найден: {SYNTHETIC_PATH}")
        return pairs
    
    with open(SYNTHETIC_PATH, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    
    print(f"📖 Синтетика: найдено {len(lines)} записей. Берем {min(limit, len(lines))}...")
    random.shuffle(lines)
    
    for item in lines[:limit]:
        q = tok.encode(item['query'])[:MAX_LEN]
        c = tok.encode(item['context'])[:MAX_LEN]
        # Паддинг
        q = q + [PAD_TOKEN_ID] * (MAX_LEN - len(q))
        c = c + [PAD_TOKEN_ID] * (MAX_LEN - len(c))
        
        pairs.append((torch.tensor(q, dtype=torch.long), 
                      torch.tensor(c, dtype=torch.long)))
    return pairs

def load_paraphraser_with_hard_negatives(limit_pairs=10000, limit_negs=20000):
    """Загружает ParaPhraser: (Text, Paraphrase) и Пул Негативов (Class -1)"""
    pos_pairs = []
    hard_neg_pool = []
    
    if not os.path.exists(PARAPHRASER_PATH):
        return pos_pairs, hard_neg_pool
    
    print("📖 Чтение ParaPhraser...")
    count_pos = 0
    count_neg = 0
    
    with open(PARAPHRASER_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if count_pos >= limit_pairs and count_neg >= limit_negs:
                break
            
            item = json.loads(line)
            cls = int(item['class'])
            
            t1 = tok.encode(item['text_1'])[:MAX_LEN]
            t2 = tok.encode(item['text_2'])[:MAX_LEN]
            t1 = t1 + [PAD_TOKEN_ID] * (MAX_LEN - len(t1))
            t2 = t2 + [PAD_TOKEN_ID] * (MAX_LEN - len(t2))
            
            if cls in [0, 1]:
                pos_pairs.append((torch.tensor(t1, dtype=torch.long),
                                  torch.tensor(t2, dtype=torch.long)))
                count_pos += 1
            elif cls == -1:
                hard_neg_pool.append(torch.tensor(t1, dtype=torch.long))
                hard_neg_pool.append(torch.tensor(t2, dtype=torch.long))
                count_neg += 2
                
    print(f"   ✅ Позитивных пар: {len(pos_pairs)}")
    print(f"   ✅ Пул сложных негативов: {len(hard_neg_pool)}")
    return pos_pairs, hard_neg_pool

# ==================== DATASET ====================
class MarginDataset(Dataset):
    def __init__(self, synth_pairs, para_pairs, hard_neg_pool):
        self.data = []
        self.data.extend([('synth', a, p) for a, p in synth_pairs])
        self.data.extend([('para', a, p) for a, p in para_pairs])
        random.shuffle(self.data)
        
        self.hard_neg_pool = hard_neg_pool
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        _, anchor, positive = self.data[idx]
        
        if len(self.hard_neg_pool) > 0:
            neg = random.choice(self.hard_neg_pool)
        else:
            neg = positive 
            
        return anchor, positive, neg

# ==================== MAIN ====================
def main():
    print(f"🔥 ОБУЧЕНИЕ logs7 (логи в logs6): MarginMSE + Hard Negatives")
    print(f"💾 Устройство: {DEVICE} | {DTYPE} | MAX_LEN={MAX_LEN}")
    
    # 1. Загрузка
    synth_pairs = load_synthetic(limit=50000)
    para_pairs, hard_neg_pool = load_paraphraser_with_hard_negatives()
    
    if len(synth_pairs) == 0 and len(para_pairs) == 0:
        print("❌ Нет данных для обучения!")
        return

    dataset = MarginDataset(synth_pairs, para_pairs, hard_neg_pool)
    print(f"📦 Всего пар для обучения: {len(dataset)}")
    
    dl = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0, pin_memory=(DEVICE=='cuda'))
    
    # 2. Модель
    print(f"\n📦 Загрузка модели из {CHECKPOINT_PATH}...")
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        if DTYPE == torch.float16:
            for k, v in cleaned.items():
                if v.dtype in [torch.float32, torch.bfloat16]: cleaned[k] = v.half()
        model.load_state_dict(cleaned, strict=False)
        print("✅ Веса загружены")
    
    model.enable_gradient_checkpointing(True)
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=True)
    scaler = GradScaler('cuda', enabled=True)
    
    steps_per_epoch = len(dataset) // BS // ACCUMULATION_STEPS
    total_steps = steps_per_epoch * EPOCHS
    get_lr = lambda s: BASE_LR*(s+1)/WARMUP_STEPS if s<WARMUP_STEPS else MIN_LR+(BASE_LR-MIN_LR)*0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/max(1,total_steps-WARMUP_STEPS)))
    
    # Списки для графика
    steps, losses, pos_sims, neg_sims, margins, lrs = [], [], [], [], [], []
    
    def margin_mse_loss(anchor, positive, negative, target_pos=1.0, target_neg=0.0):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)
        
        sim_pos = (anchor * positive).sum(dim=-1)
        sim_neg = (anchor * negative).sum(dim=-1)
        
        loss_pos = F.mse_loss(sim_pos, torch.full_like(sim_pos, target_pos))
        loss_neg = F.mse_loss(sim_neg, torch.full_like(sim_neg, target_neg))
        
        loss = loss_pos + loss_neg
        return loss, sim_pos.mean().item(), sim_neg.mean().item()
    
    def draw_plot(global_step):
        if not steps: return
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        ax[0].plot(steps, losses, 'k-')
        ax[0].set_title(f'Loss (Step {global_step})')
        ax[0].grid(alpha=0.3)
        
        ax[1].plot(steps, pos_sims, 'b-', label='Pos Sim')
        ax[1].plot(steps, neg_sims, 'r-', label='Neg Sim')
        ax[1].set_title('Similarity')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        
        ax[2].plot(steps, margins, 'g-')
        ax[2].set_title('Margin')
        ax[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
        plt.close()

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🔄 Эпоха {epoch}/{EPOCHS}")
        accum_step = 0
        opt.zero_grad(set_to_none=True)
        
        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False)
        for anchor_cpu, pos_cpu, neg_cpu in pbar:
            # 🔥 ИСПРАВЛЕНИЕ: Убираем лишние unsqueeze, DataLoader уже возвращает [1, L]
            anc = anchor_cpu.to(DEVICE, non_blocking=True)      # [1, L]
            pos = pos_cpu.to(DEVICE, non_blocking=True)         # [1, L]
            neg = neg_cpu.to(DEVICE, non_blocking=True)         # [1, L]
            
            anc_mask = (anc != 0).long()
            pos_mask = (pos != 0).long()
            neg_mask = (neg != 0).long()
            
            with autocast('cuda', dtype=DTYPE):
                za = model(anc, attention_mask=anc_mask)
                zp = model(pos, attention_mask=pos_mask)
                zn = model(neg, attention_mask=neg_mask)
                
                loss, ps, ns = margin_mse_loss(za, zp, zn)
            
            loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            accum_step += 1
            
            if accum_step % ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                
                cur_lr = get_lr(global_step)
                for pg in opt.param_groups: pg['lr'] = cur_lr
                
                cl = loss.item() * ACCUMULATION_STEPS
                margin_val = ps - ns
                
                steps.append(global_step)
                losses.append(cl)
                pos_sims.append(ps)
                neg_sims.append(ns)
                margins.append(margin_val)
                lrs.append(cur_lr)
                
                # Логирование
                with open(LOSS_LOG, 'a') as f:
                    f.write(json.dumps({'step': global_step, 'loss': cl, 'pos': ps, 'neg': ns, 'lr': cur_lr}) + '\n')
                
                # Графики
                if global_step % PLOT_INTERVAL == 0:
                    draw_plot(global_step)
                
                # Чекпоинты
                if global_step % CHECKPOINT_INTERVAL == 0:
                    ckpt_path = os.path.join(CUR_LOG_DIR, f"step_{global_step}.pth")
                    torch.save({'step': global_step, 'epoch': epoch,
                               'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                               'opt_state_dict': opt.state_dict()}, ckpt_path)
                    print(f"💾 Checkpoint saved at step {global_step}")
                
                pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{margin_val:.3f}", pos=f"{ps:.2f}", neg=f"{ns:.2f}")
                global_step += 1
            
            del anc, pos, neg, za, zp, zn, loss, anc_mask, pos_mask, neg_mask
        
        # Сохранение в конце эпохи
        epoch_path = os.path.join(CUR_LOG_DIR, f"epoch_{epoch}.pth")
        torch.save({'epoch': epoch, 'step': global_step, 
                   'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                   'opt_state_dict': opt.state_dict()}, epoch_path)
        print(f"💾 Epoch {epoch} saved: {epoch_path}")
        
        # Финальный график эпохи
        draw_plot(global_step)
        
        gc.collect(); torch.cuda.empty_cache()
    
    # Финальное сохранение
    torch.save({'step': global_step, 'epoch': EPOCHS,
               'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
               'opt_state_dict': opt.state_dict()}, MODEL_PATH)
    
    print(f"\n🏁 Обучение завершено! Финальная модель: {MODEL_PATH}")

if __name__ == '__main__':
    main()