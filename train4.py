#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, time, math
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
MAX_LEN = 128
BS = 8
ACCUMULATION_STEPS = 4
N_NEGATIVES = 3
TEMPERATURE = 0.1

BASE_LR = 2e-5
MIN_LR = 1e-6
WARMUP_STEPS = 200

PREV_LOG_DIR = "logs3_test_run"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model_final.pth")
CUR_LOG_DIR = "logs4"  # ← ЖЁСТКО ЗАФИКСИРОВАНО

DATA_PATH = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/ru_paraphraser/train.jsonl"

PLOT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000

DEVICE = 'cuda'
DTYPE = torch.float16

# ==================== НАСТРОЙКА ====================
os.makedirs(CUR_LOG_DIR, exist_ok=True)
tok = BPE()

# ==================== DATASET ====================
class ParaPhraserDataset(Dataset):
    def __init__(self, file_path, max_len):
        self.max_len = max_len
        self.positive_pairs = []
        self.all_texts = []
        self.hard_negatives = []
        
        stats = {1: 0, 0: 0, -1: 0, "unknown": 0}
        
        print("📖 Чтение train.jsonl...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Загрузка"):
                item = json.loads(line)
                t1 = item['text_1']
                t2 = item['text_2']
                
                # 🔥 ИСПРАВЛЕНИЕ: JSON хранит class как строку, конвертируем в число
                cls = int(item['class'])
                
                tok1 = tok.encode(t1)[:max_len]
                tok2 = tok.encode(t2)[:max_len]
                
                tok1 = tok1 + [PAD_TOKEN_ID] * (max_len - len(tok1))
                tok2 = tok2 + [PAD_TOKEN_ID] * (max_len - len(tok2))
                
                self.all_texts.append(tok1)
                self.all_texts.append(tok2)
                
                if cls == 1:
                    self.positive_pairs.append((tok1, tok2, 1.0))
                    stats[1] += 1
                elif cls == 0:
                    self.positive_pairs.append((tok1, tok2, 0.5))
                    stats[0] += 1
                elif cls == -1:
                    self.hard_negatives.append(tok1)
                    self.hard_negatives.append(tok2)
                    stats[-1] += 1
                else:
                    stats["unknown"] += 1
        
        print(f"📊 Статистика классов: {stats}")
        print(f"✅ Позитивных пар (class 0/1): {len(self.positive_pairs):,}")
        print(f"✅ Хард-негативов (class -1): {len(self.hard_negatives):,}")

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        t1, t2, weight = self.positive_pairs[idx]
        anchor = torch.tensor(t1, dtype=torch.long)
        positive = torch.tensor(t2, dtype=torch.long)
        
        negatives = []
        # 50% случайные, 50% хард-негативы
        for _ in range(N_NEGATIVES // 2):
            rand_text = random.choice(self.all_texts)
            negatives.append(torch.tensor(rand_text, dtype=torch.long))
        
        for _ in range(N_NEGATIVES - N_NEGATIVES // 2):
            if self.hard_negatives:
                hard_text = random.choice(self.hard_negatives)
                negatives.append(torch.tensor(hard_text, dtype=torch.long))
            else:
                rand_text = random.choice(self.all_texts)
                negatives.append(torch.tensor(rand_text, dtype=torch.long))
        
        negatives = torch.stack(negatives)
        return anchor, positive, negatives, weight

# ==================== MAIN ====================
def main():
    print(f"🔥 ДОБУЧЕНИЕ: ParaPhraser (logs4, weighted loss, FIXED)")
    print(f"💾 Устройство: {DEVICE} | {DTYPE}")
    print(f"📂 Логи: {CUR_LOG_DIR}")
    
    dataset = ParaPhraserDataset(DATA_PATH, MAX_LEN)
    
    if len(dataset) == 0:
        print("❌ ОШИБКА: Нет позитивных пар для обучения! Проверь данные.")
        return
    
    total = len(dataset)
    dl = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
    
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
                if v.dtype in [torch.float32, torch.bfloat16]: cleaned[k] = v.half()
        model.load_state_dict(cleaned, strict=False)
        print("✅ Веса загружены")
    
    model.enable_gradient_checkpointing(False)
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    scaler = GradScaler('cuda', enabled=True)
    
    total_steps = (total // BS // ACCUMULATION_STEPS) * 3
    get_lr = lambda s: BASE_LR*(s+1)/WARMUP_STEPS if s<WARMUP_STEPS else MIN_LR+(BASE_LR-MIN_LR)*0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/max(1,total_steps-WARMUP_STEPS)))
    
    steps, losses, pos_sims, neg_sims, margins = [], [], [], [], []
    
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
    
    def draw_plot(steps, losses, pos, neg, margins):
        if not steps: return
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].plot(steps, losses, 'k-'); ax[0].set_title('Loss'); ax[0].grid(alpha=0.3)
        ax[1].plot(steps, pos, 'b-', label='Pos'); ax[1].plot(steps, neg, 'r-', label='Neg')
        ax[1].set_title('Similarity'); ax[1].legend(); ax[1].grid(alpha=0.3)
        ax[2].plot(steps, margins, 'g-'); ax[2].set_title('Margin'); ax[2].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(CUR_LOG_DIR, 'training.png'), dpi=150, bbox_inches='tight'); plt.close()

    gs, os_ = 0, 0
    accum_step = 0
    t0 = time.time()
    plot_saved = False
    
    pbar = tqdm(total=total_steps, desc="TRAINING (ParaPhraser)", dynamic_ncols=True)
    opt.zero_grad(set_to_none=True)
    
    try:
        for epoch in range(3):
            print(f"\n🔄 Эпоха {epoch+1}/3")
            for anchor_cpu, pos_cpu, negs_cpu, weights in dl:
                anc = anchor_cpu.to(DEVICE, non_blocking=True)      # [BS, L]
                pos = pos_cpu.to(DEVICE, non_blocking=True)         # [BS, L]
                negs = negs_cpu.to(DEVICE, non_blocking=True)       # [BS, N_NEG, L]
                w = weights.to(DEVICE, non_blocking=True)           # [BS]
                
                anc_mask = (anc != 0).long()
                pos_mask = (pos != 0).long()
                
                # 🔥 ИСПРАВЛЕНИЕ ФОРМЫ: Разворачиваем негативы для модели
                B, N, L = negs.shape
                negs_flat = negs.view(B * N, L)                     # [BS*N, L]
                neg_mask = (negs_flat != 0).long()                  # [BS*N, L]
                
                with autocast('cuda', dtype=DTYPE):
                    za = model(anc, attention_mask=anc_mask)        # [BS, D]
                    zp = model(pos, attention_mask=pos_mask)        # [BS, D]
                    zn_flat = model(negs_flat, attention_mask=neg_mask) # [BS*N, D]
                    zn = zn_flat.view(B, N, -1)                     # [BS, N, D]
                    
                    loss, ps, ns = info_nce_loss(za, zp, zn)
                    loss = loss * w.mean()
                
                loss = loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                accum_step += 1
                
                if accum_step % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    
                    cur_lr = get_lr(os_)
                    for pg in opt.param_groups: pg['lr'] = cur_lr
                    os_ += 1
                    gs += 1
                    
                    cl = loss.item() * ACCUMULATION_STEPS / w.mean().item()
                    steps.append(gs); losses.append(cl); pos_sims.append(ps); neg_sims.append(ns); margins.append(ps-ns)
                    
                    if gs % PLOT_INTERVAL == 0:
                        draw_plot(steps, losses, pos_sims, neg_sims, margins)
                        plot_saved = True
                    if gs % CHECKPOINT_INTERVAL == 0:
                        torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()}}, 
                                  os.path.join(CUR_LOG_DIR, "model.pth"))
                    
                    elapsed = time.time() - t0
                    it_s = gs / elapsed
                    pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", it_s=f"{it_s:.2f}")
                    pbar.update(1)
                
                del anc, pos, negs, za, zp, zn, loss, anc_mask, pos_mask, neg_mask, w
            
        pbar.close()
        print("\n🏁 Обучение завершено.")
        torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()}}, 
                  os.path.join(CUR_LOG_DIR, "model.pth"))
        print(f"💾 Модель сохранена: {os.path.join(CUR_LOG_DIR, 'model.pth')}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback; traceback.print_exc()
    finally:
        if not plot_saved and steps: draw_plot(steps, losses, pos_sims, neg_sims, margins)
        elapsed = time.time() - t0
        print(f"⏱️ Время: {elapsed/60:.1f} мин | Шагов: {gs} | Скорость: {elapsed/max(1,gs):.2f} s/it")
        if DEVICE == 'cuda': torch.cuda.empty_cache()

if __name__ == '__main__':
    main()