# train3_final_fast.py — MAX_LEN=2048, RoPE экстраполяция, минимум оверхеда
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, random, time, gc, faulthandler
faulthandler.enable()

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import pandas as pd
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
BS = 1
N_NEGATIVES = 1
ACCUMULATION_STEPS = 8
MAX_LEN = 2048  # 🔥 КЛЮЧЕВОЕ: 2048 вместо 4096 → 4× ускорение
LIMIT_PER_SOURCE = 100_000

BASE_LR = 2e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
TEMPERATURE = 0.1

PREV_LOG_DIR = "logs3"
CUR_LOG_DIR = "logs3_2"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")

DATA_ROOT = "/mnt/news/llm_ds/fics"
LLM_DATASETS_ROOT = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets"

PLOT_INTERVAL = 500
CHECKPOINT_INTERVAL = 2000

DEVICE = 'cuda'
DTYPE = torch.float16

# ==================== НАСТРОЙКА ПУТЕЙ ====================
os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss_final.jsonl')
METRICS_LOG = os.path.join(CUR_LOG_DIR, 'metrics_final.jsonl')
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training_final.png')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model_final.pth")
for f in [LOSS_LOG, METRICS_LOG]: open(f, 'w').close()

tok = BPE()

# ==================== БЫСТРАЯ ЗАГРУЗКА (на CPU, без лишнего) ====================
class FastDataset:
    def __init__(self, data_root, mixed_root, ws, limit_books, limit_mixed):
        self.data = []
        
        # Книги
        tqdm.write(" Загрузка книг...")
        root = Path(data_root)
        shards = sorted([d for d in root.iterdir() if d.is_dir() and '-' in d.name])
        count = 0
        for shard in tqdm(shards, desc="Books", leave=False):
            if count >= limit_books: break
            for book_folder in sorted(shard.iterdir()):
                if count >= limit_books: break
                if not book_folder.is_dir() or not book_folder.name.startswith('book_'): continue
                chaps = sorted([str(p) for p in book_folder.glob('chap*.npy')], key=lambda x: int(Path(x).stem.replace('chap', '')))
                if not chaps: continue
                book_arr = np.concatenate([np.array([BOS_TOKEN_ID], dtype=np.int64)] + [np.load(c).astype(np.int64) for c in chaps])
                n_chunks = max(1, len(book_arr) // ws)
                for i in range(n_chunks):
                    if count >= limit_books: break
                    start, end = i * ws, min((i+1) * ws, len(book_arr))
                    chunk = book_arr[start:end]
                    self.data.append((chunk, chunk.copy()))
                    count += 1
        
        # Смешанные
        tqdm.write(" Загрузка смешанных датасетов...")
        count = 0
        for src, path, col_a, col_b in [
            ('gazeta', os.path.join(mixed_root, "gazeta/default/train/0000.parquet"), 'summary', 'text'),
            ('samsum', os.path.join(mixed_root, "samsum-ru/data/train-00000-of-00001-76cc3fe8132d8f4b.parquet"), 'summary', 'dialogue'),
        ]:
            try:
                df = pd.read_parquet(path).head(limit_mixed)
                for _, r in df.iterrows():
                    if count >= limit_mixed: break
                    if r[col_a] and r[col_b]:
                        self.data.append((
                            np.array(tok.encode(str(r[col_a]))[:MAX_LEN], dtype=np.int64),
                            np.array(tok.encode(str(r[col_b]))[:MAX_LEN], dtype=np.int64)
                        ))
                        count += 1
            except: pass
        
        try:
            df = pd.read_csv(os.path.join(mixed_root, "xlsum-russian-bbc/bbcrussian.csv.gz"), compression="gzip", nrows=limit_mixed//2)
            for _, r in df.iterrows():
                if count >= limit_mixed: break
                if r['resume'] and r['news']:
                    self.data.append((
                        np.array(tok.encode(str(r['resume']))[:MAX_LEN], dtype=np.int64),
                        np.array(tok.encode(str(r['news']))[:MAX_LEN], dtype=np.int64)
                    ))
                    count += 1
        except: pass
        
        try:
            for fname in os.listdir(os.path.join(mixed_root, "RuSentEval/data")):
                if count >= limit_mixed: break
                if fname.endswith(".txt"):
                    with open(os.path.join(mixed_root, "RuSentEval/data", fname), "r", encoding="utf-8") as f:
                        for line in f:
                            if count >= limit_mixed: break
                            parts = line.strip().split("\t")
                            if len(parts) >= 3:
                                self.data.append((
                                    np.array(tok.encode(parts[1])[:MAX_LEN], dtype=np.int64),
                                    np.array(tok.encode(parts[2])[:MAX_LEN], dtype=np.int64)
                                ))
                                count += 1
        except: pass
        
        random.shuffle(self.data)
        tqdm.write(f"📦 Готово: {len(self.data):,} пар")
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        a_np, p_np = self.data[idx]
        return torch.from_numpy(a_np).long(), torch.from_numpy(p_np).long()

# ==================== MAIN ====================
def main():
    tqdm.write(f"🔥 ФИНАЛЬНЫЙ ЗАПУСК | MAX_LEN={MAX_LEN} (RoPE экстраполяция до 4096) | BS=1 | ACCUM={ACCUMULATION_STEPS}")
    tqdm.write(f"💾 Устройство: {DEVICE} | {DTYPE}")
    
    dataset = FastDataset(DATA_ROOT, LLM_DATASETS_ROOT, MAX_LEN, LIMIT_PER_SOURCE, LIMIT_PER_SOURCE)
    total = len(dataset)
    
    # Модель
    tqdm.write(" Инициализация модели...")
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
        tqdm.write(f"✅ Веса загружены")
    
    model.enable_gradient_checkpointing(True)
    # model.compile()
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=True)
    scaler = GradScaler('cuda', enabled=True)
    
    total_steps = total // ACCUMULATION_STEPS
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
    
    def draw_plot(steps, losses, pos, neg, margins, lrs):
        if not steps: return
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0,0].plot(steps, losses, 'k-'); ax[0,0].set_title('Loss'); ax[0,0].grid(alpha=0.3)
        ax[0,1].plot(steps, pos, 'b-', label='Pos'); ax[0,1].plot(steps, neg, 'r-', label='Neg')
        ax[0,1].set_title('Similarity'); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
        ax[1,0].plot(steps, margins, 'g-'); ax[1,0].set_title('Margin'); ax[1,0].grid(alpha=0.3)
        ax[1,1].plot(steps, lrs, 'm-'); ax[1,1].set_title('LR'); ax[1,1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight'); plt.close()

    gs, os_ = 0, 0
    accum_step = 0
    t0 = time.time()
    plot_saved = False
    indices = list(range(total))
    random.shuffle(indices)
    
    pbar = tqdm(total=total_steps, desc=f"TRAINING (2048→4096 RoPE)", dynamic_ncols=True)
    opt.zero_grad(set_to_none=True)
    
    try:
        for i in range(0, total, BS):
            idx = indices[i]
            a_cpu, p_cpu = dataset[idx]
            
            # Копирование на GPU (без pinned buffer для BS=1)
            anc = a_cpu.to(DEVICE, non_blocking=True).unsqueeze(0)
            pos = p_cpu.to(DEVICE, non_blocking=True).unsqueeze(0)
            anc_mask = (anc != 0).long()
            pos_mask = (pos != 0).long()
            
            # Негатив
            n_idx = random.choice(indices)
            n_cpu, _ = dataset[n_idx]
            neg = n_cpu.to(DEVICE, non_blocking=True).unsqueeze(0)
            neg_mask = (neg != 0).long()
            
            with autocast('cuda', dtype=DTYPE):
                za = model(anc, attention_mask=anc_mask)
                zp = model(pos, attention_mask=pos_mask)
                zn = model(neg, attention_mask=neg_mask).unsqueeze(0)
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
                for pg in opt.param_groups: pg['lr'] = cur_lr
                os_ += 1
                gs += 1
                
                cl = loss.item() * ACCUMULATION_STEPS
                steps.append(gs); losses.append(cl); pos_sims.append(ps); neg_sims.append(ns); margins.append(ps-ns); lrs.append(cur_lr)
                
                with open(LOSS_LOG, 'a') as f:
                    f.write(json.dumps({'step': gs, 'loss': cl, 'pos': ps, 'neg': ns, 'lr': cur_lr}) + '\n')
                
                if gs % PLOT_INTERVAL == 0:
                    draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
                    plot_saved = True
                if gs % CHECKPOINT_INTERVAL == 0:
                    torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                                'opt_state_dict': opt.state_dict()}, MODEL_PATH)
                
                if gs % 100 == 0:
                    torch.cuda.empty_cache()
                
                elapsed = time.time() - t0
                it_s = gs / elapsed
                eta = (total_steps - gs) / it_s / 60 if it_s > 0 else 9999
                pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", it_s=f"{it_s:.2f}", eta=f"{eta:.0f}m")
                pbar.update(1)
            
            del anc, pos, neg, za, zp, zn, loss, anc_mask, pos_mask, neg_mask
            
        pbar.close()
        tqdm.write("\n🏁 Завершено.")
        torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                    'opt_state_dict': opt.state_dict()}, MODEL_PATH)
        
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Прервано")
    except Exception as e:
        tqdm.write(f"\n❌ Ошибка: {e}")
        import traceback; traceback.print_exc()
    finally:
        if not plot_saved and steps: draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
        elapsed = time.time() - t0
        tqdm.write(f"⏱️ Время: {elapsed/60:.1f} мин | Шагов: {gs} | Скорость: {elapsed/max(1,gs):.2f} s/it")
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()