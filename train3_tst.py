# train3_tst.py — ОБУЧЕНИЕ (4096 токенов, BS=1, attention_mask, БЕЗ ru-WANLI)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, random, time, gc, faulthandler
faulthandler.enable()

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False

from model import RAGEncoder, VOCAB_SIZE, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN, BOS_TOKEN_ID, PAD_TOKEN_ID
from testoBPE import BPE

# ==================== КОНСТАНТЫ ====================
WINDOW_SIZE = 4096
SAMPLE_SIZE = 1000

BATCH_SIZE = 1
ACCUMULATION_STEPS = 1
NUM_WORKERS = 0

BASE_LR = 1e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 100

N_NEGATIVES = 3
TEMPERATURE = 0.1
MARGIN_TARGET = 0.2

PREV_LOG_DIR = "logs2_2"
CUR_LOG_DIR = "logs3_test_run"
CHECKPOINT_PATH = os.path.join(PREV_LOG_DIR, "model.pth")

LLM_DATASETS_ROOT = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets"

VAL_INTERVAL = 200
PLOT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

# ==================== НАСТРОЙКА ПУТЕЙ ====================
os.makedirs(CUR_LOG_DIR, exist_ok=True)
LOSS_LOG = os.path.join(CUR_LOG_DIR, 'loss.jsonl')
METRICS_LOG = os.path.join(CUR_LOG_DIR, 'metrics.jsonl')
PLOT_PATH = os.path.join(CUR_LOG_DIR, 'training.png')
MODEL_PATH = os.path.join(CUR_LOG_DIR, "model.pth")

for f in [LOSS_LOG, METRICS_LOG]: open(f, 'w').close()

tok = BPE()

# ==================== ЗАГРУЗКА ДАТАСЕТОВ (4 источника, БЕЗ ru-WANLI) ====================
def load_gazeta(n_samples=SAMPLE_SIZE):
    path = os.path.join(LLM_DATASETS_ROOT, "gazeta/default/train/0000.parquet")
    df = pd.read_parquet(path).sample(min(n_samples, 10000), random_state=42)
    return [{'anchor': str(r['summary']), 'positive': str(r['text']), 'source': 'gazeta'} 
            for _, r in df.iterrows() if str(r['summary']) and str(r['text'])]

def load_samsum_ru(n_samples=SAMPLE_SIZE):
    path = os.path.join(LLM_DATASETS_ROOT, "samsum-ru/data/train-00000-of-00001-76cc3fe8132d8f4b.parquet")
    df = pd.read_parquet(path).sample(min(n_samples, 10000), random_state=42)
    return [{'anchor': str(r['summary']), 'positive': str(r['dialogue']), 'source': 'samsum-ru'} 
            for _, r in df.iterrows() if str(r['summary']) and str(r['dialogue'])]

def load_xlsum_bbc(n_samples=SAMPLE_SIZE):
    path = os.path.join(LLM_DATASETS_ROOT, "xlsum-russian-bbc/bbcrussian.csv.gz")
    df = pd.read_csv(path, compression="gzip", nrows=n_samples*2).sample(min(n_samples, 10000), random_state=42)
    return [{'anchor': str(r['resume']), 'positive': str(r['news']), 'source': 'xlsum-bbc'} 
            for _, r in df.iterrows() if str(r['resume']) and str(r['news'])]

def load_rusenteval(n_samples=SAMPLE_SIZE):
    data_path = os.path.join(LLM_DATASETS_ROOT, "RuSentEval/data")
    samples, loaded = [], 0
    for fname in os.listdir(data_path):
        if fname.endswith(".txt") and loaded < n_samples:
            with open(os.path.join(data_path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    if loaded >= n_samples: break
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        samples.append({'anchor': parts[1], 'positive': parts[2], 'source': 'RuSentEval'})
                        loaded += 1
    return samples

def tokenize_samples(raw_list):
    out = []
    for item in raw_list:
        a_ids = tok.encode(item['anchor'])[:WINDOW_SIZE]
        p_ids = tok.encode(item['positive'])[:WINDOW_SIZE]
        out.append({
            'anchor': torch.tensor(a_ids, dtype=torch.long),
            'positive': torch.tensor(p_ids, dtype=torch.long),
            'source': item['source']
        })
    return out

# ==================== DATASET ====================
class IntraSourceContrastiveDataset(Dataset):
    def __init__(self, samples, n_negatives=3):
        self.samples = samples
        self.n_negatives = n_negatives
        self.source_indices = defaultdict(list)
        for i, s in enumerate(samples):
            self.source_indices[s['source']].append(i)
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        anchor, positive, src = sample['anchor'], sample['positive'], sample['source']
        
        same_src_indices = self.source_indices[src]
        if len(same_src_indices) <= self.n_negatives:
            neg_pool = [i for i in same_src_indices if i != idx]
        else:
            neg_pool = random.sample([i for i in same_src_indices if i != idx], self.n_negatives)
            
        neg_tensors = [self.samples[i]['positive'] for i in neg_pool]
        # ✅ Динамический паддинг только внутри этой тройки
        negatives = pad_sequence(neg_tensors, batch_first=True, padding_value=PAD_TOKEN_ID)
        
        return anchor, positive, negatives

# ==================== MAIN ====================
def main():
    tqdm.write(f"🔥 ОБУЧЕНИЕ (4096 токенов, BS=1, attention_mask, 4 датасета)")
    tqdm.write(f"💾 Устройство: {DEVICE} | {DTYPE} | WS={WINDOW_SIZE}")
    tqdm.write(f"📂 Checkpoint: {CHECKPOINT_PATH} | Логи: {CUR_LOG_DIR}")
    
    tqdm.write("\n📥 Загрузка датасетов (БЕЗ ru-WANLI)...")
    datasets_raw = {
        'gazeta': load_gazeta(),
        'samsum-ru': load_samsum_ru(),
        'xlsum-bbc': load_xlsum_bbc(),
        'RuSentEval': load_rusenteval()
    }
    for name, data in datasets_raw.items():
        tqdm.write(f"   ✅ {name}: {len(data)} примеров")
    
    all_raw = []
    for data in datasets_raw.values(): all_raw.extend(data)
    random.shuffle(all_raw)
    all_tok = tokenize_samples(all_raw)
    tqdm.write(f"\n📊 Токенизировано: {len(all_tok)} примеров")
    
    # Разделяем train/val (10% на валидацию из тех же 4 датасетов)
    val_size = min(400, len(all_tok) // 10)
    train_samples, val_samples = all_tok[val_size:], all_tok[:val_size]
    tqdm.write(f"📚 Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    train_ds = IntraSourceContrastiveDataset(train_samples, N_NEGATIVES)
    val_ds = IntraSourceContrastiveDataset(val_samples, 1)
    
    def collate_fn(batch): return batch[0]
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Загрузка модели
    tqdm.write("\n📦 Загрузка модели...")
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        if DTYPE == torch.bfloat16:
            for k, v in cleaned.items():
                if v.dtype == torch.float32: cleaned[k] = v.bfloat16()
        model.load_state_dict(cleaned, strict=False)
        tqdm.write(f"✅ Веса загружены")
    else:
        tqdm.write(f"⚠️ Чекпоинт не найден")
    
    model.enable_gradient_checkpointing(True)
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, fused=(DEVICE=='cuda'))
    scaler = GradScaler('cuda', enabled=(DTYPE == torch.float16))
    
    total_steps = len(train_dl)
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
            # ✅ Создаём маски ДО unsqueeze
            a_mask = (anchor != PAD_TOKEN_ID).long().to(DEVICE)
            p_mask = (pos != PAD_TOKEN_ID).long().to(DEVICE)
            n_mask = (negs != PAD_TOKEN_ID).long().to(DEVICE)
            
            a = anchor.to(DEVICE).unsqueeze(0)
            p = pos.to(DEVICE).unsqueeze(0)
            ns = negs.to(DEVICE)
            
            with autocast('cuda', dtype=DTYPE):
                za = model(a, attention_mask=a_mask.unsqueeze(0))
                zp = model(p, attention_mask=p_mask.unsqueeze(0))
                # Для негативов: flatten → model → reshape
                Bn, Ln = ns.shape
                zn_flat = model(ns.view(-1, Ln), attention_mask=n_mask.view(-1, Ln))
                zn = zn_flat.unsqueeze(0).view(1, Bn, -1)
                loss, ps, nsim = info_nce_loss(za, zp, zn)
            t_loss += loss.item(); t_pos += ps; t_neg += nsim; n += 1
        model.train()
        return {'loss': t_loss/n, 'pos_sim': t_pos/n, 'neg_sim': t_neg/n, 'margin': (t_pos-t_neg)/n} if n>0 else {}
    
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
    t0 = time.time()
    plot_saved = False
    
    tqdm.write("\n🔍 Валидация (шаг 0)...")
    met = validate(model, val_dl)
    tqdm.write(f"📊 Step 0: Loss={met.get('loss',0):.3f} | Margin={met.get('margin',0):.3f}")
    
    tqdm.write(f"\n🚀 Старт: {len(train_dl)} итераций")
    pbar = tqdm(train_dl, desc="Training", dynamic_ncols=True, leave=True)
    opt.zero_grad(set_to_none=True)
    
    try:
        for epoch in range(1):
            for anc, pos, negs in pbar:
                # ✅ Создаём маски ДО добавления batch-измерения
                anc_mask = (anc != PAD_TOKEN_ID).long().to(DEVICE)
                pos_mask = (pos != PAD_TOKEN_ID).long().to(DEVICE)
                neg_mask = (negs != PAD_TOKEN_ID).long().to(DEVICE)
                
                anc = anc.to(DEVICE, non_blocking=True).unsqueeze(0)
                pos = pos.to(DEVICE, non_blocking=True).unsqueeze(0)
                negs = negs.to(DEVICE, non_blocking=True)
                
                with autocast('cuda', dtype=DTYPE):
                    za = model(anc, attention_mask=anc_mask.unsqueeze(0))
                    zp = model(pos, attention_mask=pos_mask.unsqueeze(0))
                    # Негативы: [N, L_max] → flatten → model → [N, D] → [1, N, D]
                    Bn, Ln = negs.shape
                    zn_flat = model(negs.view(-1, Ln), attention_mask=neg_mask.view(-1, Ln))
                    zn = zn_flat.unsqueeze(0).view(1, negs.size(0), -1)
                    loss, ps, ns = info_nce_loss(za, zp, zn)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                
                cur_lr = get_lr(os_)
                for pg in opt.param_groups: pg['lr'] = cur_lr
                os_ += 1
                gs += 1
                
                cl = loss.item()
                steps.append(gs); losses.append(cl); pos_sims.append(ps); neg_sims.append(ns); margins.append(ps-ns); lrs.append(cur_lr)
                
                with open(LOSS_LOG, 'a') as f:
                    f.write(json.dumps({'step': gs, 'loss': cl, 'pos': ps, 'neg': ns, 'lr': cur_lr}) + '\n')
                
                if gs % VAL_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    vm = validate(model, val_dl)
                    with open(METRICS_LOG, 'a') as f:
                        f.write(json.dumps({'step': gs, **vm}) + '\n')
                    pbar.set_postfix(val_loss=f"{vm.get('loss',0):.3f}", val_margin=f"{vm.get('margin',0):.3f}")
                    gc.collect()
                
                if gs % PLOT_INTERVAL == 0:
                    draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
                    plot_saved = True
                
                if gs % CHECKPOINT_INTERVAL == 0:
                    torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                                'opt_state_dict': opt.state_dict()}, MODEL_PATH)
                    tqdm.write(f"💾 Checkpoint: {MODEL_PATH}")
                
                pbar.set_postfix(loss=f"{cl:.3f}", margin=f"{ps-ns:.3f}", lr=f"{cur_lr:.2e}")
                del anc, pos, negs, za, zp, zn, loss, anc_mask, pos_mask, neg_mask
            
            tqdm.write("\n✅ Эпоха завершена.")
            break
        
        tqdm.write("\n📊 Финальная валидация...")
        final_met = validate(model, val_dl)
        tqdm.write(f"✅ Final: Loss={final_met.get('loss',0):.3f} | Margin={final_met.get('margin',0):.3f}")
        
        torch.save({'step': gs, 'model_state_dict': {k.replace('_orig_mod.',''):v for k,v in model.state_dict().items()},
                    'opt_state_dict': opt.state_dict()}, MODEL_PATH)
        tqdm.write(f"💾 Модель: {MODEL_PATH}")
        
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Прервано")
    except Exception as e:
        tqdm.write(f"\n❌ Ошибка: {e}")
        import traceback; traceback.print_exc()
    finally:
        if not plot_saved and steps: draw_plot(steps, losses, pos_sims, neg_sims, margins, lrs)
        tqdm.write(f"⏱️ Время: {(time.time()-t0)/60:.1f} мин | Шагов: {gs}")
        if DEVICE == 'cuda': torch.cuda.empty_cache()

if __name__ == '__main__':
    main()