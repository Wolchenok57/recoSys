#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_real_rag.py — Тест семантического поиска на deepvk/ru-WANLI
✅ Добавлен tqdm для отслеживания прогресса
✅ Отключён gradient checkpointing (ускоряет инференс в 2-3 раза)
✅ Явно отключён FlashAttention для GTX 1650
✅ FP16 инференс, безопасная обработка батчей
"""

import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==================== ОТКЛЮЧЕНИЕ FLASH ATTENTION ====================
# GTX 1650 (Turing) не поддерживает FlashAttention. Отключаем явно.
# if torch.cuda.is_available():
#     torch.backends.cuda.enable_flash_sdp(False)
#     torch.backends.cuda.enable_mem_efficient_sdp(False)
#     torch.backends.cuda.enable_math_sdp(True)  # Фоллбэк на стандартную математику
#     print("⚙️ FlashAttention отключён, включён math SDP", file=sys.stderr)
# ===================================================================

try:
    import pandas as pd
except ImportError:
    print("❌ Установи: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)

try:
    import pyarrow
except ImportError:
    print("❌ Установи: pip install pyarrow", file=sys.stderr)
    sys.exit(1)

# ==================== НАСТРОЙКИ ====================
# MODEL_PATH = "book_embeddings_custom_768.npy"
MODEL_PATH = "logs6_from_3_using_2_3/model.pth"
EMBEDDING_CHECKPOINT = "OLD/logs/checkpoint.pth"
DATASET_PATH = "/mnt/neurostuff/llm_datasets/ru-WANLI/data/train.parquet"

TEST_SIZE = 2000
BATCH_SIZE = 16  # Комфортно для 4GB VRAM
# DEVICE_INFERENCE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DEVICE_INFERENCE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16

from model import RAGEncoder, BOS_TOKEN_ID
from testoBPE import BPE
# ===================================================

def print_mem():
    if DEVICE_INFERENCE.type == 'cuda':
        alloc = torch.cuda.memory_allocated(DEVICE_INFERENCE) / 1024**2
        reserved = torch.cuda.memory_reserved(DEVICE_INFERENCE) / 1024**2
        print(f"[MEM] Alloc: {alloc:.0f}MB, Reserved: {reserved:.0f}MB", file=sys.stderr)

def load_model():
    print("🏗️ Загрузка модели...", file=sys.stderr)
    model = RAGEncoder().to(DEVICE_INFERENCE, dtype=DTYPE)
    
    if os.path.exists(EMBEDDING_CHECKPOINT):
        model.load_embeddings(EMBEDDING_CHECKPOINT)
    
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Конвертация в fp16 перед загрузкой
        for k, v in cleaned.items():
            if v.dtype == torch.float32:
                cleaned[k] = v.half()
        model.load_state_dict(cleaned, strict=False)
        print(f"✅ Веса загружены из {MODEL_PATH} (fp16)", file=sys.stderr)
    else:
        print(f"❌ {MODEL_PATH} не найден!", file=sys.stderr)
        sys.exit(1)
        
    # ОТКЛЮЧАЕМ чекпоинтинг для инференса (ускоряет forward pass)
    model.enable_gradient_checkpointing(False)
    model.eval()
    return model

def load_data(path, n_samples=2000):
    print(f"📖 Чтение {path}...", file=sys.stderr)
    if not os.path.exists(path):
        print(f"❌ Файл не найден: {path}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_parquet(path)
    print(f"   Всего строк: {len(df)}", file=sys.stderr)
    
    entail = df[df['label'] == 'entailment'][['premise', 'hypothesis']].dropna()
    contradict = df[df['label'] == 'contradiction'][['premise', 'hypothesis']].dropna()
    
    print(f"   Entailment: {len(entail)}, Contradiction: {len(contradict)}", file=sys.stderr)
    
    n = min(n_samples, len(entail), len(contradict))
    entail = entail.sample(n, random_state=42).reset_index(drop=True)
    contradict = contradict.sample(n, random_state=42).reset_index(drop=True)
    
    return entail, contradict

def get_embeddings(model, tokenizer, texts):
    """Получение эмбеддингов с tqdm и без зависаний"""
    embeddings = []
    model.eval()
    
    # Очищаем кэш перед стартом
    if DEVICE_INFERENCE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding", unit="batch", total=total_batches)
    
    for i in pbar:
        batch_texts = texts[i : i+BATCH_SIZE]
        
        # Токенизация
        batch_tokens = []
        for t in batch_texts:
            txt = t if isinstance(t, str) and t.strip() else "пусто"
            ids = tokenizer.encode(txt)
            # Конвертируем в список если тензор
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = [BOS_TOKEN_ID] + ids
            # Обрезаем длинные последовательности
            if len(ids) > 1024:
                ids = ids[:1024]
            batch_tokens.append(torch.tensor(ids, dtype=torch.long))
        
        if not batch_tokens:
            pbar.update(1)
            continue
            
        # Паддинг до макс длины в батче
        batch_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_tokens, batch_first=True, padding_value=0
        ).to(DEVICE_INFERENCE)
        
        # Forward pass (без autocast, т.к. модель уже в fp16)
        with torch.no_grad():
            embs = model(batch_tensor)  # (B, 768)
            embeddings.append(embs.cpu())
        
        pbar.update(1)
        # Обновляем прогресс-бар
        pbar.set_postfix({'batch': f'{pbar.n}/{total_batches}', 'shape': list(embs.shape)})
    
    if not embeddings:
        return torch.zeros(0, 768)
    return torch.cat(embeddings, dim=0)

def main():
    print_mem()
    model = load_model()
    tokenizer = BPE()
    
    entail_df, contradict_df = load_data(DATASET_PATH, n_samples=TEST_SIZE)
    
    queries = entail_df['premise'].tolist()
    positives = entail_df['hypothesis'].tolist()
    negatives = contradict_df['hypothesis'].tolist()
    
    print(f"\n🔮 Вычисление эмбеддингов (batch={BATCH_SIZE}, device={DEVICE_INFERENCE})...", file=sys.stderr)
    start_time = time.time()
    
    emb_q = get_embeddings(model, tokenizer, queries)
    emb_p = get_embeddings(model, tokenizer, positives)
    emb_n = get_embeddings(model, tokenizer, negatives)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Эмбеддинги вычислены за {elapsed:.1f} сек ({len(queries)*3} текстов)", file=sys.stderr)
    
    # Нормализация
    emb_q = F.normalize(emb_q, p=2, dim=1)
    emb_p = F.normalize(emb_p, p=2, dim=1)
    emb_n = F.normalize(emb_n, p=2, dim=1)
    
    print(f"📊 Подсчёт метрик...", file=sys.stderr)
    
    sim_pos = (emb_q * emb_p).sum(dim=1)
    sim_neg = (emb_q * emb_n).sum(dim=1)
    
    correct = (sim_pos > sim_neg).sum().item()
    accuracy = correct / len(queries)
    
    # Recall@K
    POOL_SIZE = min(2000, len(queries))
    print(f"🔍 Recall@K на пуле из {POOL_SIZE} кандидатов...", file=sys.stderr)
    
    q_sub = emb_q[:POOL_SIZE]
    p_pool = emb_p[:POOL_SIZE]
    scores = q_sub @ p_pool.T
    
    recalls = {1: 0, 5: 0, 10: 0, 50: 0}
    
    for i in range(POOL_SIZE):
        q_scores = scores[i]
        sorted_idx = torch.argsort(q_scores, descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
        for k in recalls:
            if rank <= k:
                recalls[k] += 1
    
    # Вывод
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ (ru-WANLI)")
    print("="*60)
    print(f"🎯 Accuracy (Pos > Neg):      {accuracy*100:.2f}%")
    print(f"🔝 Recall@1:                  {recalls[1]/POOL_SIZE * 100:.2f}%")
    print(f"🔝 Recall@5:                  {recalls[5]/POOL_SIZE * 100:.2f}%")
    print(f"🔝 Recall@10:                 {recalls[10]/POOL_SIZE * 100:.2f}%")
    print(f"🔝 Recall@50:                 {recalls[50]/POOL_SIZE * 100:.2f}%")
    print("-"*60)
    print(f"📉 Mean Sim (Positive):       {sim_pos.mean().item():.4f} ± {sim_pos.std().item():.4f}")
    print(f"📉 Mean Sim (Negative):       {sim_neg.mean().item():.4f} ± {sim_neg.std().item():.4f}")
    print(f"📊 Margin (Pos - Neg):        {(sim_pos - sim_neg).mean().item():.4f}")
    print("="*60)
    
    print("\n🎯 ИНТЕРПРЕТАЦИЯ:")
    if accuracy > 0.85:
        print("✅ ОТЛИЧНО: Модель уверенно различает семантику!")
    elif accuracy > 0.70:
        print("✅ ХОРОШО: Модель работает, есть куда расти")
    else:
        print("⚠️ ТРЕБУЕТСЯ ДООбучение: Точность ниже ожидаемой")
    
    if recalls[1] / POOL_SIZE > 0.60:
        print("✅ Recall@1 > 60% — готово для простого RAG")
    elif recalls[5] / POOL_SIZE > 0.80:
        print("✅ Recall@5 > 80% — подойдёт для RAG с reranking")
    else:
        print("⚠️ Низкий Recall — нужно больше контрастивных шагов")
    print("="*60)

if __name__ == '__main__':
    main()
