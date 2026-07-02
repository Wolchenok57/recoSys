#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_books.py — Оценка RAG-энкодера на валидационных книгах
- Запрос: первая глава книги
- Позитивные документы: все остальные главы той же книги
- Негативные документы: все главы других валидационных книг
- Метрики: Recall@K (1,5,10,50), MRR
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
# Пути (должны совпадать с train2.py)
DATA_ROOT = "/mnt/news/llm_ds/fics"
VAL_IDS_PATH = "logs2/val_books.json"         # список ID валидационных книг
CHECKPOINT_PATH = "logs6_from_3_using_2_3/model.pth"  # путь к обученной модели

# Параметры
WINDOW_SIZE = 1024          # длина последовательности (должна совпадать с обучением)
BATCH_SIZE = 16             # для инференса (подберите под свою видеокарту)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16       # или torch.bfloat16, если поддерживается

# Модель (импорты из ваших файлов)
from model import RAGEncoder, BOS_TOKEN_ID, PAD_TOKEN_ID, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, DROPOUT, N_LAYERS, OUTPUT_DIM, MAX_SEQ_LEN
from testoBPE import BPE

# ==================== ФУНКЦИИ ЗАГРУЗКИ ====================
def scan_books_paths(data_root: str) -> dict:
    """Возвращает {book_id: [путь_к_главе1, путь_к_главе2, ...]}"""
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

def load_books_to_ram(book_ids: list, all_paths: dict, window_size: int, pad_id: int) -> dict:
    """Загружает книги в RAM: {book_id: список тензоров глав}"""
    books = {}
    for book_id in tqdm(book_ids, desc="Loading books", unit="book", file=sys.stderr):
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

def get_embeddings(model, texts, batch_size=16):
    """
    Вычисляет эмбеддинги для списка текстов (тензоров токенов).
    texts: список тензоров формы [seq_len] (уже подготовленных).
    """
    model.eval()
    embeddings = []
    total = len(texts)
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Embedding", unit="batch", leave=False):
            batch = texts[i:i+batch_size]
            # Стек тензоров в батч
            batch_tensor = torch.stack(batch).to(DEVICE)
            emb = model(batch_tensor)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def load_model():
    """Инициализация и загрузка модели"""
    print("🏗️ Загрузка модели...", file=sys.stderr)
    model = RAGEncoder(dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                       ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
                       output_dim=OUTPUT_DIM, max_seq_len=MAX_SEQ_LEN).to(DEVICE, dtype=DTYPE)
    
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Конвертация в нужный тип
        if DTYPE == torch.float16:
            cleaned = {k: v.half() if v.dtype == torch.float32 else v for k, v in cleaned.items()}
        elif DTYPE == torch.bfloat16:
            cleaned = {k: v.bfloat16() if v.dtype == torch.float32 else v for k, v in cleaned.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"✅ Веса загружены из {CHECKPOINT_PATH} ({DTYPE})", file=sys.stderr)
    else:
        print(f"❌ Файл {CHECKPOINT_PATH} не найден!", file=sys.stderr)
        sys.exit(1)
    
    model.enable_gradient_checkpointing(False)
    model.eval()
    return model

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    print(f"🔧 Устройство: {DEVICE}, точность: {DTYPE}", file=sys.stderr)
    
    # 0. Загружаем модель
    model = load_model()
    # Токенизатор не нужен, т.к. главы уже токенизированы в .npy
    
    # 1. Загрузка списка валидационных книг
    if not os.path.exists(VAL_IDS_PATH):
        print(f"❌ Файл {VAL_IDS_PATH} не найден", file=sys.stderr)
        sys.exit(1)
    with open(VAL_IDS_PATH, 'r') as f:
        val_book_ids = json.load(f)
    print(f"📚 Валидационных книг: {len(val_book_ids)}", file=sys.stderr)
    
    # 2. Сканирование всех путей и фильтрация только валидационных
    all_paths = scan_books_paths(DATA_ROOT)
    val_paths = {bid: all_paths[bid] for bid in val_book_ids if bid in all_paths}
    missing = set(val_book_ids) - set(val_paths.keys())
    if missing:
        print(f"⚠️ Не найдены книги: {missing}", file=sys.stderr)
    print(f"✅ Загружено путей для {len(val_paths)} книг", file=sys.stderr)
    
    # 3. Загрузка содержимого в RAM
    print("📥 Загрузка глав в RAM...", file=sys.stderr)
    val_books = load_books_to_ram(list(val_paths.keys()), val_paths, WINDOW_SIZE, PAD_TOKEN_ID)
    del val_paths, all_paths
    gc.collect()
    
    # 4. Подготовка запросов и документов
    queries = []       # список тензоров первой главы
    query_book_ids = []
    doc_tensors = []   # все главы, КРОМЕ первой (чтобы не было self-match)
    doc_book_ids = []  # для каждого документа — id книги
    doc_chapter_idx = []  # номер главы (0-based, но первая глава исключена)

    for book_id, chapters in val_books.items():
        if len(chapters) == 0:
            continue
        # Запрос = первая глава
        queries.append(chapters[0])
        query_book_ids.append(book_id)
        # Документы = все главы, начиная со второй (индекс 1 и далее)
        for idx, chap in enumerate(chapters[1:], start=1):  # start=1 для номера главы
            doc_tensors.append(chap)
            doc_book_ids.append(book_id)
            doc_chapter_idx.append(idx)

    
    print(f"📊 Запросов (книг): {len(queries)}", file=sys.stderr)
    print(f"📄 Документов (глав): {len(doc_tensors)}", file=sys.stderr)
    
    # 5. Вычисление эмбеддингов
    print("🔮 Вычисление эмбеддингов для запросов...", file=sys.stderr)
    emb_queries = get_embeddings(model, queries, BATCH_SIZE)
    emb_queries = F.normalize(emb_queries, p=2, dim=1)
    
    print("🔮 Вычисление эмбеддингов для документов...", file=sys.stderr)
    emb_docs = get_embeddings(model, doc_tensors, BATCH_SIZE)
    emb_docs = F.normalize(emb_docs, p=2, dim=1)
    
    # 6. Построение матрицы сходства (запросы × документы)
    print("📊 Подсчёт сходства...", file=sys.stderr)
    scores = emb_queries @ emb_docs.T  # [num_queries, num_docs]
    
    # 7. Оценка метрик
    k_list = [1, 5, 10, 50]
    recalls = {k: 0 for k in k_list}
    mrr_sum = 0.0
    num_queries = len(queries)
    
    for q_idx, book_id in enumerate(tqdm(query_book_ids, desc="Evaluating", unit="query")):
        # Индексы документов, принадлежащих этой же книге
        pos_indices = [i for i, bid in enumerate(doc_book_ids) if bid == book_id]
        if not pos_indices:
            continue
        
        q_scores = scores[q_idx]
        sorted_indices = torch.argsort(q_scores, descending=True).cpu().numpy()
        best_rank = len(doc_tensors) + 1
        for pos_idx in pos_indices:
            rank = np.where(sorted_indices == pos_idx)[0][0] + 1
            if rank < best_rank:
                best_rank = rank
        
        if best_rank <= len(doc_tensors):
            mrr_sum += 1.0 / best_rank
            for k in recalls:
                if best_rank <= k:
                    recalls[k] += 1
    
    for k in recalls:
        recalls[k] /= num_queries
    mrr = mrr_sum / num_queries
    
    # 8. Вывод результатов
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ КНИЖНОГО RETRIEVAL (валидационные книги)")
    print("="*60)
    print(f"🎯 Запрос: первая глава → ищем остальные главы той же книги")
    print(f"📚 Книг в валидации: {num_queries}")
    print(f"📄 Всего документов (глав): {len(doc_tensors)}")
    print("-"*40)
    print(f"🔝 Recall@1:  {recalls[1]*100:.2f}%")
    print(f"🔝 Recall@5:  {recalls[5]*100:.2f}%")
    print(f"🔝 Recall@10: {recalls[10]*100:.2f}%")
    print(f"🔝 Recall@50: {recalls[50]*100:.2f}%")
    print(f"🎯 MRR:       {mrr:.4f}")
    print("="*60)
    
    if recalls[1] > 0.5:
        print("✅ Отлично! Первая глава точно находит остальные → RAG работает.")
    elif recalls[5] > 0.7:
        print("✅ Хорошо: топ-5 глав подходят → можно использовать с реранкингом.")
    elif recalls[10] > 0.8:
        print("⚠️ Приемлемо: нужно много кандидатов, но полезно для рекомендаций.")
    else:
        print("❌ Плохо: модель плохо отличает свои главы от чужих — требуется дообучение.")
    
    print(f"\n📊 Среднее число глав на книгу: {len(doc_tensors)/num_queries:.1f}")

if __name__ == '__main__':
    main()