#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка ранга книги Буратино после всех изменений."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from model import RAGEncoder, BOS_TOKEN_ID
from testoBPE import BPE
import mysql.connector

DB_CONFIG = {
    "host": "192.168.0.113",
    "user": "debservak",
    "password": "ТвойПароль123",
    "database": "kursach",
    "charset": "utf8mb4",
    "use_pure": True
}

MODEL_PATH = "logs6_from_3_using_2_3/model.pth"
EMB_PATH = "book_embeddings_custom_768.npy"
MAX_Q = 1024
BOS = BOS_TOKEN_ID

def db():
    return mysql.connector.connect(**DB_CONFIG)

def load_rag():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    model = RAGEncoder().to(device, dtype=dtype)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    
    tokenizer = BPE()
    return model, tokenizer, device

def encode_text(text, model, tokenizer, device):
    ids = tokenizer.encode(text)
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
    ids = [BOS] + ids[:MAX_Q]
    
    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long).to(device)
        e = model(inp)
        e = F.normalize(e, p=2, dim=1)
    
    return e.cpu().numpy().astype(np.float32).squeeze(0)

print("=" * 70)
print("ДИАГНОСТИКА ПОСЛЕ ВСЕХ ИЗМЕНЕНИЙ")
print("=" * 70)

# Загружаем модель и эмбеддинги
model, tokenizer, device = load_rag()
book_embs = np.load(EMB_PATH)

# Получаем список книг с описанием (те, для которых есть эмбеддинги)
conn = db()
cur = conn.cursor()
cur.execute("SELECT id FROM books WHERE LENGTH(description) > 10 ORDER BY id")
book_ids_with_desc = [row[0] for row in cur.fetchall()]
cur.close()
conn.close()

target_id = 44856

# Проверяем, есть ли целевая книга в эмбеддингах
if target_id not in book_ids_with_desc:
    print(f"❌ Книга ID {target_id} НЕ ВХОДИТ в список книг с описанием!")
    print(f"   Всего книг с описанием: {len(book_ids_with_desc)}")
    print(f"   Первые 10: {book_ids_with_desc[:10]}")
    sys.exit(1)

target_idx = book_ids_with_desc.index(target_id)
target_emb = book_embs[target_idx]

print(f"\n🎯 Книга ID {target_id} НАЙДЕНА в эмбеддингах!")
print(f"   Индекс: {target_idx}")
print(f"   Норма эмбеддинга: {np.linalg.norm(target_emb):.6f}")

# Проверяем, кто находится рядом с книгой
print(f"\n📊 Ближайшие соседи (по сходству с ID {target_id}):")
similarities = []
for i, emb in enumerate(book_embs):
    sim = np.dot(target_emb, emb)
    similarities.append((i, sim))

similarities.sort(key=lambda x: x[1], reverse=True)

for i, (idx, sim) in enumerate(similarities[:10]):
    book_id = book_ids_with_desc[idx]
    conn2 = db()
    cur2 = conn2.cursor(dictionary=True)
    cur2.execute("SELECT name FROM books WHERE id = %s", (book_id,))
    name = cur2.fetchone()
    cur2.close()
    conn2.close()
    marker = "✅" if book_id == target_id else " "
    print(f"   {marker} {i+1}. ID {book_id}: {name['name'][:40] if name else '?'} (sim: {sim:.6f})")

# Проверяем запрос
print(f"\n🔍 ЗАПРОС: 'деревянная кукла по-итальянски буратино'")
q_emb = encode_text("деревянная кукла по-итальянски буратино", model, tokenizer, device)

all_sims = []
for i, emb in enumerate(book_embs):
    s = np.dot(emb, q_emb)
    all_sims.append((i, s))

all_sims.sort(key=lambda x: x[1], reverse=True)

# Находим ранг целевой книги
rank = None
for i, (idx, sim) in enumerate(all_sims):
    if book_ids_with_desc[idx] == target_id:
        rank = i + 1
        target_sim = sim
        break

print(f"\n📊 Ранг ID {target_id} в результатах поиска: {rank if rank else 'не найден'}")
if rank:
    print(f"   Сходство: {target_sim:.6f}")

print(f"\n📊 Топ-5 результатов:")
for i, (idx, sim) in enumerate(all_sims[:5]):
    book_id = book_ids_with_desc[idx]
    conn2 = db()
    cur2 = conn2.cursor(dictionary=True)
    cur2.execute("SELECT name FROM books WHERE id = %s", (book_id,))
    name = cur2.fetchone()
    cur2.close()
    conn2.close()
    marker = "✅" if book_id == target_id else " "
    print(f"   {marker} {i+1}. ID {book_id}: {name['name'][:40] if name else '?'} (sim: {sim:.6f})")

print("\n" + "=" * 70)
print("💡 ВЫВОД:")
print("=" * 70)

if rank and rank <= 5:
    print("✅ Книга ID 44856 находится в ТОП-5!")
elif rank and rank <= 10:
    print(f"📌 Книга ID 44856 на позиции {rank} (не в топ-5)")
else:
    print(f"❌ Книга ID 44856 на позиции {rank if rank else 'не найдена'} (далеко от топа)")
    
print("\n🎯 Рекомендуемый запрос для поиска книги Буратино:")
print("   'деревянная кукла по-итальянски буратино'")
print(f"   (ранг: {rank if rank else 'не найден'})")
