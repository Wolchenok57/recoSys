#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import mysql.connector
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import coo_matrix, csr_matrix
import joblib
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import TruncatedSVD


# ================= КОНФИГ =================
DB_CONFIG = {
    'host': '192.168.0.113',
    'user': 'debservak',
    'password': 'ТвойПароль123',
    'database': 'kursach',
    'charset': 'utf8mb4',
    'use_pure': True
}

RELEARN_MIN_USER = 5
RELEARN_MIN_ITEM = 10
RELEARN_EPOCHS = 200
RELEARN_PCA_DIM = 368
RELEARN_N_THREADS = min(os.cpu_count() or 4, 8)


BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16
EMBEDDINGS_CACHE = "book_embeddings_full_768.npy"
# ==========================================

def fetch_data():
    print("🔍 Загрузка данных из БД...")
    query = """
    SELECT 
        ur.idUser AS user_id,
        ur.idBook AS book_id,
        CASE 
            WHEN ur.rating >= 7 THEN 1.0 
            WHEN ur.rating >= 5 THEN 0.5 
            ELSE 0.1 
        END AS weight,
        b.name,
        b.description,
        GROUP_CONCAT(bmd.data SEPARATOR ' | ') AS tags
    FROM userRating ur
    JOIN books b ON ur.idBook = b.id
    LEFT JOIN booksMetaData bmd ON b.id = bmd.idBook AND bmd.type = 'tag'
    WHERE ur.rating IS NOT NULL
    GROUP BY ur.idUser, ur.idBook, b.name, b.description
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conn.close()
    
    df = df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
    print(f"✅ Загружено {len(df)} записей. Юзеров: {df['user_id'].nunique()}, Книг: {df['book_id'].nunique()}")
    return df

def clean_text(row):
    desc = row['description']
    if pd.notna(desc) and len(str(desc).strip()) > 10:
        text = str(desc)
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        return text
    tags = row['tags'] if pd.notna(row['tags']) else "без жанра"
    return f"{row['name']}. Темы: {tags}"

def get_rag_embeddings(texts, model, tokenizer):
    print("🔮 Генерация эмбеддингов через RAG...")
    model.eval()
    embeddings = []
    BOS_TOKEN_ID = 1  # Поправь под свою модель
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        tokens = []
        for t in batch:
            ids = tokenizer.encode(str(t))
            if isinstance(ids, torch.Tensor): ids = ids.tolist()
            ids = [BOS_TOKEN_ID] + ids
            if len(ids) > 1024: ids = ids[:1024]
            tokens.append(torch.tensor(ids, dtype=torch.long))
            
        padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0).to(DEVICE)
        with torch.no_grad():
            embs = model(padded)
            embs = F.normalize(embs, p=2, dim=1)
        embeddings.append(embs.cpu())
        
    emb_tensor = torch.cat(embeddings, dim=0)
    print(f"✅ Получены эмбеддинги: {emb_tensor.shape}")
    return emb_tensor.numpy().astype(np.float32)





def main():
    df = fetch_data()
    df['clean_text'] = df.apply(clean_text, axis=1)
    
    # 🔥 Работаем только с УНИКАЛЬНЫМИ книгами для эмбеддингов
    unique_books = df[['book_id', 'clean_text']].drop_duplicates(subset='book_id')
    unique_ids = unique_books['book_id'].tolist()
    unique_texts = unique_books['clean_text'].tolist()
    print(f"📦 Уникальных книг для векторизации: {len(unique_ids)}")

    # 🔥 КЭШИРОВАНИЕ ЭМБЕДДИНГОВ
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"📂 Загрузка эмбеддингов из кэша: {EMBEDDINGS_CACHE}")
        embeddings = np.load(EMBEDDINGS_CACHE)
    else:
        try:
            from model import RAGEncoder
            from testoBPE import BPE
            
            print("🏗️ Инициализация RAGEncoder...")
            model = RAGEncoder().to(DEVICE, dtype=DTYPE)
            ckpt = torch.load("logs6_from_3_using_2_3/model.pth", map_location='cpu', weights_only=False)
            
            state = {}
            for k, v in ckpt.items():
                k_clean = k.replace('_orig_mod.', '')
                if isinstance(v, torch.Tensor):
                    state[k_clean] = v.half() if v.dtype == torch.float32 else v
                else:
                    state[k_clean] = v
                    
            model.load_state_dict(state, strict=False)
            model.enable_gradient_checkpointing(False)
            tokenizer = BPE()
            print("✅ RAG-модель загружена")
            
            embeddings = get_rag_embeddings(unique_texts, model, tokenizer)
            np.save(EMBEDDINGS_CACHE, embeddings)
            print(f"💾 Эмбеддинги сохранены в {EMBEDDINGS_CACHE}")
        except Exception as e:
            print(f"❌ Ошибка загрузки RAG: {type(e).__name__}: {e}")
            return

    # 🔥 ИНИЦИАЛИЗАЦИЯ LIGHTFM
    print("🛠️ Инициализация LightFM Dataset...")
    dataset = Dataset()
    dataset.fit(users=df['user_id'].unique(), items=df['book_id'].unique())
    
    def interaction_generator():
        for _, row in df.iterrows():
            yield (row['user_id'], row['book_id'], row['weight'])
            
    interactions, weights = dataset.build_interactions(interaction_generator())
    print(f"📊 Матрица взаимодействий: {interactions.shape}")
    
    user_map, _, item_map, _ = dataset.mapping()
    
    # 🔥 ВЫРАВНИВАНИЕ ФИЧЕЙ ПО ВНУТРЕННИМ ИНДЕКСАМ LIGHTFM
    n_items = len(item_map)
    dim = embeddings.shape[1]
    aligned = np.zeros((n_items, dim), dtype=np.float32)
    for bid, emb in zip(unique_ids, embeddings):
        if bid in item_map:
            aligned[item_map[bid]] = emb
            
    item_features_sparse = csr_matrix(aligned)
    print(f"🔗 Item features: {item_features_sparse.shape} (Full {dim}D)")

    # 🔥 ОБУЧЕНИЕ
    print("🚀 Обучение LightFM (loss='warp', Full 768D features)...")
    model = LightFM(
        loss='warp', no_components=64, learning_rate=0.05,
        item_alpha=0.0, user_alpha=0.0, max_sampled=10, random_state=42
    )
    
    n_threads = min(os.cpu_count() or 4, 8)
    model.fit(interactions, sample_weight=weights, item_features=item_features_sparse, 
              epochs=20, num_threads=n_threads, verbose=True)
              
    print("\n📈 Оценка качества...")
    k = 10
    prec = precision_at_k(model, interactions, k=k, item_features=item_features_sparse, num_threads=n_threads).mean()
    rec = recall_at_k(model, interactions, k=k, item_features=item_features_sparse, num_threads=n_threads).mean()
    print(f"Precision@{k}: {prec:.4f} | Recall@{k}: {rec:.4f}")
    
    # 🔥 СОХРАНЕНИЕ
    artifacts = {
        'model': model, 'dataset': dataset, 
        'item_features': item_features_sparse
    }
    joblib.dump(artifacts, 'lightfm_rag_artifacts.pkl')
    print("💾 Артефакты сохранены в lightfm_rag_artifacts.pkl")

if __name__ == '__main__':
    main()