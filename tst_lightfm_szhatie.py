#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент: влияние размерности PCA на качество LightFM+RAG
Цель: найти баланс между семантикой (высокая dim) и стабильностью (низкая dim)
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mysql.connector
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# ================= КОНФИГ =================
DB_CONFIG = {
    'host': '192.168.0.113', 'user': 'debservak', 'password': 'ТвойПароль123',
    'database': 'kursach', 'charset': 'utf8mb4', 'use_pure': True
}
CUSTOM_CACHE = "book_embeddings_custom_768.npy"
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16
TOP_K = 10
EPOCHS = 20
N_THREADS = min(os.cpu_count() or 4, 8)
# Размеры для теста: от полных 768D до минимальных 4D
PCA_DIMS = [768, 512, 368, 256, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4]
# ==========================================

def fetch_data():
    print("📥 Загрузка данных...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    cur.execute("""SELECT ur.idUser AS user_id, ur.idBook AS book_id,
                          CASE WHEN ur.rating >= 7 THEN 1.0 WHEN ur.rating >= 5 THEN 0.5 ELSE 0.1 END AS weight
                   FROM userRating ur WHERE ur.rating IS NOT NULL""")
    df_rat = pd.DataFrame(cur.fetchall())
    cur.execute("""SELECT b.id FROM books b""")
    df_books = pd.DataFrame(cur.fetchall())
    cur.close(); conn.close()
    
    df_rat = df_rat.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
    df_rat['user_id'] = df_rat['user_id'].astype(int)
    df_rat['book_id'] = df_rat['book_id'].astype(int)
    df_rat['weight'] = df_rat['weight'].astype(float)
    print(f"✅ {len(df_rat)} взаимодействий, {len(df_books)} книг.")
    return df_rat, df_books['id'].unique()

def get_embeddings_custom(all_book_ids):
    print("🔮 Загрузка кастомных эмбеддингов (768D)...")
    if os.path.exists(CUSTOM_CACHE):
        emb = np.load(CUSTOM_CACHE)
        print(f"📂 Кэш загружен: {emb.shape}")
        return dict(zip(all_book_ids, emb))
    return None

def split_interactions(interactions, test_frac=0.2, min_train=1):
    np.random.seed(42)
    rows, cols, data = interactions.row.copy(), interactions.col.copy(), interactions.data.copy()
    tr, tc, td, ter, tec, ted = [],[],[],[],[],[]
    for u in np.unique(rows):
        mask = rows == u
        ur, uc, ud = rows[mask], cols[mask], data[mask]
        n = len(ur)
        if n <= min_train:
            tr.extend(ur); tc.extend(uc); td.extend(ud)
        else:
            idx = np.random.permutation(n)
            nt = max(1, int(n * test_frac))
            ti, trai = idx[:nt], idx[nt:]
            tr.extend(ur[trai]); tc.extend(uc[trai]); td.extend(ud[trai])
            ter.extend(ur[ti]); tec.extend(uc[ti]); ted.extend(ud[ti])
    s = interactions.shape
    return coo_matrix((td, (tr, tc)), shape=s), coo_matrix((ted, (ter, tec)), shape=s)

def prepare_features_with_pca(book_ids, emb_dict, item_map, n_components):
    # Фильтруем только те книги, которые есть в item_map
    valid_ids = [bid for bid in book_ids if bid in item_map and bid in emb_dict]
    if not valid_ids:
        return None, 0.0
        
    embeddings = np.array([emb_dict[bid] for bid in valid_ids])
    
    if n_components >= embeddings.shape[1] or n_components == 768:
        # Без сжатия или размерность >= исходной
        reduced = embeddings
        var_explained = 1.0
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(embeddings)
        var_explained = svd.explained_variance_ratio_.sum()
    
    # Выравниваем под внутренние индексы LightFM
    n_items = len(item_map)
    d = reduced.shape[1]
    arr = np.zeros((n_items, d), dtype=np.float32)
    for bid, emb in zip(valid_ids, reduced):
        arr[item_map[bid]] = emb
    return csr_matrix(arr), var_explained

def train_and_eval(train_inter, test_inter, item_features, name, dim, var_exp):
    t0 = time.time()
    # Единые гиперпараметры для всех запусков
    model = LightFM(loss='warp', no_components=64, learning_rate=0.01,
                    item_alpha=1e-4, user_alpha=1e-4, max_sampled=5, random_state=42)
    
    model.fit(train_inter, sample_weight=train_inter, item_features=item_features, 
              epochs=EPOCHS, num_threads=N_THREADS, verbose=False)
    
    p = precision_at_k(model, test_inter, train_interactions=train_inter, 
                       item_features=item_features, k=TOP_K, num_threads=N_THREADS).mean()
    r = recall_at_k(model, test_inter, train_interactions=train_inter, 
                    item_features=item_features, k=TOP_K, num_threads=N_THREADS).mean()
    
    t_train = time.time() - t0
    print(f"   {name:<6} | P@{TOP_K}: {p:.4f} | R@{TOP_K}: {r:.4f} | Var: {var_exp:.3f} | Time: {t_train:.1f}s")
    return p, r, var_exp, t_train

def main():
    df_rat, all_book_ids = fetch_data()
    emb_dict = get_embeddings_custom(all_book_ids)
    if emb_dict is None: print("❌ Нет эмбеддингов"); return

    dataset = Dataset()
    dataset.fit(users=df_rat['user_id'].unique(), items=all_book_ids)
    interactions, _ = dataset.build_interactions((r.user_id, r.book_id, r.weight) for r in df_rat.itertuples())
    _, _, item_map, _ = dataset.mapping()
    train_inter, test_inter = split_interactions(interactions)
    print(f"📐 Train: {train_inter.nnz} | Test: {test_inter.nnz}\n")
    
    print(f"🔬 Запуск эксперимента: {len(PCA_DIMS)} конфигураций")
    print(f"{'Dim':<6} | {'Precision@10':<12} | {'Recall@10':<12} | {'Var.Expl.':<10} | {'Time'}")
    print("-"*60)
    
    results = []
    for dim in PCA_DIMS:
        item_features, var_exp = prepare_features_with_pca(all_book_ids, emb_dict, item_map, dim)
        if item_features is None: continue
        p, r, ve, t = train_and_eval(train_inter, test_inter, item_features, f"{dim}D", dim, var_exp)
        results.append({"dim": dim, "prec": p, "rec": r, "var_exp": ve, "time": t})
    
    # === ИТОГОВАЯ ТАБЛИЦА ===
    print("\n" + "="*70)
    print(f"{'Размерность':<12} | {'Precision@10':<14} | {'Recall@10':<14} | {'Вар. объясн.':<14} | {'Время (с)'}")
    print("-"*70)
    for r in results:
        print(f"{r['dim']:<12} | {r['prec']:<14.4f} | {r['rec']:<14.4f} | {r['var_exp']:<14.3f} | {r['time']:.1f}")
    print("="*70)
    
    # Лучшая по Recall
    best = max(results, key=lambda x: x['rec'])
    print(f"\n🏆 Лучшая по Recall@{TOP_K}: {best['dim']}D (Recall={best['rec']:.4f}, Var.Exp={best['var_exp']:.3f})")
    print("💡 Если метрики близки — выбирай меньшую размерность для скорости инференса.")

if __name__ == '__main__':
    main()
