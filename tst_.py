#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФИНАЛЬНАЯ СРАВНИТЕЛЬНАЯ ОЦЕНКА
Все размерности + Pure CF baseline + AUC
Оптимальные настройки: фильтрация (5,10), loss='warp', alpha=0.9
"""
import os
import numpy as np
import pandas as pd
import mysql.connector
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, recall_at_k, precision_at_k
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
MIN_USER, MIN_ITEM = 5, 10
TOP_K = 10
ALPHA = 0.9
EPOCHS = 200
N_THREADS = min(os.cpu_count() or 4, 8)
PCA_DIMS = [768, 512, 368, 256, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4]
# ==========================================

def fetch_and_filter():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    cur.execute("""SELECT ur.idUser AS user_id, ur.idBook AS book_id,
                          CASE WHEN ur.rating >= 7 THEN 1.0 WHEN ur.rating >= 5 THEN 0.5 ELSE 0.1 END AS weight
                   FROM userRating ur WHERE ur.rating IS NOT NULL""")
    df = pd.DataFrame(cur.fetchall())
    cur.execute("""SELECT b.id, b.name, b.description, GROUP_CONCAT(bmd.data SEPARATOR ' | ') AS tags
                   FROM books b LEFT JOIN booksMetaData bmd ON b.id = bmd.idBook AND bmd.type = 'tag'
                   GROUP BY b.id""")
    df_books = pd.DataFrame(cur.fetchall())
    cur.close(); conn.close()
    
    df = df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
    df['user_id'] = df['user_id'].astype(int)
    df['book_id'] = df['book_id'].astype(int)
    df['weight'] = df['weight'].astype(float)
    df_books['text'] = df_books.apply(lambda r: 
        BeautifulSoup(str(r['description']), "html.parser").get_text(separator=" ", strip=True) 
        if pd.notna(r['description']) and len(str(r['description']).strip()) > 10 
        else f"{r['name']}. Темы: {r['tags'] if pd.notna(r['tags']) else 'без жанра'}", axis=1)
    
    uc, ic = df['user_id'].value_counts(), df['book_id'].value_counts()
    df = df[(df['user_id'].isin(uc[uc>=MIN_USER].index)) & 
            (df['book_id'].isin(ic[ic>=MIN_ITEM].index))]
    print(f"✅ После фильтрации: {len(df)} взаимодействий, {df['user_id'].nunique()} юзеров, {df['book_id'].nunique()} книг")
    return df, df_books

def get_embeddings_dict(df_books):
    if os.path.exists(CUSTOM_CACHE):
        emb = np.load(CUSTOM_CACHE)
        return dict(zip(df_books['id'], emb))
    return None

def build_interactions(df):
    users, items = df['user_id'].unique(), df['book_id'].unique()
    umap = {u: i for i, u in enumerate(users)}
    imap = {it: i for i, it in enumerate(items)}
    r = [umap[x] for x in df['user_id']]
    c = [imap[x] for x in df['book_id']]
    d = df['weight'].values
    return coo_matrix((d, (r, c)), shape=(len(users), len(items))), umap, imap

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

def prepare_features(book_ids, emb_dict, item_map, n_components):
    valid = [bid for bid in book_ids if bid in item_map and bid in emb_dict]
    if not valid: return None, 0.0
    embeddings = np.array([emb_dict[bid] for bid in valid])
    if n_components >= embeddings.shape[1]:
        reduced = embeddings
        var_exp = 1.0
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(embeddings)
        var_exp = svd.explained_variance_ratio_.sum()
    n_items = len(item_map)
    arr = np.zeros((n_items, reduced.shape[1]), dtype=np.float32)
    for bid, emb in zip(valid, reduced):
        arr[item_map[bid]] = emb
    return csr_matrix(arr), var_exp

def norm_scores(s):
    mask = s > -1e9
    if not np.any(mask): return np.zeros_like(s, dtype=np.float64)
    mn, mx = s[mask].min(), s[mask].max()
    res = np.zeros_like(s, dtype=np.float64)
    if mx > mn: res[mask] = (s[mask] - mn) / (mx - mn)
    return res

def evaluate_hybrid(model, test_inter, train_inter, item_features, item_map, alpha=ALPHA, top_k=TOP_K):
    """Ручной расчёт Precision/Recall для гибрида. item_features передаётся явно в predict."""
    precisions, recalls = [], []
    n_items = item_features.shape[0]
    
    for u in np.unique(test_inter.row):
        # 🔥 ФИКС: явно передаём item_features в predict
        cf = model.predict(np.full(n_items, u, dtype=np.int32), np.arange(n_items), item_features=item_features)
        
        cand_idx = np.argsort(-cf)[:200]
        train_mask = train_inter.row == u
        train_items = train_inter.col[train_mask]
        valid_mask = ~np.isin(cand_idx, train_items)
        cand_idx = cand_idx[valid_mask]
        if len(cand_idx) == 0: continue
        
        # Профиль из исходных эмбеддингов (не из item_features, чтобы сохранить семантику)
        # Для упрощения берём среднее по лайкнутым книгам (веса опустим для стабильности)
        profile = np.zeros(item_features.shape[1], dtype=np.float32) # Placeholder, ниже возьмём из aligned
        # NOTE: в main мы передадим aligned_embs отдельно, здесь упростим для чистоты API
        # Но чтобы не ломать логику, вернёмся к original embeddings в main
        
        test_mask = test_inter.row == u
        test_items = set(test_inter.col[test_mask])
        if not test_items: continue
        
        # Упрощённая оценка для скорости (топ CF)
        # Полная гибридная логика вынесена в main для чистоты
        top = cand_idx[:top_k]
        hits = len(set(top) & test_items)
        precisions.append(hits / top_k)
        recalls.append(hits / len(test_items))
        
    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0

def main():
    print("📥 Загрузка данных...")
    df, df_books = fetch_and_filter()
    emb_dict = get_embeddings_dict(df_books)
    if emb_dict is None: print("❌ Кэш эмбеддингов не найден"); return
    
    train_inter_full, umap, item_map = build_interactions(df)
    train_inter, test_inter = split_interactions(train_inter_full)
    print(f"📐 Train: {train_inter.nnz} | Test: {test_inter.nnz}")
    
    # Precompute aligned embeddings for hybrid reranking
    n_items = len(item_map)
    aligned_embs = np.zeros((n_items, 768), dtype=np.float32)
    for bid, idx in item_map.items():
        if bid in emb_dict: aligned_embs[idx] = emb_dict[bid]
        
    print("\n🔬 Запуск финальной оценки...")
    print(f"{'Конфигурация':<15} | {'Dim':<6} | {'Prec@10':<10} | {'Recall@10':<10} | {'AUC':<10} | {'Var.Exp'}")
    print("-"*75)
    
    results = []
    
    # 1. Pure CF baseline (без фич)
    print("⏳ Pure CF (без фич)...", end=" ", flush=True)
    model_cf = LightFM(loss='warp', no_components=64, learning_rate=0.01,
                       item_alpha=1e-4, user_alpha=1e-4, max_sampled=5, random_state=42)
    model_cf.fit(train_inter, sample_weight=train_inter, epochs=EPOCHS, num_threads=N_THREADS, verbose=False)
    p_cf = precision_at_k(model_cf, test_inter, train_interactions=train_inter, k=TOP_K, num_threads=N_THREADS).mean()
    r_cf = recall_at_k(model_cf, test_inter, train_interactions=train_inter, k=TOP_K, num_threads=N_THREADS).mean()
    auc_cf = auc_score(model_cf, test_inter, train_interactions=train_inter, num_threads=N_THREADS, preserve_rows=True).mean()
    print(f"✅ Prec={p_cf:.4f}, Rec={r_cf:.4f}, AUC={auc_cf:.4f}")
    results.append({'config': 'Pure CF', 'dim': '—', 'prec': p_cf, 'rec': r_cf, 'auc': auc_cf, 'var': 1.0})
    
    # 2. Гибрид с разными размерностями
    for dim in PCA_DIMS:
        print(f"⏳ Hybrid {dim}D...", end=" ", flush=True)
        item_features, var_exp = prepare_features(df_books['id'], emb_dict, item_map, dim)
        if item_features is None: continue
        
        model = LightFM(loss='warp', no_components=64, learning_rate=0.01,
                        item_alpha=1e-4, user_alpha=1e-4, max_sampled=5, random_state=42)
        model.fit(train_inter, sample_weight=train_inter, item_features=item_features, 
                  epochs=EPOCHS, num_threads=N_THREADS, verbose=False)
        
        # Ручная оценка гибрида (CF + RAG reranking)
        p_h, r_h, auc_h = 0.0, 0.0, 0.0
        # Prec/Rec
        precisions, recalls = [], []
        for u in np.unique(test_inter.row):
            cf = model.predict(np.full(n_items, u, dtype=np.int32), np.arange(n_items), item_features=item_features)
            cand_idx = np.argsort(-cf)[:200]
            train_mask = train_inter.row == u
            train_items = train_inter.col[train_mask]
            valid_mask = ~np.isin(cand_idx, train_items)
            cand_idx = cand_idx[valid_mask]
            if len(cand_idx) == 0: continue
            
            # RAG профиль
            profile = aligned_embs[train_items].mean(axis=0)
            norm_p = np.linalg.norm(profile)
            if norm_p > 0: profile /= norm_p
            rag = aligned_embs[cand_idx] @ profile
            
            # Блендинг
            final = ALPHA * norm_scores(cf[cand_idx]) + (1 - ALPHA) * norm_scores(rag)
            top = cand_idx[np.argsort(-final)[:TOP_K]]
            
            test_mask = test_inter.row == u
            test_items = set(test_inter.col[test_mask])
            if not test_items: continue
            hits = len(set(top) & test_items)
            precisions.append(hits / TOP_K)
            recalls.append(hits / len(test_items))
            
        p_h = np.mean(precisions) if precisions else 0.0
        r_h = np.mean(recalls) if recalls else 0.0
        
        # AUC через lightfm
        auc_h = auc_score(model, test_inter, train_interactions=train_inter, 
                          item_features=item_features, num_threads=N_THREADS, preserve_rows=True).mean()
        
        print(f"✅ Prec={p_h:.4f}, Rec={r_h:.4f}, AUC={auc_h:.4f}")
        results.append({'config': f'Hybrid {dim}D', 'dim': str(dim), 'prec': p_h, 'rec': r_h, 'auc': auc_h, 'var': var_exp})
    
    # === ТАБЛИЦА ДЛЯ EXCEL (запятая как разделитель) ===
    print("\n" + "="*85)
    print("📊 ТАБЛИЦА ДЛЯ ВСТАВКИ В EXCEL (копируй блок ниже):")
    print("="*85)
    print("Конфигурация\tDim\tPrecision@10\tRecall@10\tAUC\tVar.Exp")
    for r in results:
        print(f"{r['config']}\t{r['dim']}\t{r['prec']:.4f}\t{r['rec']:.4f}\t{r['auc']:.4f}\t{r['var']:.3f}")
    print("="*85)
    
    best_rec = max(results, key=lambda x: x['rec'])
    best_auc = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 Лучшая по Recall: {best_rec['config']} (Recall={best_rec['rec']:.4f})")
    print(f"🏆 Лучшая по AUC: {best_auc['config']} (AUC={best_auc['auc']:.4f})")
    
    cf_rec = next(r['rec'] for r in results if r['config']=='Pure CF')
    cf_auc = next(r['auc'] for r in results if r['config']=='Pure CF')
    print(f"\n📈 Относительно Pure CF (Recall={cf_rec:.4f}, AUC={cf_auc:.4f}):")
    for r in results:
        if r['config'] != 'Pure CF':
            d_rec = r['rec'] - cf_rec
            d_auc = r['auc'] - cf_auc
            sign_rec = '+' if d_rec >= 0 else ''
            sign_auc = '+' if d_auc >= 0 else ''
            print(f"   {r['config']}: ΔRecall={sign_rec}{d_rec:.4f}, ΔAUC={sign_auc}{d_auc:.4f}")

if __name__ == '__main__':
    main()
