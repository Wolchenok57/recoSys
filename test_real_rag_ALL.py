#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_all.py — Полный бенчмарк моделей на всех доступных данных
✅ Сравнивает несколько моделей
✅ Проходит все файлы ru-WANLI + синтетику
✅ Выводит сводную таблицу с детализацией
"""

import os, sys, json, time, random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ==================== НАСТРОЙКИ ====================
MODELS = {
    "logs1": "logs/model.pth",
    "logs2": "logs2/model.pth",
    "logs2_2": "logs2_2/model.pth",
    "logs3": "logs3_test_run/model_final.pth",
    "logs4": "logs4/model.pth",
    "logs5": "logs5/model.pth",
}

EMBEDDING_CHECKPOINT = "OLD/logs/checkpoint.pth"
WANLI_ROOT = "llm_datasets/ru-WANLI/data"
SYNTHETIC_PATH = "llm_datasets/My_RAG_DS/data.jsonl"

TEST_SIZE = 2000  # Сколько примеров брать из каждого файла
BATCH_SIZE = 16
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16

from model import RAGEncoder, BOS_TOKEN_ID
from testoBPE import BPE
# ===================================================

tok = BPE()

def print_mem():
    if DEVICE.type == 'cuda':
        alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
        print(f"[MEM] Alloc: {alloc:.0f}MB, Reserved: {reserved:.0f}MB", file=sys.stderr)

def load_model(model_path):
    """Загружает модель в режиме инференса"""
    model = RAGEncoder().to(DEVICE, dtype=DTYPE)
    
    if os.path.exists(EMBEDDING_CHECKPOINT):
        model.load_embeddings(EMBEDDING_CHECKPOINT)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        for k, v in cleaned.items():
            if v.dtype == torch.float32:
                cleaned[k] = v.half()
        model.load_state_dict(cleaned, strict=False)
    else:
        print(f"❌ {model_path} не найден!", file=sys.stderr)
        return None
        
    model.enable_gradient_checkpointing(False)
    model.eval()
    return model

def get_embeddings(model, texts, batch_size=BATCH_SIZE):
    """Получение эмбеддингов с прогресс-баром"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i+batch_size]
        
        batch_tokens = []
        for t in batch_texts:
            txt = t if isinstance(t, str) and t.strip() else "пусто"
            ids = tok.encode(txt)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = [BOS_TOKEN_ID] + ids
            if len(ids) > 1024:
                ids = ids[:1024]
            batch_tokens.append(torch.tensor(ids, dtype=torch.long))
        
        if not batch_tokens:
            continue
            
        batch_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_tokens, batch_first=True, padding_value=0
        ).to(DEVICE)
        
        with torch.no_grad():
            embs = model(batch_tensor)
            embeddings.append(embs.cpu())
    
    if not embeddings:
        return torch.zeros(0, 768)
    return torch.cat(embeddings, dim=0)

def evaluate_retrieval(emb_q, emb_p, emb_n):
    """Считает метрики: Accuracy, Recall@K, Margin"""
    emb_q = F.normalize(emb_q, p=2, dim=1)
    emb_p = F.normalize(emb_p, p=2, dim=1)
    emb_n = F.normalize(emb_n, p=2, dim=1)
    
    sim_pos = (emb_q * emb_p).sum(dim=1)
    sim_neg = (emb_q * emb_n).sum(dim=1)
    
    correct = (sim_pos > sim_neg).sum().item()
    accuracy = correct / len(emb_q) if len(emb_q) > 0 else 0
    
    # Recall@K
    POOL_SIZE = min(200, len(emb_q))
    if POOL_SIZE == 0:
        recalls = {1: 0, 5: 0, 10: 0, 50: 0}
    else:
        q_sub = emb_q[:POOL_SIZE]
        p_pool = emb_p[:POOL_SIZE]
        scores = q_sub @ p_pool.T
        
        recalls = {1: 0, 5: 0, 10: 0, 50: 0}
        for i in range(POOL_SIZE):
            q_scores = scores[i]
            sorted_idx = torch.argsort(q_scores, descending=True)
            try:
                rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
                for k in recalls:
                    if rank <= k:
                        recalls[k] += 1
            except:
                pass
        recalls = {k: v/POOL_SIZE for k, v in recalls.items()}
    
    margin = (sim_pos - sim_neg).mean().item()
    
    return {
        'accuracy': accuracy,
        'recall@1': recalls.get(1, 0),
        'recall@5': recalls.get(5, 0),
        'recall@10': recalls.get(10, 0),
        'recall@50': recalls.get(50, 0),
        'sim_pos_mean': sim_pos.mean().item(),
        'sim_pos_std': sim_pos.std().item(),
        'sim_neg_mean': sim_neg.mean().item(),
        'sim_neg_std': sim_neg.std().item(),
        'margin': margin,
        'n_samples': len(emb_q)
    }

def load_wanli_file(filepath, n_samples=TEST_SIZE):
    """Загружает файл ru-WANLI и возвращает пары для теста"""
    if not os.path.exists(filepath):
        return None, None
    
    df = pd.read_parquet(filepath)
    entail = df[df['label'] == 'entailment'][['premise', 'hypothesis']].dropna()
    contradict = df[df['label'] == 'contradiction'][['premise', 'hypothesis']].dropna()
    
    n = min(n_samples, len(entail), len(contradict))
    if n == 0:
        return None, None
        
    entail = entail.sample(n, random_state=42).reset_index(drop=True)
    contradict = contradict.sample(n, random_state=42).reset_index(drop=True)
    
    return entail['premise'].tolist(), entail['hypothesis'].tolist(), contradict['hypothesis'].tolist()

def load_synthetic_file(filepath, n_samples=TEST_SIZE):
    """Загружает синтетический датасет в формате query/context"""
    if not os.path.exists(filepath):
        return None, None, None
    
    queries, contexts, negatives = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    if len(lines) < n_samples * 2:
        return None, None, None
    
    # Для синтетики: query = anchor, context = positive, random other context = negative
    samples = random.sample(lines, n_samples)
    other_contexts = [l['context'] for l in lines if l not in samples]
    
    for s in samples:
        queries.append(s['query'])
        contexts.append(s['context'])
        negatives.append(random.choice(other_contexts) if other_contexts else s['context'])
    
    return queries, contexts, negatives

def run_benchmark(model_name, model_path):
    """Запускает бенчмарк для одной модели на всех данных"""
    print(f"\n🔬 Бенчмарк: {model_name} ({model_path})")
    print("="*80)
    
    results = []
    model = load_model(model_path)
    if model is None:
        print(f"❌ Не удалось загрузить {model_path}")
        return []
    
    # 1. ru-WANLI файлы
    wanli_files = ['train.parquet', 'val.parquet', 'test.parquet']
    for fname in wanli_files:
        fpath = os.path.join(WANLI_ROOT, fname)
        if not os.path.exists(fpath):
            continue
            
        print(f"\n📄 {fname}...")
        queries, positives, negatives = load_wanli_file(fpath)
        if queries is None:
            continue
        
        emb_q = get_embeddings(model, queries)
        emb_p = get_embeddings(model, positives)
        emb_n = get_embeddings(model, negatives)
        
        metrics = evaluate_retrieval(emb_q, emb_p, emb_n)
        metrics['model'] = model_name
        metrics['dataset'] = 'ru-WANLI'
        metrics['file'] = fname
        metrics['type'] = 'official'
        results.append(metrics)
        
        print(f"   ✅ Accuracy: {metrics['accuracy']*100:.1f}% | Recall@1: {metrics['recall@1']*100:.1f}% | Margin: {metrics['margin']:.3f}")
    
    # 2. Синтетический датасет
    if os.path.exists(SYNTHETIC_PATH):
        print(f"\n📄 My_RAG_DS/data.jsonl...")
        queries, positives, negatives = load_synthetic_file(SYNTHETIC_PATH)
        if queries:
            emb_q = get_embeddings(model, queries)
            emb_p = get_embeddings(model, positives)
            emb_n = get_embeddings(model, negatives)
            
            metrics = evaluate_retrieval(emb_q, emb_p, emb_n)
            metrics['model'] = model_name
            metrics['dataset'] = 'My_RAG_DS'
            metrics['file'] = 'data.jsonl'
            metrics['type'] = 'synthetic'
            results.append(metrics)
            
            print(f"   ✅ Accuracy: {metrics['accuracy']*100:.1f}% | Recall@1: {metrics['recall@1']*100:.1f}% | Margin: {metrics['margin']:.3f}")
    
    del model
    torch.cuda.empty_cache()
    return results

def print_summary_table(all_results):
    """Выводит сводную таблицу результатов"""
    if not all_results:
        print("\n❌ Нет данных для вывода")
        return
    
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*120)
    print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*120)
    
    # Детальная таблица по файлам
    print("\n🔹 ПО ФАЙЛАМ:")
    print("-"*120)
    detail_cols = ['model', 'dataset', 'file', 'type', 'n_samples', 'accuracy', 'recall@1', 'recall@5', 'recall@10', 'margin']
    df_detail = df[detail_cols].copy()
    df_detail['accuracy'] = (df_detail['accuracy'] * 100).round(1).astype(str) + '%'
    df_detail['recall@1'] = (df_detail['recall@1'] * 100).round(1).astype(str) + '%'
    df_detail['recall@5'] = (df_detail['recall@5'] * 100).round(1).astype(str) + '%'
    df_detail['recall@10'] = (df_detail['recall@10'] * 100).round(1).astype(str) + '%'
    df_detail['margin'] = df_detail['margin'].round(3).astype(str)
    print(df_detail.to_string(index=False))
    
    # Агрегация по датасету
    print("\n🔹 ПО ДАТАСЕТАМ (среднее по файлам):")
    print("-"*120)
    agg_dataset = df.groupby(['model', 'dataset', 'type']).agg({
        'n_samples': 'sum',
        'accuracy': 'mean',
        'recall@1': 'mean',
        'recall@5': 'mean',
        'recall@10': 'mean',
        'margin': 'mean'
    }).reset_index()
    
    agg_dataset['accuracy'] = (agg_dataset['accuracy'] * 100).round(1).astype(str) + '%'
    agg_dataset['recall@1'] = (agg_dataset['recall@1'] * 100).round(1).astype(str) + '%'
    agg_dataset['recall@5'] = (agg_dataset['recall@5'] * 100).round(1).astype(str) + '%'
    agg_dataset['recall@10'] = (agg_dataset['recall@10'] * 100).round(1).astype(str) + '%'
    agg_dataset['margin'] = agg_dataset['margin'].round(3).astype(str)
    print(agg_dataset.to_string(index=False))
    
    # Общая статистика по моделям
    print("\n🔹 ОБЩАЯ СТАТИСТИКА ПО МОДЕЛЯМ:")
    print("-"*120)
    agg_model = df.groupby('model').agg({
        'n_samples': 'sum',
        'accuracy': 'mean',
        'recall@1': 'mean',
        'recall@5': 'mean',
        'recall@10': 'mean',
        'recall@50': 'mean',
        'margin': 'mean'
    }).reset_index()
    
    agg_model['accuracy'] = (agg_model['accuracy'] * 100).round(1).astype(str) + '%'
    agg_model['recall@1'] = (agg_model['recall@1'] * 100).round(1).astype(str) + '%'
    agg_model['recall@5'] = (agg_model['recall@5'] * 100).round(1).astype(str) + '%'
    agg_model['recall@10'] = (agg_model['recall@10'] * 100).round(1).astype(str) + '%'
    agg_model['recall@50'] = (agg_model['recall@50'] * 100).round(1).astype(str) + '%'
    agg_model['margin'] = agg_model['margin'].round(3).astype(str)
    print(agg_model.to_string(index=False))
    
    # Сохранение в CSV для дальнейшего анализа
    output_path = "benchmark_results.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n💾 Детальные результаты сохранены в {output_path}")
    
    # Интерпретация
    print("\n🎯 ИНТЕРПРЕТАЦИЯ:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        avg_acc = model_df['accuracy'].mean() * 100
        avg_r1 = model_df['recall@1'].mean() * 100
        avg_margin = model_df['margin'].mean()
        
        status = "✅ ОТЛИЧНО" if avg_acc > 85 else ("✅ ХОРОШО" if avg_acc > 70 else "⚠️ ТРЕБУЕТ ДООбучения")
        print(f"   {model}: Accuracy={avg_acc:.1f}% | Recall@1={avg_r1:.1f}% | Margin={avg_margin:.3f} → {status}")

# ==================== MAIN ====================
def main():
    print_mem()
    print(f"🚀 Запуск полного бенчмарка")
    print(f"📁 MODELS: {list(MODELS.keys())}")
    print(f"📁 WANLI: {WANLI_ROOT}")
    print(f"📁 SYNTHETIC: {SYNTHETIC_PATH}")
    
    all_results = []
    
    for model_name, model_path in MODELS.items():
        results = run_benchmark(model_name, model_path)
        all_results.extend(results)
    
    print_summary_table(all_results)
    
    print("\n✨ Бенчмарк завершён!")

if __name__ == '__main__':
    main()