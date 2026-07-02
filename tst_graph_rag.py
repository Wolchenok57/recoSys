#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_lengths_rag.py — Анализ распределения длин в My_RAG_DS
Сохраняет график как graph2.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from testoBPE import BPE

# ==================== КОНСТАНТЫ ====================
DATA_PATH = "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/My_RAG_DS/data.jsonl"
OUTPUT_PATH = "graph2.png"
MAX_LEN = 4096  # Верхняя граница для обрезки выбросов

# ==================== АНАЛИЗ ====================

def analyze_dataset(path):
    tok = BPE()
    
    query_lengths = []
    context_lengths = []
    
    print(f"📖 Чтение {path}...")
    
    # Считаем строки для прогресс-бара
    with open(path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Токенизация", unit="пар"):
            item = json.loads(line)
            
            q_len = len(tok.encode(item['query']))
            c_len = len(tok.encode(item['context']))
            
            query_lengths.append(q_len)
            context_lengths.append(c_len)
    
    return np.array(query_lengths), np.array(context_lengths)

def plot_distributions(query_lengths, context_lengths, output_path):
    # Обрезаем на 99-м квантиле
    q_cap = np.percentile(query_lengths, 99)
    c_cap = np.percentile(context_lengths, 99)
    
    query_clipped = np.clip(query_lengths, 0, q_cap)
    context_clipped = np.clip(context_lengths, 0, c_cap)
    
    # Определяем тип графика: столбцы если <10 уникальных значений
    def get_plot_type(data):
        return 'bar' if len(np.unique(data)) < 10 else 'line'
    
    query_type = get_plot_type(query_clipped)
    context_type = get_plot_type(context_clipped)
    
    # Создаём график: два подграфика в строку
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Распределение длин в токенах (testoBPE) — My_RAG_DS', fontsize=14, fontweight='bold')
    
    # Подграфик 1: Запросы (queries)
    if query_type == 'bar':
        counts = np.bincount(query_clipped.astype(int))
        bins = np.arange(len(counts))
        ax1.bar(bins, counts, width=0.8, alpha=0.7, color='steelblue', edgecolor='black')
    else:
        hist, bins = np.histogram(query_clipped, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax1.plot(bin_centers, hist, linewidth=2, color='steelblue')
        ax1.fill_between(bin_centers, hist, alpha=0.3, color='steelblue')
    
    ax1.set_xlabel('Длина запроса (токены)')
    ax1.set_ylabel('Частота')
    ax1.set_title(f'Запросы (99% квантиль: {q_cap:.0f})\nТип: {query_type}')
    ax1.grid(alpha=0.3, axis='y')
    ax1.axvline(x=np.median(query_clipped), color='red', linestyle='--', label=f'Медиана: {np.median(query_clipped):.0f}')
    ax1.legend()
    
    # Подграфик 2: Контексты (contexts)
    if context_type == 'bar':
        counts = np.bincount(context_clipped.astype(int))
        bins = np.arange(len(counts))
        ax2.bar(bins, counts, width=0.8, alpha=0.7, color='coral', edgecolor='black')
    else:
        hist, bins = np.histogram(context_clipped, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(bin_centers, hist, linewidth=2, color='coral')
        ax2.fill_between(bin_centers, hist, alpha=0.3, color='coral')
    
    ax2.set_xlabel('Длина контекста (токены)')
    ax2.set_ylabel('Частота')
    ax2.set_title(f'Контексты (99% квантиль: {c_cap:.0f})\nТип: {context_type}')
    ax2.grid(alpha=0.3, axis='y')
    ax2.axvline(x=np.median(context_clipped), color='red', linestyle='--', label=f'Медиана: {np.median(context_clipped):.0f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'query': {'min': int(np.min(query_lengths)), 'max': int(np.max(query_lengths)), 
                  'median': float(np.median(query_lengths)), 'mean': float(np.mean(query_lengths)),
                  'q99': float(q_cap)},
        'context': {'min': int(np.min(context_lengths)), 'max': int(np.max(context_lengths)), 
                    'median': float(np.median(context_lengths)), 'mean': float(np.mean(context_lengths)),
                    'q99': float(c_cap)}
    }

# ==================== MAIN ====================

def main():
    print(f"🔍 Анализ распределения длин: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Файл не найден: {DATA_PATH}")
        return
    
    query_lengths, context_lengths = analyze_dataset(DATA_PATH)
    
    print(f"\n📊 Статистика:")
    stats = plot_distributions(query_lengths, context_lengths, OUTPUT_PATH)
    
    print(f"\n🔤 Запросы:")
    print(f"   Мин: {stats['query']['min']:,} | Макс: {stats['query']['max']:,} | Медиана: {stats['query']['median']:.0f} | Среднее: {stats['query']['mean']:.0f} | 99%: {stats['query']['q99']:.0f}")
    
    print(f"\n📝 Контексты:")
    print(f"   Мин: {stats['context']['min']:,} | Макс: {stats['context']['max']:,} | Медиана: {stats['context']['median']:.0f} | Среднее: {stats['context']['mean']:.0f} | 99%: {stats['context']['q99']:.0f}")
    
    print(f"\n✅ График сохранён: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()