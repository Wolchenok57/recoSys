#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_dataset.py — Обработка Ficbook: токенизация + структура с шардированием
✅ Точный учёт места: os.path.getsize() по всем .npy файлам
✅ Прерываемость: пропускает уже обработанные книги + индекс-трекинг
✅ Лимит диска: точная проверка каждые CHECK_LIMIT_EVERY книг
✅ Статистика: info.json на книгу + total_info.json (агрегация без пересчёта)
✅ График: простая гистограмма каждые PLOT_INTERVAL книг
✅ Токенизация: через tok (BPE), сохранение как uint16 .npy
"""

import os
import sys
import json
import math
import shutil
import itertools
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testoBPE import BPE

# ==================== КОНСТАНТЫ ====================
# Лимит на размер токенизированных данных (.npy файлы) в байтах
# Установи False, чтобы обработать ВЕСЬ датасет без ограничений
DISK_LIMIT_BYTES = 10 * 1024**3  # 10 GB

# Как часто ПРОВЕРЯТЬ достижение лимита (в книгах)
CHECK_LIMIT_EVERY = 1000

# Как часто обновлять ГРАФИК распределения длин (в книгах)
PLOT_INTERVAL = 10_000

# Размер шарда (сколько книг в одной подпапке)
SHARD_SIZE = 10_000

# Базовая папка для вывода
OUTPUT_BASE = "/mnt/news/llm_ds/fics"

# Токен для пробела в твоём BPE
SPACE_TOKEN_ID = 173

# Dtype для сохранения токенов
NPY_DTYPE = np.uint16

# Как часто сбрасывать агрегированную статистику в total_info.json
TOTAL_INFO_FLUSH_EVERY = 500
# ====================================================

# Инициализация токенизатора ОДИН раз на процесс
tok = BPE()

def calculate_actual_disk_usage(base_dir: str) -> int:
    """
    Возвращает точный размер всех .npy файлов в директории в байтах.
    Использует os.path.getsize(), без эвристик и заголовков.
    """
    total_size = 0
    base = Path(base_dir)
    if not base.exists():
        return 0
    
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npy'):
                total_size += os.path.getsize(os.path.join(root, f))
    return total_size

def get_shard_path(book_idx: int, base: str) -> Path:
    shard_start = (book_idx // SHARD_SIZE) * SHARD_SIZE
    shard_end = shard_start + SHARD_SIZE - 1
    return Path(base) / f"{shard_start:05d}-{shard_end:05d}"

def get_book_path(book_idx: int, book_id: str, base: str) -> Path:
    return get_shard_path(book_idx, base) / f"book_{book_id}"

def is_book_processed(book_path: Path, expected_chapters: int) -> bool:
    if not book_path.exists():
        return False
    
    info_path = book_path / "info.json"
    if not info_path.exists():
        return False
    
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        if info.get('chapter_count') != expected_chapters:
            return False
        
        # Проверяем физические файлы глав
        for i in range(1, expected_chapters + 1):
            chap_file = book_path / f"chap{i}.npy"
            if not chap_file.exists() or chap_file.stat().st_size == 0:
                return False
        return True
    except Exception:
        return False

def draw_length_distribution(lengths: list[int], output_path: str):
    """
    Рисует сглаженную ЛИНИЮ распределения длин глав (не гистограмму!).
    """
    if not lengths:
        return
    
    from scipy.ndimage import gaussian_filter1d
    
    # Обрезаем экстремальные выбросы, чтобы график не растягивался
    max_len = min(np.percentile(lengths, 99), 10000)
    bin_size = 50  # 50 токенов на точку графика
    bins = np.arange(0, max_len + bin_size, bin_size)
    
    # Гистограмма для последующего сглаживания
    counts, _ = np.histogram(lengths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Сглаживание Гауссом (sigma=1.5 бина)
    counts_smooth = gaussian_filter1d(counts.astype(float), sigma=1.5)
    
    plt.figure(figsize=(12, 6))
    
    # Основная линия
    plt.plot(bin_centers, counts_smooth, color='#2E86AB', linewidth=2.5, label='Distribution')
    
    # Заливка под кривой для визуального веса
    plt.fill_between(bin_centers, counts_smooth, alpha=0.15, color='#2E86AB')
    
    # Статистика в углу
    stats = (
        f"Chapters: {len(lengths):,}\n"
        f"Mean: {np.mean(lengths):.0f}\n"
        f"Median: {np.median(lengths):.0f}\n"
        f"95%: {np.percentile(lengths, 95):.0f}"
    )
    plt.text(0.98, 0.98, stats, transform=plt.gca().transAxes,
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    plt.title('Chapter Length Distribution (tokens)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Chapter Length (tokens)', fontsize=11)
    plt.ylabel('Number of Chapters', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--', axis='both')
    plt.xlim(0, max_len)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def update_total_info(agg: dict, output_base: str):
    total_path = Path(output_base) / "total_info.json"
    tmp_path = str(total_path) + '.tmp'
    agg['last_updated'] = f"{agg['books_processed']} books processed"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, total_path)

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Восстановление индекса и агрегатов
    resume_idx = 0
    agg = {'total_chars': 0, 'total_spaces': 0, 'total_tokens': 0, 
           'total_npy_bytes': 0, 'books_processed': 0, 'chapters_processed': 0}
    
    idx_file = Path(OUTPUT_BASE) / "last_processed_idx.txt"
    total_info_file = Path(OUTPUT_BASE) / "total_info.json"
    
    if total_info_file.exists():
        try:
            with open(total_info_file, 'r', encoding='utf-8') as f:
                agg.update(json.load(f))
        except:
            pass
            
    if idx_file.exists():
        try:
            resume_idx = int(idx_file.read_text().strip())
        except:
            resume_idx = agg.get('books_processed', 0)
    
    # ТОЧНЫЙ расчёт места на диске при старте
    actual_disk_bytes = calculate_actual_disk_usage(OUTPUT_BASE)
    agg['total_npy_bytes'] = actual_disk_bytes
    print(f"🚀 Запуск обработки Ficbook", file=sys.stderr)
    print(f"📁 Вывод: {OUTPUT_BASE}", file=sys.stderr)
    print(f"💾 Лимит: {DISK_LIMIT_BYTES / 1024**3:.1f} GB" if DISK_LIMIT_BYTES else "💾 Лимит: НЕТ", file=sys.stderr)
    print(f"📥 Возобновление с книги #{resume_idx}. Занято на диске: {actual_disk_bytes / 1024**3:.2f} GB", file=sys.stderr)
    
    dataset = load_dataset('IlyaGusev/ficbook', split="train", streaming=True)
    all_lengths = []
    books_done = 0
    current_disk_bytes = actual_disk_bytes
    
    # tqdm с корректным total=None для streaming
    pbar = tqdm(desc="Books", unit="book", file=sys.stderr, total=None)
    pbar.update(resume_idx)
    
    limit_reached = False
    try:
        for idx, example in enumerate(dataset):
            if idx < resume_idx:
                continue
                
            if books_done % CHECK_LIMIT_EVERY == 0 and books_done > 0:
                current_disk_bytes = calculate_actual_disk_usage(OUTPUT_BASE)
                agg['total_npy_bytes'] = current_disk_bytes
                if DISK_LIMIT_BYTES and current_disk_bytes >= DISK_LIMIT_BYTES:
                    print(f"\n🎯 Лимит диска достигнут: {current_disk_bytes / 1024**3:.2f} GB", file=sys.stderr)
                    limit_reached = True
                    break
            
            book_id = str(example.get('url', f'book_{idx}')).rstrip('/').split('/')[-1] or f'book_{idx}'
            parts = example.get('parts', [])
            if not parts:
                pbar.update(1)
                continue
            
            book_path = get_book_path(idx, book_id, OUTPUT_BASE)
            
            if is_book_processed(book_path, len(parts)):
                pbar.update(1)
                books_done += 1
                continue
            
            book_path.mkdir(parents=True, exist_ok=True)
            
            book_chars = 0
            book_spaces = 0
            book_tokens = 0
            chapter_infos = []
            book_lengths = []
            
            for chap_idx, part in enumerate(parts, start=1):
                text = part.get('clean_text', '')
                if not text:
                    continue
                
                char_count = len(text)
                tokens = tok.encode(text)
                token_array = np.array(tokens, dtype=NPY_DTYPE)
                space_count = int(np.sum(token_array == SPACE_TOKEN_ID))
                
                chap_path = book_path / f"chap{chap_idx}.npy"
                np.save(chap_path, token_array)
                chap_size = os.path.getsize(chap_path)
                
                book_chars += char_count
                book_spaces += space_count
                book_tokens += len(tokens)
                book_lengths.append(len(tokens))
                current_disk_bytes += chap_size
                
                chapter_infos.append({
                    'chapter': chap_idx,
                    'title': part.get('title', f'Chapter {chap_idx}'),
                    'original_length_chars': char_count,
                    'space_token_count': space_count,
                    'token_count': len(tokens),
                    'npy_size_bytes': chap_size
                })
                agg['chapters_processed'] += 1
            
            if not chapter_infos:
                shutil.rmtree(book_path, ignore_errors=True)
                pbar.update(1)
                books_done += 1
                continue
            
            book_info = {
                'book_id': book_id,
                'book_idx': idx,
                'title': example.get('title', 'Unknown'),
                'source_parquet': example.get('url', '').split('/')[-2] if '/' in str(example.get('url', '')) else 'unknown',
                'source_record_id': book_id,
                'chapter_count': len(chapter_infos),
                'total_chars': book_chars,
                'total_spaces': book_spaces,
                'total_tokens': book_tokens,
                'chapters': chapter_infos
            }
            
            tmp_info = str(book_path / "info.json") + '.tmp'
            with open(tmp_info, 'w', encoding='utf-8') as f:
                json.dump(book_info, f, indent=2, ensure_ascii=False)
            os.replace(tmp_info, book_path / "info.json")
            
            agg['total_chars'] += book_chars
            agg['total_spaces'] += book_spaces
            agg['total_tokens'] += book_tokens
            agg['books_processed'] += 1
            
            all_lengths.extend(book_lengths)
            books_done += 1
            
            pbar.set_postfix({
                'GB': f"{current_disk_bytes / 1024**3:.2f}",
                'books': f"{books_done:,}"
            })
            pbar.update(1)
            
            # Периодическая запись агрегатов
            if books_done % TOTAL_INFO_FLUSH_EVERY == 0:
                update_total_info(agg, OUTPUT_BASE)
                idx_file.write_text(str(idx))
            
            # График
            if books_done % PLOT_INTERVAL == 0 and all_lengths:
                plot_path = Path(OUTPUT_BASE) / "graph_lengths.png"
                draw_length_distribution(all_lengths, str(plot_path))
                
            if limit_reached:
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Прервано", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
    finally:
        # Финальная синхронизация
        final_bytes = calculate_actual_disk_usage(OUTPUT_BASE)
        agg['total_npy_bytes'] = final_bytes
        update_total_info(agg, OUTPUT_BASE)
        if idx_file.exists():
            idx_file.write_text(str(resume_idx + books_done - (1 if limit_reached else 0)))
            
        if all_lengths:
            draw_length_distribution(all_lengths, str(Path(OUTPUT_BASE) / "graph_lengths_final.png"))
            
        pbar.close()
        print(f"\n✨ Завершено. Книг: {agg['books_processed']:,} | Место: {final_bytes / 1024**3:.2f} GB", file=sys.stderr)

if __name__ == '__main__':
    main()