#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rag_dataset_stable.py — Стабильная версия без дедлоков
10k чанков, упорядоченно, с ресумом, параллельно, БЕЗ ОЧЕРЕДЕЙ.
"""

# ==================== КОНСТАНТА ====================
TARGET_CHUNKS = 10000  # 🔥 Сколько чанков обработать
# ==================================================

import os, json, requests, numpy as np, time, re, pickle, signal, sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testoBPE import BPE
tok = BPE()

# ==================== КОНФИГУРАЦИЯ ====================
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"

SOURCE_DIR = "/mnt/news/llm_ds/fics"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_datasets/My_RAG_DS")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.pkl")

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS = 50
MAX_WORKERS = 16
CHECKPOINT_EVERY = 100

# 🔥 ТАЙМАУТЫ (критично!)
REQUEST_TIMEOUT = 60  # секунд на один запрос к LM Studio
FUTURE_TIMEOUT = 120  # секунд на ожидание завершения всех задач
HEALTH_CHECK_EVERY = 100  # проверять LM Studio каждые N запросов

SYSTEM_PROMPT = """Ты помогаешь обучать поисковый движок для библиотеки фанфиков.
Придумай 3-5 поисковых запросов на русском, по которым пользователь мог бы найти этот отрывок.
Разнообразие: один вопрос, одно описание, одни ключевые слова.
Верни ТОЛЬКО JSON-список строк, без пояснений. Пример: ["запрос 1", "запрос 2"]"""

# ==================== ГЛОБАЛЬНЫЕ ФЛАГИ ====================
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️  Получен сигнал завершения. Завершаем текущие задачи и сохраняем прогресс...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==================== ФУНКЦИИ ====================

def sanitize_text(text):
    return text.replace('\u00A0', ' ').replace('\u200B', '')[:4000]

def decode_tokens(tokens):
    try:
        return tok.decode(tokens.tolist())
    except:
        return ""

def split_into_chunks(text, chunk_size=CHUNK_SIZE_TOKENS):
    tokens = tok.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    stride = chunk_size - OVERLAP_TOKENS
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(tok.decode(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
    return chunks

def query_llm(context_text, source_path, chunk_idx, worker_id):
    """Запрос к LM Studio с таймаутами и санитизацией"""
    if shutdown_requested:
        return None
    
    clean_context = sanitize_text(context_text)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{clean_context}\n\nСписок запросов в формате JSON:"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        # 🔥 Таймаут на запрос
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
        
        if resp.status_code != 200:
            return {'error': f"HTTP {resp.status_code}: {resp.text[:100]}"}
        
        result = resp.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Парсинг JSON
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return {'error': "No JSON found"}
        
        queries = json.loads(match.group(0))
        if not isinstance(queries, list):
            return {'error': "Response is not a list"}
        
        return {
            'success': True,
            'queries': [q for q in queries if isinstance(q, str) and len(q.strip()) > 3],
            'context': clean_context,
            'source': source_path,
            'chunk_idx': chunk_idx,
            'unique_id': f"{Path(source_path).name}_{chunk_idx}"
        }
            
    except requests.exceptions.Timeout:
        return {'error': f"Request timeout ({REQUEST_TIMEOUT}s)"}
    except requests.exceptions.ConnectionError:
        return {'error': "Connection refused"}
    except json.JSONDecodeError:
        return {'error': "JSON decode error"}
    except Exception as e:
        return {'error': f"{type(e).__name__}: {str(e)[:50]}"}

def check_lm_studio_health():
    """Быстрая проверка, отвечает ли LM Studio"""
    try:
        resp = requests.post(
            LM_STUDIO_URL,
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
            timeout=10
        )
        return resp.status_code == 200
    except:
        return False

def collect_chunks_ordered(source_dir, target_count, skip_unique_ids=None):
    """Собирает чанки в упорядоченном виде, пропуская обработанные"""
    chunks = []
    root = Path(source_dir)
    if not root.exists():
        return chunks
    
    shards = sorted([d for d in root.iterdir() if d.is_dir() and '-' in d.name])
    
    for shard in shards:
        if len(chunks) >= target_count or shutdown_requested:
            break
        books = sorted([b for b in shard.iterdir() if b.is_dir() and b.name.startswith('book_')])
        for book in books:
            if len(chunks) >= target_count or shutdown_requested:
                break
            chapters = sorted(book.glob('chap*.npy'), key=lambda p: int(p.stem.replace('chap', '')))
            for chap in chapters:
                if len(chunks) >= target_count or shutdown_requested:
                    break
                
                unique_prefix = Path(chap).name
                if skip_unique_ids and any(uid.startswith(unique_prefix) for uid in skip_unique_ids):
                    continue
                
                try:
                    tokens = np.load(chap).astype(np.int64)
                    text = decode_tokens(tokens)
                    if len(text) < 100:
                        continue
                    for i, chunk_text in enumerate(split_into_chunks(text)):
                        if len(chunks) >= target_count or shutdown_requested:
                            break
                        if len(chunk_text.strip()) < 50:
                            continue
                        chunks.append({
                            'text': chunk_text,
                            'source': str(chap),
                            'chunk_idx': i,
                            'unique_id': f"{unique_prefix}_{i}"
                        })
                except Exception as e:
                    tqdm.write(f"⚠️ Ошибка чтения {chap}: {e}")
    
    return chunks

def save_batch(results_batch, output_file, checkpoint_data, checkpoint_file, pbar):
    """Сохраняет пакет результатов в файл + чекпоинт"""
    saved = 0
    for item in results_batch:
        if item and item.get('success') and item.get('queries'):
            with open(output_file, 'a', encoding='utf-8') as f:
                for q in item['queries']:
                    entry = {
                        "query": q.strip(),
                        "context": item['context'].strip(),
                        "source": item['source'],
                        "chunk_idx": item['chunk_idx']
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f.flush()
                    saved += 1
                    pbar.update(1)
            
            # Обновляем чекпоинт
            checkpoint_data['processed_unique_ids'].add(item['unique_id'])
            checkpoint_data['last_saved_pairs'] = checkpoint_data.get('last_saved_pairs', 0) + saved
    
    # Сохраняем чекпоинт периодически
    if saved >= CHECKPOINT_EVERY:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    return saved

# ==================== MAIN ====================

def main():
    global shutdown_requested
    tok = BPE()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 Генерация | Целевые чанки: {TARGET_CHUNKS} | Workers: {MAX_WORKERS}")
    print(f"🔌 LM Studio: {LM_STUDIO_URL}")
    
    # Загрузка чекпоинта
    checkpoint_data = {'last_saved_pairs': 0, 'processed_unique_ids': set()}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                loaded = pickle.load(f)
                checkpoint_data.update(loaded)
                print(f"♻️  Продолжаем: {checkpoint_data['last_saved_pairs']} пар сохранено")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить чекпоинт: {e}")
    
    # Сбор чанков
    print("📦 Сбор чанков...")
    chunks = collect_chunks_ordered(
        SOURCE_DIR, 
        TARGET_CHUNKS,
        skip_unique_ids=checkpoint_data['processed_unique_ids']
    )
    
    if not chunks:
        print("✅ Все целевые чанки уже обработаны!")
        return
    
    print(f"✅ Чанков к обработке: {len(chunks)}")
    
    # Прогресс-бар
    initial = checkpoint_data.get('last_saved_pairs', 0)
    pbar = tqdm(total=len(chunks) * 3, initial=initial, desc="Генерация", unit="пар", dynamic_ncols=True)
    
    # 🔥 Обработка БЕЗ очередей: результаты собираем в список, пишем пачками
    results_buffer = []
    processed_count = 0
    health_check_counter = 0
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Отправляем все задачи
            future_to_chunk = {
                executor.submit(query_llm, c['text'], c['source'], c['chunk_idx'], i % MAX_WORKERS): c
                for i, c in enumerate(chunks)
            }
            
            # 🔥 Обрабатываем результаты по мере готовности, с таймаутом
            for future in as_completed(future_to_chunk, timeout=FUTURE_TIMEOUT):
                if shutdown_requested:
                    print("\n⚠️  Завершение по сигналу...")
                    break
                
                chunk = future_to_chunk[future]
                
                try:
                    result = future.result(timeout=10)  # 🔥 Таймаут на получение результата
                    if result:
                        results_buffer.append(result)
                        processed_count += 1
                    
                    # Периодическая запись буфера
                    if len(results_buffer) >= MAX_WORKERS * 2:
                        save_batch(results_buffer, OUTPUT_FILE, checkpoint_data, CHECKPOINT_FILE, pbar)
                        results_buffer = []
                    
                    # Проверка здоровья LM Studio
                    health_check_counter += 1
                    if health_check_counter % HEALTH_CHECK_EVERY == 0:
                        if not check_lm_studio_health():
                            tqdm.write("⚠️  LM Studio не отвечает, ждём 10с...")
                            time.sleep(10)
                            if not check_lm_studio_health():
                                tqdm.write("❌  LM Studio всё ещё не отвечает, пропускаем следующие 10 запросов")
                                for _ in range(10):
                                    pbar.update(3)  # пропускаем ~3 пары
                                
                except FuturesTimeoutError:
                    tqdm.write(f"⚠️  Таймаут ожидания задачи для {chunk['unique_id']}")
                except Exception as e:
                    tqdm.write(f"⚠️  Ошибка обработки {chunk['unique_id']}: {e}")
                
                # Обновляем прогресс в описании
                elapsed = time.time() - getattr(main, 'start_time', time.time())
                if elapsed > 0:
                    rate = (checkpoint_data.get('last_saved_pairs', 0) + len(results_buffer)*3) / elapsed
                    pbar.set_postfix(rate=f"{rate:.1f}пар/с")
            
            # 🔥 Финальная запись оставшегося буфера
            if results_buffer:
                save_batch(results_buffer, OUTPUT_FILE, checkpoint_data, CHECKPOINT_FILE, pbar)
                
    except FuturesTimeoutError:
        print(f"\n⚠️  Таймаут ожидания задач ({FUTURE_TIMEOUT}с). Некоторые запросы не завершены.")
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
    finally:
        # 🔥 Гарантированная запись чекпоинта при выходе
        if results_buffer:
            save_batch(results_buffer, OUTPUT_FILE, checkpoint_data, CHECKPOINT_FILE, pbar)
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        pbar.close()
    
    print(f"\n🎉 Завершено! Всего пар: {checkpoint_data['last_saved_pairs']}")
    print(f"📂 Файл: {OUTPUT_FILE}")
    print(f"💡 Для продолжения: просто запусти скрипт снова")

if __name__ == "__main__":
    main.start_time = time.time()
    main()