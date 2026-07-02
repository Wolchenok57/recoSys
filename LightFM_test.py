#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BookRecs API v6.5 — Клиентская фильтрация книг с картинками.
Сервер возвращает все найденные книги, клиент сам решает, какие показывать.
"""
import os
import sys
import re
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Set
import numpy as np
import joblib
import mysql.connector
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("bookrecs")

DB_CONFIG = {
    "host": "192.168.0.113",
    "user": "debservak",
    "password": "ТвойПароль123",
    "database": "kursach",
    "charset": "utf8mb4",
    "use_pure": True
}

HYBRID_MODEL_PATH = "hy_model.pkl"
CF_MODEL_PATH = "cf_model.pkl"
RAG_MODEL_PATH = "logs6_from_3_using_2_3/model.pth"
SQL_MODEL_PATH = "/mnt/news/buffer/OLD/logs8/model_final.pth"
DIALOG_MODEL_PATH = "/mnt/news/buffer/OLD/logs5/model_final_tuned.pth"

EMBEDDINGS_PATHS = [
    "book_embeddings_full_768.npy",
    "book_embeddings_custom_768.npy",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "book_embeddings_full_768.npy"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "book_embeddings_custom_768.npy"),
]

BOS_TOKEN_ID = 1
MAX_QUERY_LEN = 1024
TEMP_SQL, TOP_K_SQL, MAX_SQL = 0.3, 20, 512
TEMP_DIALOG, TOP_K_DIALOG, MAX_DIALOG = 0.8, 50, 256

SQL_PROMPT = """Ты — ассистент, специализирующийся на написании SQL-запросов для базы книг.
Отвечай ТОЛЬКО корректным SQL после слова "SQL:".
Используй таблицу books и правильные JOIN."""
DIALOG_PROMPT = "Твое имя Стас. Ты - опытный писатель сценариев. Твоя любимая тема - книги."

KEYWORD_RULES = {
    "recommend": ["посоветуй", "порекомендуй", "мне нравится", "похож", "подбери", "что почитать", "дай список"],
    "search": ["найди", "поиск", "о чём", "про что", "жанр", "тема", "смысл", "семантический"],
    "sql": ["сколько", "кто написал", "когда вышла", "покажи все", "статистика", "какие книги", "автор", "год"],
    "dialog": ["привет", "как дела", "объясни", "почему", "что такое", "помоги", "спасибо", "расскажи"]
}

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = None
    top_k: int = 6
    filter_images: bool = False   # клиент использует, сервер игнорирует

class DirectRecommendRequest(BaseModel):
    user_id: int
    top_k: int = 6

class SearchRequest(BaseModel):
    text: str
    top_k: int = 6
    filter_images: bool = False

class SQLRequest(BaseModel):
    question: str
    execute: bool = False
    top_k: int = 10

class DialogRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    history: Optional[List[dict]] = None

class CFRecommendRequest(BaseModel):
    user_id: int
    top_k: int = 6

class GenerateEmbeddingsRequest(BaseModel):
    batch_size: int = 64
    force: bool = False

class TextSearchRequest(BaseModel):
    text: str
    top_k: int = 10   # сколько названий вернуть

# Глобальные переменные
hy_model = None
hy_item_map = None
hy_user_map = None
hy_item_features = None
cf_model = None
cf_item_map = None
cf_user_map = None
book_embeddings = None
book_meta = {}          # {id: {"title":..., "desc":...}}
_book_ids_ordered = []
rag_model = None
rag_tokenizer = None
rag_device = None
rag_dtype = None
sql_model = None
sql_tokenizer = None
sql_device = None
sql_dtype = None
dialog_model = None
dialog_tokenizer = None
dialog_device = None
dialog_dtype = None
embeddings_generated = False

def route_intent(q: str) -> str:
    q_lower = q.lower()
    for intent, keywords in KEYWORD_RULES.items():
        if any(k in q_lower for k in keywords):
            return intent
    return "recommend"

def extract_sql_from_answer(answer: str) -> str:
    answer = answer.strip()
    for line in answer.split('\n'):
        if 'SQL:' in line.upper():
            return clean_sql(line.split('SQL:', 1)[1].strip())
    if answer.upper().startswith('SELECT'):
        return clean_sql(answer)
    m = re.search(r'(SELECT\s+[\w\s\.\,\*\(\)\'\"%]+?\s+FROM\s+[\w\s\.\,\'\"]+?(?:\s+WHERE\s+.*?|\s+GROUP\s+BY\s+.*?|\s+ORDER\s+BY\s+.*?|\s+LIMIT\s+.*?|\s+HAVING\s+.*?|;|$))', answer, re.IGNORECASE | re.DOTALL)
    return clean_sql(m.group(1)) if m else ""

def clean_sql(sql: str) -> str:
    for tag in ['<[EOS]>', '<[USR]>', '<[BOT]>', '<[SYS]>', '<[BOS]>']:
        sql = sql.replace(tag, '')
    sql = ' '.join(sql.split()).rstrip(';').strip()
    return sql if sql.upper().startswith('SELECT') else ""

def execute_sql_safe(db_conn, sql: str, limit: int = 100):
    if not sql or not sql.upper().startswith('SELECT'):
        return None, "Только SELECT разрешен"
    if 'LIMIT' not in sql.upper():
        sql = sql.rstrip(';') + f" LIMIT {limit}"
    try:
        cursor = db_conn.cursor(dictionary=True)
        cursor.execute(sql)
        res = cursor.fetchall()
        for row in res:
            for k, v in row.items():
                if isinstance(v, (bytes, bytearray)):
                    row[k] = v.decode('utf-8', errors='ignore')
                elif isinstance(v, str) and len(v) > 200:
                    row[k] = v[:200] + "..."
        return res, None
    except Exception as e:
        return None, str(e)[:200]

def generate_response(model, tok, input_ids, max_new_tokens, temperature, top_k, device, dtype):
    model.eval()
    input_ids = input_ids.to(device)
    eos_encoded = tok.encode("<[EOS]>")
    eos_id = eos_encoded[0] if eos_encoded else None
    generated = input_ids.clone()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype if device.type == 'cuda' else torch.float32):
        for _ in range(max_new_tokens):
            seq_len = generated.shape[1]
            inputs = generated[:, -1024:] if seq_len > 1024 else generated
            logits, _ = model(inputs)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                indices_to_remove = torch.ones_like(next_token_logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, topk_indices, False)
                next_token_logits[indices_to_remove] = -float("Inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token_id), dim=1)
            if eos_id is not None and next_token_id.item() == eos_id:
                break
    return generated[0]

def decode_response(tok, output_ids_1d, input_len):
    response_ids = output_ids_1d[input_len:].tolist()
    try:
        return tok.decode(response_ids)
    except:
        return str(response_ids)

def load_hybrid_model():
    global hy_model, hy_item_map, hy_user_map, hy_item_features
    if hy_model is not None:
        return hy_model, hy_item_map, hy_user_map, hy_item_features
    logger.info("📦 Загрузка гибридной LightFM модели...")
    artifacts = joblib.load(HYBRID_MODEL_PATH)
    hy_model = artifacts["model"]
    hy_item_map = artifacts["item_map"]
    hy_user_map = artifacts["user_map"]
    hy_item_features = artifacts["item_features"]
    logger.info(f"✅ Гибридная модель загружена: {len(hy_user_map)} юзеров, {len(hy_item_map)} книг")
    return hy_model, hy_item_map, hy_user_map, hy_item_features

def load_cf_model():
    global cf_model, cf_item_map, cf_user_map
    if cf_model is not None:
        return cf_model, cf_item_map, cf_user_map
    logger.info("📦 Загрузка Pure CF модели...")
    artifacts = joblib.load(CF_MODEL_PATH)
    cf_model = artifacts["model"]
    cf_item_map = artifacts["item_map"]
    cf_user_map = artifacts["user_map"]
    logger.info(f"✅ Pure CF модель загружена: {len(cf_user_map)} юзеров, {len(cf_item_map)} книг")
    return cf_model, cf_item_map, cf_user_map

def load_embeddings_and_metadata():
    global book_embeddings, _book_ids_ordered, book_meta
    if book_embeddings is not None:
        return True
    logger.info("📖 Загрузка метаданных книг из БД...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, LEFT(description, 500) as description FROM books ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    for r in rows:
        book_meta[r["id"]] = {"title": r["name"], "desc": r["description"] or ""}
    _book_ids_ordered = [r["id"] for r in rows]
    logger.info(f"✅ Метаданные загружены: {len(book_meta)} книг")
    
    for path in EMBEDDINGS_PATHS:
        if os.path.exists(path):
            try:
                logger.info(f"📂 Загрузка эмбеддингов из {path}...")
                book_embeddings = np.load(path)
                norms = np.linalg.norm(book_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                book_embeddings = book_embeddings / norms
                logger.info(f"✅ Эмбеддинги загружены: {book_embeddings.shape}")
                return True
            except Exception as e:
                logger.warning(f"Ошибка загрузки из {path}: {e}")
    logger.warning("⚠️ Эмбеддинги не найдены. Семантический поиск будет работать в fallback-режиме.")
    book_embeddings = None
    return False

def load_rag_model():
    global rag_model, rag_tokenizer, rag_device, rag_dtype
    if rag_model is not None:
        return rag_model, rag_tokenizer
    logger.info("🤖 Загрузка RAG-модели...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import RAGEncoder
    from testoBPE import BPE
    rag_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rag_dtype = torch.float16
    rm = RAGEncoder().to(rag_device, dtype=rag_dtype)
    ckpt = torch.load(RAG_MODEL_PATH, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): (v.half() if v.dtype == torch.float32 else v) if isinstance(v, torch.Tensor) else v for k, v in ckpt.items()}
    rm.load_state_dict(state, strict=False)
    rm.enable_gradient_checkpointing(False)
    rm.eval()
    rag_model = rm
    rag_tokenizer = BPE()
    logger.info("✅ RAG-модель загружена")
    return rag_model, rag_tokenizer

def load_sql_model():
    global sql_model, sql_tokenizer, sql_device, sql_dtype
    if sql_model is not None:
        return sql_model, sql_tokenizer
    logger.info("🔧 Загрузка SQL-модели...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_llm import DenseTransformer, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, N_LAYERS
    from testoBPE import BPE
    sql_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sql_dtype = torch.bfloat16 if (sql_device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    tok = BPE()
    vocab_size = len(tok)
    model = DenseTransformer(vocab_size=vocab_size, dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=0.0).to(sql_device)
    ckpt = torch.load(SQL_MODEL_PATH, map_location="cpu", weights_only=False)
    state = {k.replace('_orig_mod.', ''): v for k, v in (ckpt.get('model_state_dict', ckpt)).items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    sql_model, sql_tokenizer = model, tok
    logger.info("✅ SQL-модель готова")
    return model, tok

def load_dialog_model():
    global dialog_model, dialog_tokenizer, dialog_device, dialog_dtype
    if dialog_model is not None:
        return dialog_model, dialog_tokenizer
    logger.info("🔧 Загрузка диалоговой модели...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_llm import DenseTransformer, MODEL_DIM, N_HEADS, FFN_DIM, N_LAYERS
    from testoBPE import BPE
    dialog_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dialog_dtype = torch.bfloat16 if (dialog_device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    tok = BPE()
    vocab_size = len(tok)
    model = DenseTransformer(vocab_size=vocab_size, dim=MODEL_DIM, n_heads=N_HEADS, ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=0.0).to(dialog_device)
    ckpt = torch.load(DIALOG_MODEL_PATH, map_location="cpu", weights_only=False)
    state = {k.replace('_orig_mod.', ''): v for k, v in (ckpt.get('model_state_dict', ckpt)).items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    dialog_model, dialog_tokenizer = model, tok
    logger.info("✅ Диалоговая модель готова")
    return model, tok

def encode_query(text: str) -> np.ndarray:
    if rag_model is None:
        load_rag_model()
    ids = rag_tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    ids = [BOS_TOKEN_ID] + ids
    if len(ids) > MAX_QUERY_LEN:
        ids = ids[:MAX_QUERY_LEN]
    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long).to(rag_device)
        emb = rag_model(inp)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32).squeeze(0)

def get_known_books(user_id: int, user_map: dict) -> Set[int]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT idBook FROM userRating WHERE idUser=%s AND rating IS NOT NULL", (user_id,))
        known = {r["idBook"] for r in cursor.fetchall() if r["idBook"] in user_map}
        cursor.close()
        conn.close()
        return known
    except:
        return set()

def format_books(book_ids: list, scores: list) -> list:
    return [{"book_id": int(bid), "score": round(float(sc), 4)} for bid, sc in zip(book_ids, scores)]

def fallback_popular(k: int, item_map: dict) -> dict:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT idBook, COUNT(*) as cnt FROM userRating WHERE rating IS NOT NULL GROUP BY idBook ORDER BY cnt DESC LIMIT %s", (k,))
        top = [(r["idBook"], r["cnt"]) for r in cursor.fetchall() if r["idBook"] in item_map]
        cursor.close()
        conn.close()
        return {"user_id": 0, "note": "cold_start_fallback", "recommendations": format_books([t[0] for t in top], [float(t[1]) for t in top])}
    except:
        return {"error": "fallback_failed", "recommendations": []}

def _generate_embeddings_background(batch_size: int = 64):
    global book_embeddings, _book_ids_ordered, embeddings_generated
    if rag_model is None:
        load_rag_model()
    if not _book_ids_ordered:
        load_embeddings_and_metadata()
    logger.info(f"🔄 Начинаю фоновую генерацию эмбеддингов для {len(_book_ids_ordered)} книг...")
    all_embs = []
    for i in range(0, len(_book_ids_ordered), batch_size):
        batch_ids = _book_ids_ordered[i:i+batch_size]
        batch_embs = []
        for bid in batch_ids:
            meta = book_meta.get(bid, {})
            text = f"{meta.get('title', '')}. {meta.get('desc', '')}"
            try:
                ids = rag_tokenizer.encode(text)
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                ids = [BOS_TOKEN_ID] + ids
                if len(ids) > MAX_QUERY_LEN:
                    ids = ids[:MAX_QUERY_LEN]
                with torch.no_grad():
                    inp = torch.tensor([ids], dtype=torch.long).to(rag_device)
                    emb = rag_model(inp)
                    emb = F.normalize(emb, p=2, dim=1)
                batch_embs.append(emb.cpu().numpy().astype(np.float32).squeeze(0))
            except Exception as e:
                logger.warning(f"Ошибка генерации для книги {bid}: {e}")
                batch_embs.append(np.zeros(768, dtype=np.float32))
        all_embs.extend(batch_embs)
        if (i // batch_size) % 10 == 0:
            logger.info(f"🔄 Прогресс: {i}/{len(_book_ids_ordered)} книг")
    book_embeddings = np.array(all_embs)
    output_path = "book_embeddings_custom_768.npy"
    try:
        np.save(output_path, book_embeddings)
        logger.info(f"💾 Эмбеддинги сохранены в {output_path}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить эмбеддинги: {e}")
    embeddings_generated = True
    logger.info(f"✅ Генерация завершена: {book_embeddings.shape}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_hybrid_model()
    load_embeddings_and_metadata()
    try:
        load_rag_model()
    except Exception as e:
        logger.warning(f"RAG не загружен: {e}")
    logger.info("✅ Сервис полностью готов.")
    yield

app = FastAPI(title="BookRecs v6.5", version="6.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hybrid_model": hy_model is not None,
        "cf_model": cf_model is not None,
        "rag": rag_model is not None,
        "embeddings_loaded": book_embeddings is not None,
        "embeddings_generated": embeddings_generated,
        "books": len(book_meta)
    }

@app.post("/recommend")
def recommend(req: DirectRecommendRequest):
    hy_model, hy_item_map, hy_user_map, hy_item_features = load_hybrid_model()
    uid = req.user_id
    if uid not in hy_user_map:
        logger.warning(f"Cold-start для user_id={uid}")
        return fallback_popular(req.top_k, hy_item_map)
    try:
        user_idx = hy_user_map[uid]
        n_items = len(hy_item_map)
        scores = hy_model.predict(user_idx, np.arange(n_items), item_features=hy_item_features)
        for bid in get_known_books(uid, hy_user_map):
            if bid in hy_item_map:
                scores[hy_item_map[bid]] = -np.inf
        top_idx = np.argsort(-scores)[:req.top_k]
        top_books = [k for k, v in hy_item_map.items() if v in top_idx]
        top_scores = [float(scores[hy_item_map[bid]]) for bid in top_books]
        return {"user_id": uid, "recommendations": format_books(top_books, top_scores)}
    except Exception as e:
        logger.error(f"/recommend error: {e}")
        raise HTTPException(500, str(e))

# ---------- НОВЫЙ GET-ЭНДПОИНТ ДЛЯ СОВМЕСТИМОСТИ СО СТАРЫМ PHP ----------
@app.get("/predict")
def predict_legacy(idUser: int, lim: int = 10):
    """
    Старый GET-эндпоинт для совместимости с PHP кодом.
    Параметры: idUser, lim (аналог top_k)
    """
    # Создаём такой же запрос, как в /recommend
    req = DirectRecommendRequest(user_id=idUser, top_k=lim)
    return recommend(req)
# --------------------------------------------------------------------

@app.post("/cf")
def cf_recommend(req: CFRecommendRequest):
    model, item_map, user_map = load_cf_model()
    uid = req.user_id
    if uid not in user_map:
        return fallback_popular(req.top_k, item_map)
    try:
        user_idx = user_map[uid]
        n_items = len(item_map)
        scores = model.predict(user_idx, np.arange(n_items))
        for bid in get_known_books(uid, user_map):
            if bid in item_map:
                scores[item_map[bid]] = -np.inf
        top_idx = np.argsort(-scores)[:req.top_k]
        top_books = [k for k, v in item_map.items() if v in top_idx]
        top_scores = [float(scores[item_map[bid]]) for bid in top_books]
        return {"user_id": uid, "model": "pure_cf", "recommendations": format_books(top_books, top_scores)}
    except Exception as e:
        logger.error(f"/cf error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search")
def semantic_search(req: SearchRequest):
    # Игнорируем req.filter_images – вся фильтрация на клиенте
    if book_embeddings is None or len(_book_ids_ordered) == 0:
        scores = np.zeros(len(_book_ids_ordered))
        ql = req.text.lower()
        for i, bid in enumerate(_book_ids_ordered):
            meta = book_meta.get(bid, {})
            if ql in meta.get("title", "").lower():
                scores[i] += 2.0
            if ql in meta.get("desc", "").lower():
                scores[i] += 1.0
        idx = np.argsort(-scores)[:req.top_k]
        books = [_book_ids_ordered[i] for i in idx if i < len(_book_ids_ordered)]
        return {"query": req.text, "method": "fallback_substring", "results": format_books(books, [float(scores[i]) for i in idx if i < len(_book_ids_ordered)])}
    try:
        if rag_model and rag_tokenizer:
            qemb = encode_query(req.text)
            scores = book_embeddings @ qemb
            method = "rag_cosine"
        else:
            scores = np.zeros(len(_book_ids_ordered))
            ql = req.text.lower()
            for i, bid in enumerate(_book_ids_ordered):
                meta = book_meta.get(bid, {})
                if ql in meta.get("title", "").lower():
                    scores[i] += 2.0
                if ql in meta.get("desc", "").lower():
                    scores[i] += 1.0
            method = "fallback"
        idx = np.argsort(-scores)[:req.top_k]
        books = [_book_ids_ordered[i] for i in idx if i < len(_book_ids_ordered)]
        return {"query": req.text, "method": method, "results": format_books(books, [float(scores[i]) for i in idx if i < len(_book_ids_ordered)])}
    except Exception as e:
        logger.error(f"/search error: {e}")
        raise HTTPException(500, str(e))

@app.post("/sql")
def sql_query(req: SQLRequest):
    model, tok = load_sql_model()
    prompt = f"<[BOS]><[SYS]>{SQL_PROMPT}<[EOS]><[USR]>{req.question}<[EOS]><[BOT]>"
    input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    output_ids_1d = generate_response(model, tok, input_ids, MAX_SQL, TEMP_SQL, TOP_K_SQL, sql_device, sql_dtype)
    raw = decode_response(tok, output_ids_1d, len(tok.encode(prompt)))
    for tag in ['<[EOS]>', '<[USR]>', '<[BOT]>', '<[SYS]>', '<[BOS]>']:
        raw = raw.replace(tag, '')
    sql = extract_sql_from_answer(raw.strip())
    result = None
    error = None
    if req.execute and sql:
        conn = mysql.connector.connect(**DB_CONFIG)
        result, error = execute_sql_safe(conn, sql, limit=req.top_k)
        conn.close()
    return {"question": req.question, "generated_sql": sql or "(not extracted)", "raw_answer": raw.strip(), "executed": req.execute and bool(sql), "result": result, "error": error}

@app.post("/dialog")
def dialog_chat(req: DialogRequest):
    model, tok = load_dialog_model()
    ctx = f"<[BOS]><[SYS]>{DIALOG_PROMPT}<[EOS]>"
    if req.history:
        for msg in req.history[-5:]:
            role = "USR" if msg.get("role") == "user" else "BOT"
            ctx += f"<[{role}]>{msg.get('content', '')}<[EOS]>"
    ctx += f"<[USR]>{req.message}<[EOS]><[BOT]>"
    input_ids = torch.tensor([tok.encode(ctx)], dtype=torch.long)
    output_ids_1d = generate_response(model, tok, input_ids, MAX_DIALOG, TEMP_DIALOG, TOP_K_DIALOG, dialog_device, dialog_dtype)
    raw = decode_response(tok, output_ids_1d, len(tok.encode(ctx)))
    for tag in ['<[EOS]>', '<[USR]>', '<[BOT]>', '<[SYS]>', '<[BOS]>']:
        raw = raw.replace(tag, '')
    return {"message": req.message, "response": raw.strip(), "user_id": req.user_id}

@app.post("/query")
def unified_query(req: QueryRequest):
    intent = route_intent(req.query)
    try:
        if intent == "recommend":
            uid = req.user_id if req.user_id is not None else 0
            resp = recommend(DirectRecommendRequest(user_id=uid, top_k=req.top_k))
            if "recommendations" in resp:
                return resp
            else:
                return {"recommendations": resp.get("recommendations", [])}
        elif intent == "search":
            resp = semantic_search(SearchRequest(text=req.query, top_k=req.top_k))
            return resp
        elif intent == "sql":
            resp = sql_query(SQLRequest(question=req.query, execute=False, top_k=10))
            return resp
        elif intent == "dialog":
            resp = dialog_chat(DialogRequest(message=req.query, user_id=req.user_id))
            return resp
        else:
            return {"response": "Не удалось определить тип запроса.", "intent": intent}
    except Exception as e:
        logger.error(f"/query error: {e}")
        raise HTTPException(500, str(e))

@app.post("/generate-embeddings")
async def trigger_embeddings_generation(req: GenerateEmbeddingsRequest = None, background_tasks: BackgroundTasks = None):
    if rag_model is None:
        raise HTTPException(503, "RAG model not loaded, cannot generate embeddings")
    batch_size = req.batch_size if req else 64
    force = req.force if req else False
    if book_embeddings is not None and not force:
        return {"status": "already_loaded", "message": "Embeddings already in memory. Use force=true to regenerate."}
    if background_tasks:
        background_tasks.add_task(_generate_embeddings_background, batch_size)
        return {"status": "started", "message": f"Background generation started (batch_size={batch_size})"}
    else:
        _generate_embeddings_background(batch_size)
        return {"status": "completed", "message": f"Generated {len(_book_ids_ordered)} embeddings"}

@app.post("/search-text")
def text_search(req: TextSearchRequest):
    """Простой текстовый поиск по названиям книг (возвращает только названия)."""
    if not book_meta or not _book_ids_ordered:
        return {"results": []}
    
    q_lower = req.text.lower().strip()
    if not q_lower:
        return {"results": []}
    
    results = []
    # Идём по всем книгам, ищем вхождения в title или description
    for book_id in _book_ids_ordered:
        meta = book_meta.get(book_id)
        if not meta:
            continue
        title = meta.get("title", "")
        desc = meta.get("desc", "")
        if q_lower in title.lower() or (desc and q_lower in desc.lower()):
            results.append(title)
            if len(results) >= req.top_k:
                break
    
    return {"results": results}

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"💥 {type(exc).__name__}: {exc}")
    return JSONResponse(content={"error": "internal_error", "detail": str(exc)[:200]}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")