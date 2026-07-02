#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BookRecs API v6.8 — с генерацией эмбеддингов через /generate-embeddings."""
import os, sys, re, logging, numpy as np, joblib, mysql.connector, torch, torch.nn.functional as F
from contextlib import asynccontextmanager
from typing import List, Optional, Set, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
from model import RAGEncoder, BOS_TOKEN_ID
from testoBPE import BPE

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

MODEL_PATHS = {
    "hy": "hy_model.pkl",
    "cf": "cf_model.pkl",
    "rag": "logs6_from_3_using_2_3/model.pth",
    "sql": "/mnt/news/buffer/OLD/logs8/model_final.pth",
    "dlg": "/mnt/news/buffer/OLD/logs5/model_final_tuned.pth"
}

EMB_PATHS = [
    os.path.join(os.path.dirname(__file__), "book_embeddings_custom_768.npy"),
    "book_embeddings_custom_768.npy",
    os.path.join(os.path.dirname(__file__), "book_embeddings_pretrained.npy"),
    "book_embeddings_pretrained.npy",
]

BOS = BOS_TOKEN_ID
MAX_Q = 1024
T_SQL, K_SQL, M_SQL = 0.3, 20, 512
T_DLG, K_DLG, M_DLG = 0.8, 50, 256

SQL_SYS = "Ты — ассистент по SQL для базы книг. Отвечай ТОЛЬКО корректным SQL после 'SQL:'. Используй таблицу books и правильные JOIN."
DLG_SYS = "Твое имя Стас. Ты - опытный писатель сценариев. Твоя любимая тема - книги."

KEYWORDS = {
    "rec": ["посоветуй", "порекомендуй", "мне нравится", "похож", "подбери", "что почитать", "дай список"],
    "srch": ["найди", "поиск", "о чём", "про что", "жанр", "тема", "смысл"],
    "sql": ["сколько", "кто написал", "когда вышла", "покажи все", "статистика", "какие книги", "автор", "год"],
    "dlg": ["привет", "как дела", "объясни", "почему", "что такое", "помоги", "спасибо", "расскажи"]
}

class QReq(BaseModel):
    query: str
    user_id: Optional[int] = None
    top_k: int = 6
    filter_images: bool = False

class RecReq(BaseModel):
    user_id: int
    top_k: int = 6
    filter_images: bool = False

class SReq(BaseModel):
    text: str
    top_k: int = 6
    filter_images: bool = False

class SQLReq(BaseModel):
    question: str
    execute: bool = False
    top_k: int = 10

class DReq(BaseModel):
    message: str
    user_id: Optional[int] = None
    history: Optional[List[dict]] = None

class TxtSReq(BaseModel):
    text: str
    top_k: int = 10
    filter_images: bool = False

hy_m = hy_im = hy_um = hy_if = None
cf_m = cf_im = cf_um = None
emb = None
meta: Dict[int, Dict] = {}
book_ids: List[int] = []
book_ids_with_covers: List[int] = []
rag_m = rag_tok = rag_dev = rag_dt = None
sql_m = sql_tok = sql_dev = sql_dt = None
dlg_m = dlg_tok = dlg_dev = dlg_dt = None

def db():
    return mysql.connector.connect(**DB_CONFIG)

def load_meta():
    global meta, book_ids, book_ids_with_covers
    if meta:
        return True
    logger.info("Loading metadata...")
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT b.id, b.name, LEFT(b.description, 500) as description, i.src as cover_src
        FROM books b
        LEFT JOIN images i ON b.idCover = i.id
        ORDER BY b.id
    """)
    rows = cur.fetchall()
    for r in rows:
        meta[r["id"]] = {
            "title": r["name"],
            "desc": r["description"] or "",
            "has_cover": bool(r["cover_src"] and str(r["cover_src"]).strip())
        }
    book_ids = [r["id"] for r in rows]
    book_ids_with_covers = [bid for bid in book_ids if meta[bid]["has_cover"]]
    cur.close()
    conn.close()
    logger.info(f"Loaded {len(meta)} books, {len(book_ids_with_covers)} with covers")
    return True

def fmt_books(ids, sc):
    return [{"book_id": int(b), "score": round(float(s), 4)} for b, s in zip(ids, sc)]

def known_books(uid: int, umap: dict) -> Set[int]:
    try:
        conn = db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT idBook FROM userRating WHERE idUser=%s AND rating IS NOT NULL", (uid,))
        res = {r["idBook"] for r in cur.fetchall() if r["idBook"] in umap}
        cur.close()
        conn.close()
        return res
    except:
        return set()

def popular_fallback(k: int, imap: dict, flag: bool) -> dict:
    try:
        conn = db()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT idBook, COUNT(*) as cnt FROM userRating WHERE rating IS NOT NULL GROUP BY idBook ORDER BY cnt DESC LIMIT %s",
            (k * 10,)
        )
        top = [(r["idBook"], r["cnt"]) for r in cur.fetchall() if r["idBook"] in imap]
        cur.close()
        conn.close()
        if flag:
            top = [(b, c) for b, c in top if meta.get(b, {}).get("has_cover")]
        return {
            "user_id": 0,
            "note": "cold_start",
            "recommendations": fmt_books([t[0] for t in top[:k]], [float(t[1]) for t in top[:k]])
        }
    except:
        return {"error": "fallback_fail", "recommendations": []}

def gen_text(model, tok, prompt, max_new, temp, topk, dev, dt):
    model.eval()
    inp = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(dev)
    eos = tok.encode("<[EOS]>")
    eos_id = eos[0] if eos else None
    gen = inp.clone()
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt if dev.type == 'cuda' else torch.float32):
        for _ in range(max_new):
            seq = gen[:, -1024:] if gen.shape[1] > 1024 else gen
            logits, _ = model(seq)
            nxt = logits[:, -1, :] / max(temp, 1e-6)
            if topk > 0:
                tk_val, tk_idx = torch.topk(nxt, topk, dim=-1)
                mask = torch.ones_like(nxt, dtype=torch.bool)
                mask.scatter_(1, tk_idx, False)
                nxt[mask] = -float("Inf")
            probs = torch.softmax(nxt, dim=-1)
            nxt_id = torch.multinomial(probs, 1)
            gen = torch.cat((gen, nxt_id), dim=1)
            if eos_id and nxt_id.item() == eos_id:
                break
    out = gen[0][len(tok.encode(prompt)):].tolist()
    try:
        return tok.decode(out)
    except:
        return str(out)

def extract_sql(ans: str) -> str:
    ans = ans.strip()
    for ln in ans.split('\n'):
        if 'SQL:' in ln.upper():
            return clean_sql(ln.split('SQL:', 1)[1].strip())
    if ans.upper().startswith('SELECT'):
        return clean_sql(ans)
    m = re.search(
        r'(SELECT\s+[\w\s\.\,\*\(\)\'\"%]+?\s+FROM\s+[\w\s\.\,\'\"]+?(?:\s+WHERE\s+.*?|\s+GROUP\s+BY\s+.*?|\s+ORDER\s+BY\s+.*?|\s+LIMIT\s+.*?|;|$))',
        ans, re.I | re.S
    )
    return clean_sql(m.group(1)) if m else ""

def clean_sql(s: str) -> str:
    for t in ['<[EOS]>', '<[USR]>', '<[BOT]>', '<[SYS]>', '<[BOS]>']:
        s = s.replace(t, '')
    s = ' '.join(s.split()).rstrip(';').strip()
    return s if s.upper().startswith('SELECT') else ""

def exec_sql(conn, sql: str, limit: int = 100):
    if not sql or not sql.upper().startswith('SELECT'):
        return None, "Only SELECT"
    if 'LIMIT' not in sql.upper():
        sql = sql.rstrip(';') + f" LIMIT {limit}"
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        res = cur.fetchall()
        for row in res:
            for k, v in row.items():
                if isinstance(v, (bytes, bytearray)):
                    row[k] = v.decode('utf-8', errors='ignore')
                elif isinstance(v, str) and len(v) > 200:
                    row[k] = v[:200] + "..."
        return res, None
    except Exception as e:
        return None, str(e)[:200]

def route_intent(q: str) -> str:
    ql = q.lower()
    for intent, kws in KEYWORDS.items():
        if any(k in ql for k in kws):
            return intent
    return "rec"

def load_hy():
    global hy_m, hy_im, hy_um, hy_if
    if hy_m:
        return hy_m, hy_im, hy_um, hy_if
    logger.info("Loading hybrid model...")
    art = joblib.load(MODEL_PATHS["hy"])
    hy_m, hy_im, hy_um, hy_if = art["model"], art["item_map"], art["user_map"], art["item_features"]
    logger.info(f"Hybrid loaded: {len(hy_um)} users, {len(hy_im)} books")
    return hy_m, hy_im, hy_um, hy_if

def load_cf():
    global cf_m, cf_im, cf_um
    if cf_m:
        return cf_m, cf_im, cf_um
    logger.info("Loading CF model...")
    art = joblib.load(MODEL_PATHS["cf"])
    cf_m, cf_im, cf_um = art["model"], art["item_map"], art["user_map"]
    logger.info(f"CF loaded: {len(cf_um)} users, {len(cf_im)} books")
    return cf_m, cf_im, cf_um

def load_emb():
    global emb
    if emb is not None:
        return True

    load_meta()

    for p in EMB_PATHS:
        if os.path.exists(p):
            try:
                logger.info(f"Loading embeddings from {p}")
                emb_candidate = np.load(p)

                if emb_candidate.shape[0] != len(book_ids):
                    logger.warning(f"Size mismatch: {emb_candidate.shape[0]} vs {len(book_ids)} books, skipping")
                    continue

                norms = np.linalg.norm(emb_candidate, axis=1, keepdims=True)
                norms[norms == 0] = 1
                emb = emb_candidate / norms
                logger.info(f"Embeddings loaded: {emb.shape}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

    logger.warning("No valid embeddings found")
    emb = None
    return False

def load_rag():
    global rag_m, rag_tok, rag_dev, rag_dt
    if rag_m:
        return rag_m, rag_tok

    logger.info("Loading RAG model...")
    sys.path.insert(0, os.path.dirname(__file__))
    from model import RAGEncoder, BOS_TOKEN_ID
    from testoBPE import BPE

    rag_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rag_dt = torch.float16
    rm = RAGEncoder().to(rag_dev, dtype=rag_dt)
    ckpt = torch.load(MODEL_PATHS["rag"], map_location="cpu", weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    state = {
        k.replace("_orig_mod.", ""): (
            v.half() if v.dtype == torch.float32 else v
        ) if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
    rm.load_state_dict(state, strict=False)
    rm.eval()

    rag_tok = BPE()
    if len(rag_tok) == 0:
        logger.error("Tokenizer vocabulary is empty!")
        raise RuntimeError("BPE tokenizer failed to load vocabulary")

    rag_m = rm
    logger.info(f"RAG model loaded, tokenizer vocab size: {len(rag_tok)}, pooling: {rm.pooling}")
    return rag_m, rag_tok

def load_sql():
    global sql_m, sql_tok, sql_dev, sql_dt
    if sql_m:
        return sql_m, sql_tok

    logger.info("Loading SQL model...")
    sys.path.insert(0, os.path.dirname(__file__))
    from model_llm import DenseTransformer, MODEL_DIM, N_HEADS, N_KV_HEADS, FFN_DIM, N_LAYERS
    from testoBPE import BPE

    sql_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sql_dt = torch.bfloat16 if (sql_dev.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    tok = BPE()
    model = DenseTransformer(
        vocab_size=len(tok), dim=MODEL_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
        ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=0.0
    ).to(sql_dev)
    ckpt = torch.load(MODEL_PATHS["sql"], map_location="cpu", weights_only=False)
    state = {
        k.replace('_orig_mod.', ''): v
        for k, v in (ckpt.get('model_state_dict', ckpt)).items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    sql_m, sql_tok = model, tok
    logger.info("SQL model loaded")
    return model, tok

def load_dlg():
    global dlg_m, dlg_tok, dlg_dev, dlg_dt
    if dlg_m:
        return dlg_m, dlg_tok

    logger.info("Loading dialog model...")
    sys.path.insert(0, os.path.dirname(__file__))
    from model_llm import DenseTransformer, MODEL_DIM, N_HEADS, FFN_DIM, N_LAYERS
    from testoBPE import BPE

    dlg_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dlg_dt = torch.bfloat16 if (dlg_dev.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    tok = BPE()
    model = DenseTransformer(
        vocab_size=len(tok), dim=MODEL_DIM, n_heads=N_HEADS,
        ffn_dim=FFN_DIM, n_layers=N_LAYERS, dropout=0.0
    ).to(dlg_dev)
    ckpt = torch.load(MODEL_PATHS["dlg"], map_location="cpu", weights_only=False)
    state = {
        k.replace('_orig_mod.', ''): v
        for k, v in (ckpt.get('model_state_dict', ckpt)).items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    dlg_m, dlg_tok = model, tok
    logger.info("Dialog model loaded")
    return model, tok

def encode_q(text: str) -> np.ndarray:
    if rag_m is None:
        load_rag()

    ids = rag_tok.encode(text)
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
    ids = [BOS] + ids[:MAX_Q]

    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long).to(rag_dev)
        e = rag_m(inp)
        e = F.normalize(e, p=2, dim=1)

    return e.cpu().numpy().astype(np.float32).squeeze(0)

def generate_all_embeddings():
    global emb

    load_meta()
    load_rag()

    logger.info(f"Generating embeddings for {len(book_ids)} books...")

    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, description, authors FROM books ORDER BY id")
    books = cur.fetchall()
    cur.close()
    conn.close()

    all_embeddings = []
    batch_size = 32

    for i in range(0, len(books), batch_size):
        batch = books[i:i + batch_size]
        batch_embs = []

        for book in batch:
            desc = book['description'] or ""
            name = book['name'] or ""
            authors = book['authors'] or ""
            
            if len(desc.strip()) >= 10:
                text = f"{name}. {desc}"
            else:
                if authors and len(authors.strip()) > 2:
                    text = f"{name}. Автор: {authors}"
                else:
                    text = name if name else f"Книга {book['id']}"
            
            tokens = rag_tok.encode(text)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            tokens = [BOS] + tokens[:MAX_Q]

            with torch.no_grad():
                inp = torch.tensor([tokens], dtype=torch.long).to(rag_dev)
                e = rag_m(inp)
                e = F.normalize(e, p=2, dim=1)
            batch_embs.append(e.cpu().numpy().astype(np.float32))

        all_embeddings.extend(batch_embs)
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {min(i + batch_size, len(books))}/{len(books)} books")

    emb = np.vstack(all_embeddings)
    
    unique_embeddings = np.unique(emb, axis=0)
    
    logger.info(f"📊 Статистика генерации:")
    logger.info(f"   - Всего книг: {len(books)}")
    logger.info(f"   - Уникальных эмбеддингов: {len(unique_embeddings)} из {len(emb)}")
    
    if len(unique_embeddings) < len(emb):
        logger.warning(f"⚠️ {len(emb) - len(unique_embeddings)} дубликатов!")

    output_path = os.path.join(os.path.dirname(__file__), "book_embeddings_custom_768.npy")
    np.save(output_path, emb)
    logger.info(f"Embeddings saved to {output_path}, shape: {emb.shape}")

    return emb

def relearn_models():
    global hy_m, hy_im, hy_um, hy_if, cf_m, cf_im, cf_um
    
    logger.info("🔄 Starting model re-learning...")
    
    try:
        conn = db()
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT ur.idUser AS user_id, ur.idBook AS book_id,
               CASE WHEN ur.rating >= 7 THEN 1.0 
                    WHEN ur.rating >= 5 THEN 0.5 
                    ELSE 0.1 END AS weight,
               b.name, b.description
        FROM userRating ur
        JOIN books b ON ur.idBook = b.id
        WHERE ur.rating IS NOT NULL
        GROUP BY ur.idUser, ur.idBook, b.name, b.description
        """
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())
        cursor.close()
        conn.close()

        if df.empty:
            logger.warning("⚠️ No data found for re-training.")
            return {"status": "error", "message": "No data in database"}

        logger.info(f"📊 Loaded {len(df)} ratings from {df['user_id'].nunique()} users and {df['book_id'].nunique()} books")

        load_emb()
        load_rag()
        
        dataset = Dataset()
        users = df['user_id'].unique()
        items = df['book_id'].unique()
        dataset.fit(users=users, items=items)
        
        interactions, weights = dataset.build_interactions(
            ((row['user_id'], row['book_id'], row['weight']) for _, row in df.iterrows())
        )
        
        user_map, _, item_map, _ = dataset.mapping()
        logger.info(f"📦 Dataset built: {len(user_map)} users, {len(item_map)} items")

        item_features_sparse = None
        if emb is not None and len(item_map) > 0:
            n_items = len(item_map)
            dim = emb.shape[1]
            aligned = np.zeros((n_items, dim), dtype=np.float32)
            
            for bid, idx in item_map.items():
                if bid in book_ids:
                    book_idx = book_ids.index(bid)
                    if book_idx < emb.shape[0]:
                        aligned[idx] = emb[book_idx]
            
            item_features_sparse = csr_matrix(aligned)
            logger.info(f"🧠 Item features shape: {item_features_sparse.shape}")
        
        logger.info("🚀 Training hybrid model...")
        hy_model = LightFM(
            loss='warp', 
            no_components=64, 
            learning_rate=0.05, 
            item_alpha=0.0, 
            user_alpha=0.0, 
            max_sampled=10, 
            random_state=42
        )
        hy_model.fit(
            interactions, 
            sample_weight=weights, 
            item_features=item_features_sparse, 
            epochs=20, 
            num_threads=min(os.cpu_count() or 4, 8)
        )
        
        hy_artifacts = {
            'model': hy_model,
            'item_map': item_map,
            'user_map': user_map,
            'item_features': item_features_sparse
        }
        joblib.dump(hy_artifacts, MODEL_PATHS["hy"])
        logger.info(f"✅ Hybrid model saved to {MODEL_PATHS['hy']}")
        
        logger.info("🚀 Training CF model...")
        cf_model = LightFM(
            loss='warp', 
            no_components=64, 
            learning_rate=0.05, 
            item_alpha=0.0, 
            user_alpha=0.0, 
            max_sampled=10, 
            random_state=42
        )
        cf_model.fit(
            interactions, 
            sample_weight=weights, 
            epochs=20, 
            num_threads=min(os.cpu_count() or 4, 8)
        )
        
        cf_artifacts = {
            'model': cf_model,
            'item_map': item_map,
            'user_map': user_map
        }
        joblib.dump(cf_artifacts, MODEL_PATHS["cf"])
        logger.info(f"✅ CF model saved to {MODEL_PATHS['cf']}")
        
        hy_m, hy_im, hy_um, hy_if = hy_model, item_map, user_map, item_features_sparse
        cf_m, cf_im, cf_um = cf_model, item_map, user_map
        
        logger.info(f"✅ Models updated in memory: hybrid ({len(hy_um)} users, {len(hy_im)} items), CF ({len(cf_um)} users, {len(cf_im)} items)")
        
        return {
            "status": "success",
            "message": "Models re-trained and saved successfully",
            "hybrid": {"users": len(hy_um), "items": len(hy_im)},
            "cf": {"users": len(cf_um), "items": len(cf_im)}
        }
        
    except Exception as e:
        logger.error(f"❌ Error during re-learning: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_hy()
    load_meta()
    
    emb_exists = False
    for p in EMB_PATHS:
        if os.path.exists(p):
            emb_exists = True
            break
    
    if not emb_exists:
        logger.warning("⚠️ No embeddings found! They will be generated on first search request.")
    
    load_emb()
    try:
        load_rag()
    except Exception as e:
        logger.warning(f"RAG failed to load: {e}")
    logger.info("Service ready")
    yield

app = FastAPI(title="BookRecs v6.8", version="6.8.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hy": hy_m is not None,
        "cf": cf_m is not None,
        "rag": rag_m is not None,
        "emb": emb is not None,
        "books": len(meta),
        "with_covers": len(book_ids_with_covers),
        "embeddings_file_exists": any(os.path.exists(p) for p in EMB_PATHS)
    }

@app.get("/relearn")
async def relearn_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(relearn_models)
    return {
        "status": "started",
        "message": "Model re-learning started in background. Check logs for progress."
    }

@app.post("/generate-embeddings")
async def generate_embeddings_endpoint():
    try:
        logger.info("Starting embeddings generation...")
        new_emb = generate_all_embeddings()
        return {
            "status": "ok",
            "message": "Embeddings generated successfully",
            "shape": list(new_emb.shape),
            "path": os.path.join(os.path.dirname(__file__), "book_embeddings_custom_768.npy")
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/recommend")
def rec(req: RecReq):
    hy_m, hy_im, hy_um, hy_if = load_hy()
    uid = req.user_id
    if uid not in hy_um:
        return popular_fallback(req.top_k, hy_im, req.filter_images)
    try:
        uidx = hy_um[uid]
        candidate_ids = book_ids_with_covers if req.filter_images else book_ids
        candidate_idx = [hy_im[bid] for bid in candidate_ids if bid in hy_im]
        if not candidate_idx:
            return {"user_id": uid, "recommendations": []}

        scores = hy_m.predict(uidx, np.array(candidate_idx), item_features=hy_if)
        for bid in known_books(uid, hy_um):
            if bid in hy_im and hy_im[bid] in candidate_idx:
                scores[candidate_idx.index(hy_im[bid])] = -np.inf

        top_idx = np.argsort(-scores)[:req.top_k]
        top_books = [candidate_ids[i] for i in top_idx]
        top_scores = [float(scores[i]) for i in top_idx]
        return {"user_id": uid, "recommendations": fmt_books(top_books, top_scores)}
    except Exception as e:
        logger.error(f"/recommend error: {e}")
        raise HTTPException(500, str(e))

@app.get("/predict")
def predict_legacy(idUser: int, lim: int = 10):
    return rec(RecReq(user_id=idUser, top_k=lim))

@app.post("/cf")
def cf_rec(req: RecReq):
    cf_m, cf_im, cf_um = load_cf()
    uid = req.user_id
    if uid not in cf_um:
        return popular_fallback(req.top_k, cf_im, req.filter_images)
    try:
        uidx = cf_um[uid]
        candidate_ids = book_ids_with_covers if req.filter_images else book_ids
        candidate_idx = [cf_im[bid] for bid in candidate_ids if bid in cf_im]
        if not candidate_idx:
            return {"user_id": uid, "recommendations": []}

        scores = cf_m.predict(uidx, np.array(candidate_idx))
        for bid in known_books(uid, cf_um):
            if bid in cf_im and cf_im[bid] in candidate_idx:
                scores[candidate_idx.index(cf_im[bid])] = -np.inf

        top_idx = np.argsort(-scores)[:req.top_k]
        top_books = [candidate_ids[i] for i in top_idx]
        top_scores = [float(scores[i]) for i in top_idx]
        return {"user_id": uid, "recommendations": fmt_books(top_books, top_scores)}
    except Exception as e:
        logger.error(f"/cf error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search")
def sem_search(req: SReq):
    load_meta()
    search_pool = book_ids_with_covers if (req.filter_images and book_ids_with_covers) else book_ids
    if not search_pool:
        return {"query": req.text, "method": "no_books", "results": []}

    if emb is None:
        logger.info("No embeddings found, generating...")
        try:
            generate_all_embeddings()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return fallback_search(req, search_pool)

    if emb is not None and rag_m is not None:
        try:
            qemb = encode_q(req.text)
            pool_indices = [book_ids.index(bid) for bid in search_pool if bid in book_ids]
            pool_emb = emb[pool_indices]
            scores = pool_emb @ qemb

            idx = np.argsort(-scores)[:req.top_k]
            top_books = [search_pool[i] for i in idx]
            top_scores = [float(scores[i]) for i in idx]
            return {"query": req.text, "method": "rag_cosine", "results": fmt_books(top_books, top_scores)}
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to substring")

    return fallback_search(req, search_pool)

def fallback_search(req: SReq, search_pool: List[int]) -> dict:
    scores = []
    ql = req.text.lower().strip()
    for bid in search_pool:
        m = meta.get(bid, {})
        sc = 0.0
        if ql in m.get("title", "").lower():
            sc += 2.0
        if ql in m.get("desc", "").lower():
            sc += 1.0
        scores.append(sc)

    idx = np.argsort(-np.array(scores))[:req.top_k]
    top_books = [search_pool[i] for i in idx]
    top_scores = [float(scores[i]) for i in idx]
    return {"query": req.text, "method": "fallback", "results": fmt_books(top_books, top_scores)}

@app.post("/search-text")
def txt_search(req: TxtSReq):
    load_meta()
    if not meta:
        return {"results": []}
    ql = req.text.lower().strip()
    if not ql:
        return {"results": []}

    search_pool = book_ids_with_covers if (req.filter_images and book_ids_with_covers) else book_ids
    matching_ids = []
    for bid in search_pool:
        m = meta.get(bid)
        if not m:
            continue
        title, desc = m.get("title", ""), m.get("desc", "")
        if ql in title.lower() or (desc and ql in desc.lower()):
            matching_ids.append(bid)
            if len(matching_ids) >= req.top_k:
                break

    return {"results": [{"book_id": bid, "score": 1.0} for bid in matching_ids]}

@app.post("/sql")
def sql_q(req: SQLReq):
    sql_m, sql_tok = load_sql()
    prompt = f"<[BOS]><[SYS]>{SQL_SYS}<[EOS]><[USR]>{req.question}<[EOS]><[BOT]>"
    sql, raw = "", ""

    for attempt in range(10):
        try:
            raw = gen_text(sql_m, sql_tok, prompt, M_SQL, T_SQL, K_SQL, sql_dev, sql_dt)
            sql = extract_sql(raw.strip())
            if sql:
                break
        except:
            pass

    result, error, is_books = None, None, False

    if req.execute and sql:
        conn = db()
        result, error = exec_sql(conn, sql, limit=req.top_k)
        conn.close()

        if result and len(result) > 0:
            keys = [k.lower() for k in result[0].keys()]
            if any(k in ('idbook', 'book_id', 'bookid', 'id') for k in keys):
                is_books = True
                book_results = []
                for row in result:
                    bid = row.get('idBook') or row.get('book_id') or row.get('bookid') or row.get('id')
                    if bid:
                        book_results.append({"book_id": int(bid), "score": 1.0})
                result = book_results

    return {
        "question": req.question,
        "generated_sql": sql or "(not extracted)",
        "raw_answer": raw.strip(),
        "executed": req.execute and bool(sql),
        "results": result,
        "error": error,
        "is_books": is_books
    }

@app.post("/dialog")
def dlg_chat(req: DReq):
    dlg_m, dlg_tok = load_dlg()
    ctx = f"<[BOS]><[SYS]>{DLG_SYS}<[EOS]>"
    if req.history:
        for msg in req.history[-5:]:
            role = "USR" if msg.get("role") == "user" else "BOT"
            ctx += f"<[{role}]>{msg.get('content', '')}<[EOS]>"
    ctx += f"<[USR]>{req.message}<[EOS]><[BOT]>"
    out = gen_text(dlg_m, dlg_tok, ctx, M_DLG, T_DLG, K_DLG, dlg_dev, dlg_dt)
    for t in ['<[EOS]>', '<[USR]>', '<[BOT]>', '<[SYS]>', '<[BOS]>']:
        out = out.replace(t, '')
    return {"message": req.message, "response": out.strip(), "user_id": req.user_id}

@app.post("/query")
def unified(req: QReq):
    intent = route_intent(req.query)
    try:
        if intent == "rec":
            uid = req.user_id if req.user_id is not None else 0
            return rec(RecReq(user_id=uid, top_k=req.top_k, filter_images=req.filter_images))
        elif intent == "srch":
            return sem_search(SReq(text=req.query, top_k=req.top_k, filter_images=req.filter_images))
        elif intent == "sql":
            return sql_q(SQLReq(question=req.query, execute=False, top_k=10))
        elif intent == "dlg":
            return dlg_chat(DReq(message=req.query, user_id=req.user_id))
        else:
            return {"response": "Could not determine query type.", "intent": intent}
    except Exception as e:
        logger.error(f"/query error: {e}")
        raise HTTPException(500, str(e))

@app.exception_handler(Exception)
async def glob_err(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(content={"error": "internal", "detail": str(exc)[:200]}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
