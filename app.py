import os
import time
import json
from typing import List, Dict

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np

# ---------------------- App / Keys ----------------------
st.set_page_config(page_title="Store Assistant — Furniture", page_icon="🛋️", layout="wide")
load_dotenv()

st.title("🛋️ Store Assistant — Furniture (Gemini + RAG)")

# Session-state init
if "history" not in st.session_state:
    st.session_state["history"] = []

DEFAULT_MODEL = "gemini-1.5-flash"
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.subheader("🔑 API & Settings")
    api_key = st.text_input(
        "GOOGLE_API_KEY",
        value=api_key,
        type="password",
        help="Paste your Gemini API key. Use .env locally, Secrets on Streamlit Cloud.",
    )
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, 64)
    st.caption("Tip: Flash = fast & cheap. Pro = better reasoning.")
    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state["history"].clear()
        st.rerun()

if not api_key:
    st.error("Missing API key. Add it in the sidebar (GOOGLE_API_KEY).")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------- Tiny Vector Store (NumPy) ----------------------
DB_DIR = "db"
EMB_FILE = os.path.join(DB_DIR, "embeddings.npy")     # float32 [N, D]
META_FILE = os.path.join(DB_DIR, "meta.csv")          # product metadata
DIM_FILE = os.path.join(DB_DIR, "dim.json")           # {"dim": 768}

EMBED_MODEL = "text-embedding-004"

os.makedirs(DB_DIR, exist_ok=True)

def embed_text(text: str) -> np.ndarray:
    """Return a float32 embedding vector for a string."""
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    v = np.array(resp["embedding"], dtype=np.float32)
    return v

def load_store() -> tuple[np.ndarray, pd.DataFrame, int]:
    """Load embeddings + metadata. Returns (embs, meta, dim)."""
    if os.path.exists(EMB_FILE) and os.path.exists(META_FILE) and os.path.exists(DIM_FILE):
        embs = np.load(EMB_FILE)
        meta = pd.read_csv(META_FILE)
        dim = json.load(open(DIM_FILE, "r")).get("dim", embs.shape[1])
        return embs, meta, dim
    # empty store
    return np.zeros((0, 0), dtype=np.float32), pd.DataFrame(), 0

def save_store(embs: np.ndarray, meta: pd.DataFrame):
    """Persist embeddings + metadata."""
    os.makedirs(DB_DIR, exist_ok=True)
    np.save(EMB_FILE, embs.astype(np.float32))
    meta.to_csv(META_FILE, index=False)
    json.dump({"dim": int(embs.shape[1]) if embs.size else 0}, open(DIM_FILE, "w"))

def build_doc(row: pd.Series) -> str:
    return (
        f"SKU: {row['sku']}\n"
        f"Name: {row['name']}\n"
        f"Category: {row['category']}\n"
        f"Color: {row['color']}\n"
        f"Dimensions (W×D×H cm): {row['width_cm']}×{row['depth_cm']}×{row['height_cm']}\n"
        f"Description: {row['description']}\n"
        f"URL: {row['product_url']}"
    )

def reindex_products(df: pd.DataFrame):
    """Embed the uploaded CSV and REPLACE the current index (no SQLite required)."""
    required = ["sku", "name", "category", "description", "color",
                "width_cm", "depth_cm", "height_cm", "image_url", "product_url"]
    for col in required:
        if col not in df.columns:
            st.error(f"CSV missing required column: {col}")
            st.stop()

    total = len(df)
    prog = st.progress(0, text=f"Embedding 0/{total} items…")
    status = st.empty()

    embs: List[np.ndarray] = []
    metas: List[Dict] = []

    step = max(1, total // 100)
    t0 = time.time()

    for i, row in df.iterrows():
        doc = build_doc(row)
        try:
            v = embed_text(doc)
        except Exception as e:
            status.warning(f"Row {i+1}: embedding failed ({e}); skipping")
            continue
        # store meta
        metas.append({
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "color": str(row["color"]),
            "width_cm": float(row["width_cm"]),
            "depth_cm": float(row["depth_cm"]),
            "height_cm": float(row["height_cm"]),
            "image_url": str(row["image_url"]),
            "product_url": str(row["product_url"]),
        })
        embs.append(v)

        if (i + 1) % step == 0 or (i + 1) == total:
            prog.progress((i + 1) / max(1, total), text=f"Embedding {i + 1}/{total} items…")

    if not embs:
        st.error("No embeddings created. Check your CSV content.")
        return

    # Stack and save
    E = np.stack(embs).astype(np.float32)
    meta_df = pd.DataFrame(metas)
    save_store(E, meta_df)
    prog.progress(1.0, text=f"Done! Indexed {len(meta_df)} items in {time.time()-t0:.1f}s ✅")

def search_products(query: str, k: int = 8) -> List[Dict]:
    E, meta, dim = load_store()
    if E.size == 0 or meta.empty:
        return []

    q = embed_text(query).astype(np.float32)
    if dim and q.shape[0] != dim:
        st.warning("Embedding dimension mismatch; reindex your CSV.")
        return []

    # cosine similarity
    q_norm = np.linalg.norm(q) + 1e-8
    E_norms = np.linalg.norm(E, axis=1) + 1e-8
    sims = (E @ q) / (E_norms * q_norm)
    idx = np.argsort(-sims)[:k]

    out: List[Dict] = []
    for i in idx:
        row = meta.iloc[int(i)].to_dict()
        row["score"] = float(sims[int(i)])
        out.append(row)
    return out

# ---------------------- UI Tabs ----------------------
tab_chat, tab_ingest, tab_help = st.tabs(["💬 Chat & Recommend", "📦 Ingest Products (CSV)", "❓ Help"])

with tab_ingest:
    st.header("Load product catalog from CSV")
    st.caption("Required columns: **sku, name, category, description, color, width_cm, depth_cm, height_cm, image_url, product_url**")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except UnicodeDecodeError:
            up.seek(0)
            df = pd.read_csv(up, encoding="latin-1")
        st.write(df.head(3))
        if st.button("Embed & index products", type="primary"):
            reindex_products(df)

with tab_chat:
    st.header("Ask about products / get recommendations")
    cols = st.columns([2, 1])

    with cols[0]:
        user_query = st.text_area(
            "Ask anything (e.g., 'sofa untuk ruangan 3x4 meter, beige, modern, budget 5–7 juta')",
            height=120,
        )
        profile_notes = st.text_input("Optional: customer notes (kids? pets? preferred style?)", "")
        budget = st.text_input("Optional: budget range (e.g., 5–7 juta)", "")

        if st.button("🤖 Generate recommendations", type="primary", disabled=not user_query.strip()):
            # save user message
            st.session_state["history"].append({"role": "user", "parts": [user_query]})

            # retrieve top-k items
            prog = st.progress(0, text="Searching catalog…")
            t0 = time.time()
            matches = search_products(user_query, k=8)
            prog.progress(0.2, text="Preparing prompt…")

            ctx_lines = [
                f"- {m.get('name', '(no name)')} | {m.get('category', '')} | "
                f"{m.get('color', '')} | {m.get('width_cm', '?')}×{m.get('depth_cm', '?')}×{m.get('height_cm', '?')} cm | "
                f"{m.get('product_url', '')}"
                for m in matches
            ]
            catalog_ctx = "\n".join(ctx_lines) if ctx_lines else "(no matches)"

            system_prompt = (
                "You are a helpful retail assistant for a furniture store. "
                "Use the provided 'CATALOG CONTEXT' as the primary source for recommendations. "
                "Cite product names and include product_url links from the context when recommending. "
                "Be concise, prefer bullet points, and explain why each item fits the user's room and constraints."
            )

            prompt = (
                f"USER QUERY:\n{user_query}\n\n"
                f"CUSTOMER NOTES:\n{profile_notes}\n\n"
                f"BUDGET:\n{budget}\n\n"
                f"CATALOG CONTEXT:\n{catalog_ctx}\n\n"
                "Write clear bullet-point recommendations and include the product_url links."
            )

            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)

            placeholder = st.empty()
            full_text = ""
            try:
                prog.progress(0.4, text="Thinking with Gemini…")
                stream = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    stream=True,
                )
                for chunk in stream:
                    token = getattr(chunk, "text", "") or ""
                    if token:
                        full_text += token
                        placeholder.markdown(full_text)
                        prog.progress(min(0.95, 0.4 + len(full_text) / 4000.0), text="Generating…")

                prog.progress(1.0, text=f"Done in {time.time()-t0:.1f}s ✅")
                st.session_state["history"].append({"role": "model", "parts": [full_text]})

            except Exception as e:
                prog.progress(0.0, text="Failed")
                st.error(f"Model error: {e}")

    with cols[1]:
        st.subheader("Catalog matches (auto)")
        q_preview = (user_query or "").strip()
        if q_preview:
            try:
                for m in search_products(q_preview, k=5):
                    st.write(f"**{m.get('name','(no name)')}**")
                    st.caption(f"{m.get('category','')} · {m.get('color','')} · "
                               f"{m.get('width_cm','?')}×{m.get('depth_cm','?')}×{m.get('height_cm','?')} cm")
                    url = m.get("product_url","")
                    if url:
                        st.write(url)
                    st.divider()
            except Exception:
                st.caption("No matches yet. Upload your CSV in the Ingest tab.")

with tab_help:
    st.markdown("""
### How to use
1. Go to **📦 Ingest Products (CSV)** and upload your product catalog. This rebuilds the vector index (no database required).
2. Switch to **💬 Chat & Recommend** and ask for what you need (room size, style, color, budget).
3. The assistant grounds answers on your catalog and includes product links.

**Note:** Visualization & background removal were removed. Uses a lightweight NumPy vector index instead of Chroma/SQLite.
""")
