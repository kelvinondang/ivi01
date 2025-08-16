import os
import io
import csv
import time
import base64
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from PIL import Image, ImageOps
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import pandas as pd

# Optional image background remover
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ---------------------- App / Keys ----------------------
st.set_page_config(page_title="Store Assistant — Furniture", page_icon="🛋️", layout="wide")
load_dotenv()

st.title("🛋️ Store Assistant — Furniture (Gemini + RAG + Vision)")

# Get API key (env or sidebar)
DEFAULT_MODEL = "gemini-1.5-flash"
api_key = os.getenv("GOOGLE_API_KEY", "")

# ---- Session state (init once) ----
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.subheader("🔑 API & Settings")
    api_key = st.text_input("GOOGLE_API_KEY", value=api_key, type="password", help="Paste your Gemini API key. Prefer .env for local, Secrets for cloud.")
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, 64)
    st.caption("Tip: Flash = fast & cheap. Pro = better reasoning.")
    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.rerun()

if not api_key:
    st.error("Missing API key. Add it in the sidebar (GOOGLE_API_KEY).")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------- Vector DB (Chroma) ----------------------
DB_PATH = "db"
client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None  # We'll pass embeddings manually from Gemini
)

EMBED_MODEL = "text-embedding-004"

def embed_text(text: str):
    # Gemini embedding
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=text)
        return resp["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        raise

def upsert_products(df: pd.DataFrame):
    # Required columns
    required = ["sku", "name", "category", "description", "color",
                "width_cm", "depth_cm", "height_cm", "image_url", "product_url"]
    for col in required:
        if col not in df.columns:
            st.error(f"CSV missing required column: {col}")
            st.stop()

    total = len(df)
    progress = st.progress(0, text=f"Indexing 0/{total} items…")
    status = st.empty()

    ids, docs, metas, embeds = [], [], [], []
    # Update roughly every 1% (at least every 1 row for very small files)
    step = max(1, total // 100)

    for idx, row in df.iterrows():
        # Build the text that will be embedded
        doc = (
            f"SKU: {row['sku']}\n"
            f"Name: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"Color: {row['color']}\n"
            f"Dimensions (W×D×H cm): {row['width_cm']}×{row['depth_cm']}×{row['height_cm']}\n"
            f"Description: {row['description']}\n"
            f"URL: {row['product_url']}"
        )
        ids.append(str(row["sku"]))
        docs.append(doc)
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

        # Embed with Gemini
        try:
            embeds.append(embed_text(doc))
        except Exception as e:
            status.warning(f"Row {idx+1}: embedding failed ({e})")
            # skip this row and continue
            ids.pop(); docs.pop(); metas.pop()
            continue

        # Update progress bar periodically
        if (idx + 1) % step == 0 or (idx + 1) == total:
            progress.progress((idx + 1) / max(1, total),
                              text=f"Indexing {idx + 1}/{total} items…")

    # Upsert to Chroma in chunks
    CHUNK = 64
    for i in range(0, len(ids), CHUNK):
        collection.upsert(
            ids=ids[i:i+CHUNK],
            documents=docs[i:i+CHUNK],
            metadatas=metas[i:i+CHUNK],
            embeddings=embeds[i:i+CHUNK],
        )

    progress.progress(1.0, text=f"Done! Indexed {len(ids)} items.")
    status.info("You can now switch to **Chat & Recommend**.")

def search_products(query: str, k: int = 8):
    q_emb = embed_text(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    items = []
    if res and res["ids"] and len(res["ids"]) > 0:
        for idx in range(len(res["ids"][0])):
            meta = res["metadatas"][0][idx]
            doc = res["documents"][0][idx]
            dist = res["distances"][0][idx]
            score = 1 - dist  # cosine similarity proxy
            items.append({"meta": meta, "doc": doc, "score": float(score)})
    return items

# ---------------------- Helpers ----------------------
def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    return img

def remove_bg(img: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        return img  # fallback: no bg removal
    # rembg returns raw bytes
    b = io.BytesIO()
    img.save(b, format="PNG")
    out = rembg_remove(b.getvalue())
    return Image.open(io.BytesIO(out)).convert("RGBA")

def composite(room_img: Image.Image, product_img: Image.Image, scale=100, x=0, y=0, rotation_deg=0):
    # scale product
    w, h = product_img.size
    new_w = max(1, int(w * (scale / 100.0)))
    new_h = max(1, int(h * (scale / 100.0)))
    prod = product_img.resize((new_w, new_h), Image.LANCZOS)
    if rotation_deg:
        prod = prod.rotate(rotation_deg, expand=True)
    # paste onto room
    canvas = room_img.convert("RGBA").copy()
    canvas.alpha_composite(prod, dest=(x, y))
    return canvas

def format_products_md(products):
    lines = []
    for i, item in enumerate(products, 1):
        m = item["meta"]
        lines.append(f"**{i}. {m['name']}**  \\SKU: {m['sku']} | {m['category']} | {m['color']}  \\Size (cm): {m['width_cm']}×{m['depth_cm']}×{m['height_cm']}  \\Link: {m['product_url']}")
    return "\n\n".join(lines) if lines else "_No products found yet. Upload a CSV in the Ingest tab._"

# ---------------------- Tabs ----------------------
tab_chat, tab_visual, tab_ingest, tab_help = st.tabs(["💬 Chat & Recommend", "🖼️ Visualize in Room", "📦 Ingest Products (CSV)", "❓ Help"])

# ---------- Ingest Tab ----------
with tab_ingest:
    st.subheader("Load product catalog from CSV")
    st.write("Required columns: `sku, name, category, description, color, width_cm, depth_cm, height_cm, image_url, product_url`.")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        with st.spinner("Embedding and indexing products…"):
            upsert_products(df)
        st.success(f"Loaded {len(df)} products into the vector DB.")
    st.caption("Tip: Export product data from your catalog/ERP or scrape your site into this format.")

# ---------- Chat Tab ----------
with tab_chat:
    st.subheader("Ask about products / get recommendations")

    colA, colB = st.columns([3, 2])
    with colA:
        user_query = st.text_area(
            "Ask anything (e.g., 'sofa for 3x4m living room, beige, modern, budget 5–7 juta')",
            height=100
        )
        uploaded_room = st.file_uploader(
            "Optional: upload room photo (JPG/PNG) to analyze style & color",
            type=["png", "jpg", "jpeg"],
            key="room_for_chat"
        )
        profile_notes = st.text_input("Optional: customer profile notes (pets? kids? style? brand prefs?)")
        budget = st.text_input("Optional: budget range (e.g., 5-7 juta)")

    with colB:
        st.markdown("**Catalog matches (auto):**")
        products = search_products(user_query) if user_query else []
        st.markdown(format_products_md(products))

    if st.button("🤖 Generate recommendations", type="primary", disabled=not user_query):
        sys_prompt = """You are a retail store assistant for a furniture brand.
        Use the provided product context to recommend the best matches for the user's room and preferences.
        Always explain *why* each item fits (size, color harmony, style, budget).
        If dimensions or constraints are missing, ask exactly 2 clarifying questions at the end.
        Output a concise, scannable answer.
        """

        # Build context from products
        ctx = "\n\n".join([p["doc"] for p in products])

        inputs = []
        if uploaded_room:
            room_img = Image.open(uploaded_room)
            inputs.append(room_img)

        prompt = (
            f"USER QUERY:\n{user_query}\n\n"
            f"CUSTOMER PROFILE:\n{profile_notes}\n\n"
            f"BUDGET:\n{budget}\n\n"
            f"CATALOG CONTEXT (top-K):\n{ctx}\n\n"
            f"Write recommendations with bullets and include product links."
        )

        # ---- Progress UI
        prog = st.progress(0, text="Preparing…")
        status = st.empty()
        t0 = time.time()

        try:
            # 1) Catalog search already happened above
            prog.progress(0.25, text="Searching catalog…")

            # 2) Ask model and stream tokens
            model = genai.GenerativeModel(model_name, system_instruction=sys_prompt)
            parts = inputs + [prompt] if inputs else [prompt]

            prog.progress(0.50, text="Asking Gemini…")
            placeholder = st.empty()
            full_text = ""

            stream = model.generate_content(
                parts,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                stream=True,
            )

            # 3) Stream output + advance progress
            for chunk in stream:
                token = getattr(chunk, "text", "") or ""
                if token:
                    full_text += token
                    placeholder.markdown(full_text)
                    approx = min(0.5 + len(full_text) / 4000.0, 0.98)
                    prog.progress(approx, text="Generating answer…")

            elapsed = time.time() - t0
            prog.progress(1.0, text=f"Done in {elapsed:.1f}s ✅")
            st.session_state.history.append({"role": "model", "parts": [full_text]})

        except Exception as e:
            prog.progress(0.0, text="Failed")
            status.error(f"Model error: {e}")

# ---------- Visualize Tab ----------
with tab_visual:
    st.subheader("Place product into a room image")
    c1, c2 = st.columns([2, 2])

    with c1:
        room_upload = st.file_uploader("Room photo (JPG/PNG)", type=["png", "jpg", "jpeg"], key="room_for_viz")
        room_img = None
        if room_upload:
            room_img = Image.open(room_upload).convert("RGBA")
            st.image(room_img, caption="Room", use_column_width=True)

    with c2:
        prod_url = st.text_input("Product image URL (PNG/JPG). Prefer background-light images.")
        if st.button("Load product image", disabled=not prod_url):
            try:
                prog = st.progress(0, text="Fetching image…")
                t0 = time.time()
                pimg = fetch_image(prod_url)
                prog.progress(0.4, text="Decoding image…")
                if REMBG_AVAILABLE:
                    prog.progress(0.6, text="Removing background…")
                    pimg = remove_bg(pimg)
                st.session_state["product_img"] = pimg
                prog.progress(1.0, text=f"Ready in {time.time()-t0:.1f}s ✅")
                st.success("Product image loaded.")
            except Exception as e:
                st.error(f"Failed to load product image: {e}")

        if "product_img" in st.session_state:
            st.image(st.session_state["product_img"], caption="Product", use_column_width=True)

    st.markdown("---")
    if room_img is not None and "product_img" in st.session_state:
        scale = st.slider("Scale %", 10, 400, 120, 1)
        x = st.slider("X position (px)", 0, max(1, room_img.size[0]-1), 50, 1)
        y = st.slider("Y position (px)", 0, max(1, room_img.size[1]-1), 50, 1)
        rot = st.slider("Rotation (deg)", -45, 45, 0, 1)

        if st.button("🖼️ Compose"):
            prog = st.progress(0, text="Compositing…")
            t0 = time.time()
            composed = composite(
                room_img,
                st.session_state["product_img"],
                scale=scale,
                x=x,
                y=y,
                rotation_deg=rot
            )
            prog.progress(0.7, text="Preparing download…")
            st.image(composed, caption="Composite preview", use_column_width=True)
            buf = io.BytesIO()
            composed.save(buf, format="PNG")
            prog.progress(1.0, text=f"Done in {time.time()-t0:.1f}s ✅")
            st.download_button("Download PNG", data=buf.getvalue(), file_name="composite.png", mime="image/png")
    else:
        st.info("Upload a room photo and load a product image to start compositing.")

# ---------- Help Tab ----------
with tab_help:
    st.markdown("""
    ### How to use
    1. Go to **📦 Ingest Products (CSV)** and upload your catalog as CSV (use the required columns).
    2. In **💬 Chat & Recommend**, type a question (room size, color, style). Optionally upload a room photo.
    3. Read the top catalog matches and the AI's recommendations.
    4. In **🖼️ Visualize in Room**, upload a room photo and paste a product image URL to place it visually, then download the composite.

    ### CSV format (columns)
    `sku, name, category, description, color, width_cm, depth_cm, height_cm, image_url, product_url`

    ### Tips
    - For best results, include **detailed descriptions** and **dimensions** in the CSV.
    - Product images with uniform backgrounds work best for background removal.
    - If background removal fails, try using images with clear product edges or transparent PNGs.
    - Scale/position controls help approximate the real size; for precise scale, measure the wall in cm and compute pixels-per-cm manually.
    """)