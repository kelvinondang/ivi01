# Retail Store Assistant (Furniture) — Gemini + Streamlit

Purpose-built assistant for store teams:
- **Product Q&A** and **recommendations** using your catalog (RAG).
- **Room image analysis** (Gemini 1.5) to match style/color.
- **Product-on-room visualization** (background removal + overlay) to show customers how items look.

## 0) What you need
- Python 3.10+
- A Google Gemini API key (create in Google AI Studio)
- Your product catalog as **CSV** with columns:
  `sku, name, category, description, color, width_cm, depth_cm, height_cm, image_url, product_url`

## 1) Setup
```bash
# create / activate virtual env (Windows)
py -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with:
```env
GOOGLE_API_KEY=YOUR_KEY
```

## 2) Run
```bash
streamlit run app.py
```

## 3) Upload your catalog
- Go to the **Ingest Products (CSV)** tab → upload your CSV.
- The app builds a local Chroma vector DB in **./db** (persists across runs).

## 4) Recommend & Visualize
- Ask questions in **Chat & Recommend** (optionally upload room photos).
- Use **Visualize in Room** to overlay product images (paste product image URL).

## Deployment (Streamlit Cloud)
1. Push this folder to a GitHub repo.
2. In Streamlit Cloud → **New app** → select repo, main file `app.py`.
3. Add your key in **App settings → Secrets**:
   ```
   GOOGLE_API_KEY="YOUR_KEY"
   ```
4. Deploy.

## Notes
- Background removal uses **rembg** and may download a model on first use.
- For pixel-to-cm accurate scaling, you need a known reference in the room photo or manual measurement.
- This starter avoids scraping; use CSV exports from your site/ERP for reliability.