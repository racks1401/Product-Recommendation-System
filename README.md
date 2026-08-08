# Product Recommendation System

A Flask-based product search and recommendation system built from a Walmart product-review dataset.

## What the project does

The project has two related ML features:

1. **Product search**
   - User enters a keyword such as `shampoo`, `lipstick`, or `razor`.
   - TF-IDF converts the query and product information into vectors.
   - Cosine similarity ranks products by relevance.

2. **Content-based recommendations**
   - When a user opens a product, the system compares its product tags with all other products.
   - Products with the most similar TF-IDF vectors are shown as related products.

## Project structure

```text
product-recommendation-system/
├── app.py                         # Flask application + search/recommendation logic
├── Recommendation_System.ipynb    # Original ML training/preprocessing notebook
├── requirements.txt
├── .gitignore
├── models/
│   ├── clean_data.csv             # Processed product dataset
│   └── trending_products.csv      # Top-rated products used on the home page
├── templates/
│   ├── index.html                 # Search page
│   ├── main.html                  # Search results
│   └── product.html               # Product details + recommendations
└── static/
    ├── style.css
    └── img/                       # Fallback product images
```

## ML pipeline

```text
Raw Walmart TSV
      |
      v
Select useful columns
      |
      v
Fill missing values + rename columns
      |
      v
Clean Category / Brand / Description with spaCy
      |
      v
Create Tags
      |
      +------------------------+
      |                        |
      v                        v
Rating-based ranking      TF-IDF text representation
      |                        |
      v                        v
trending_products.csv    Cosine similarity
                               |
                               v
                       Related products
```

## Important distinction

## 🔎 ML-Powered Product Search

The system provides relevance-based product search using **TF-IDF vectorization and cosine similarity**.

Instead of relying only on exact product-name matching, each product is represented using a combined text representation containing:

- Product name
- Category
- Brand
- Processed product tags

The user's search query is transformed into a TF-IDF vector and compared against the product representations using cosine similarity. Products are then ranked by relevance and the most relevant results are returned.

```text
User Query
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Relevance Ranking
    ↓
Top Matching Products


`SearchText` combines:

- product name
- category
- brand
- processed tags

## Run locally

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- The supplied dataset does not contain a real `Price` column. The UI therefore still uses placeholder prices.
- Product images may come from the dataset's original URLs; local images are used as fallbacks.
- The original notebook uses spaCy's `en_core_web_sm` model for preprocessing. The exported `clean_data.csv` already contains processed tags, so the Flask app does not need spaCy at runtime.
