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

The original application did **not** use the ML model for the `/main?query=...` search page.

The old search logic was:

```text
query -> exact substring search inside Product Name
      -> if no match -> random products
```

So a query such as `shampoo` only worked when `shampoo` appeared in the product name. Any unmatched query fell back to random products, which made the search look broken.

The fixed application now uses:

```text
query
  -> TF-IDF query vector
  -> cosine similarity against product SearchText
  -> rank by relevance
  -> return top products
```

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
