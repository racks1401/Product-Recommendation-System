from flask import Flask, request, render_template, redirect, url_for
import hashlib
import os
import random
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from text_repair import repair_encoding

app = Flask(__name__)

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "models"
# CLEAN_DATA_PATH can point the app at another build (the tests use this).
CLEAN_DATA_PATH = Path(
    os.environ.get("CLEAN_DATA_PATH", MODELS_DIR / "clean_data.csv")
)

if not CLEAN_DATA_PATH.exists():
    sys.exit(
        f"{CLEAN_DATA_PATH} not found. Build it first with:\n"
        "    python build_data.py"
    )

trending_products = pd.read_csv(MODELS_DIR / "trending_products.csv")
train_data = pd.read_csv(CLEAN_DATA_PATH)

_missing_columns = [
    c for c in ("Price", "DescriptionText") if c not in train_data.columns
]
if _missing_columns:
    sys.exit(
        f"{CLEAN_DATA_PATH} has no {', '.join(_missing_columns)} column "
        "(built by an older version). Rebuild it with:\n"
        "    python build_data.py"
    )

_BROKEN_QUOTE = re.compile(r"&quot;?")


def clean_name(name):
    """Product name as it should read.

    - The source dropped the ";" from &quot in some names (e.g. 12&quot,
      &quotLiv&quotin). Other "&" text (P&G, J&J) is left alone.
    - A few names are garbled by an encoding mix-up (e.g.
      "SuperStay 24Ã‚Â Liquid Lipstick"); see text_repair.py.

    Applied to the data and to names that arrive in URLs and forms, so
    links and typed names that still use the old form keep working.
    """
    name = _BROKEN_QUOTE.sub('"', str(name))
    repaired = repair_encoding(name)
    if repaired != name:
        # The repair can turn a garbled character into a non-breaking
        # space; use an ordinary one so a typed name still matches.
        repaired = repaired.replace("\xa0", " ")
    return repaired


# Keep product rows usable by the web app.
train_data["Name"] = train_data["Name"].fillna("").astype(str).map(clean_name)
trending_products["Name"] = trending_products["Name"].fillna("").astype(str).map(clean_name)
train_data["Tags"] = train_data["Tags"].fillna("").astype(str)
train_data["Category"] = train_data["Category"].fillna("").astype(str)
train_data["Brand"] = train_data["Brand"].fillna("").astype(str)
# Readable product description for the product page; "" when there is none.
train_data["DescriptionText"] = train_data["DescriptionText"].fillna("").astype(str)

# Price is in USD; missing prices stay NaN and are shown as unavailable.
train_data["Price"] = pd.to_numeric(train_data["Price"], errors="coerce")

# Local fallback images
FALLBACK_IMAGE_URLS = [
    "/static/img/img_1.png",
    "/static/img/img_2.png",
    "/static/img/img_3.png",
    "/static/img/img_4.png",
    "/static/img/img_5.png",
    "/static/img/img_6.png",
    "/static/img/img_7.png",
    "/static/img/img_8.png",
]

_IMAGE_URL = re.compile(r"^https?://\S+$")


def first_image_url(value):
    """First usable URL from an ImageURL field, or "" if there is none.

    The field often holds several URLs joined with " | "; only one can go
    into an <img src>.
    """
    if not isinstance(value, str):
        return ""
    for part in value.split("|"):
        part = part.strip()
        if _IMAGE_URL.match(part):
            return part
    return ""


def fallback_image(name):
    """A local placeholder image picked from the product name.

    Uses a stable hash (Python's hash() changes between runs), so a product
    always gets the same placeholder.
    """
    digest = hashlib.md5(str(name).lower().encode("utf-8")).digest()
    return FALLBACK_IMAGE_URLS[int.from_bytes(digest[:4], "big") % len(FALLBACK_IMAGE_URLS)]


def prepare_images(df):
    """Give every row one ImageURL, plus a FallbackImage for the browser to
    use if that URL fails to load."""
    df["FallbackImage"] = df["Name"].map(fallback_image)
    df["ImageURL"] = df["ImageURL"].map(first_image_url)
    missing = df["ImageURL"] == ""
    df.loc[missing, "ImageURL"] = df.loc[missing, "FallbackImage"]


prepare_images(train_data)
prepare_images(trending_products)

# Products are identified by name (URLs are /product/<name>), and some
# names appear on several rows with different prices and images. Every row
# gets the price and image of the first row with that name, which is the
# row the product page shows, so a product looks the same on every page.
_name_key = train_data["Name"].str.lower()
_first_row_by_name = (
    train_data.assign(_name_key=_name_key)
    .drop_duplicates(subset=["_name_key"])
    .set_index("_name_key")
)
for _column in ("Price", "ImageURL"):
    train_data[_column] = _name_key.map(_first_row_by_name[_column])

# -------------------------------------------------------------------
# Search model
# -------------------------------------------------------------------
# We now build one TF-IDF representation from:
#   product name + category + brand + tags
# and rank products by cosine similarity to the user's query.
train_data["SearchText"] = (
    train_data["Name"]
    + " "
    + train_data["Category"]
    + " "
    + train_data["Brand"]
    + " "
    + train_data["Tags"]
).str.lower()

search_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
)
search_matrix = search_vectorizer.fit_transform(train_data["SearchText"])


def search_products(query, top_n=20):
    """Return products ranked by semantic-ish TF-IDF similarity."""
    query = query.strip()
    if not query:
        return pd.DataFrame()

    query_vector = search_vectorizer.transform([query.lower()])
    scores = linear_kernel(query_vector, search_matrix).ravel()

    result = train_data.copy()
    result["SearchScore"] = scores

    # Give a small boost to products whose names contain the complete
    # query. This helps exact product-name searches without making
    # normal keyword searches dependent on exact spelling.
    name_contains = result["Name"].str.lower().str.contains(
        query.lower(), regex=False, na=False
    )
    result.loc[name_contains, "SearchScore"] += 0.25

    # No meaningful match: don't show random unrelated products.
    result = result[result["SearchScore"] > 0].copy()

    if result.empty:
        return result

    result = (
        result.sort_values(
            ["SearchScore", "Rating", "ReviewCount"],
            ascending=[False, False, False],
        )
        .drop_duplicates(subset=["Name"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# -------------------------------------------------------------------
# Recommendation model
# -------------------------------------------------------------------
# Remove duplicate product names so the recommendation carousel does
# not contain the same product multiple times.
recommendation_data = (
    train_data[train_data["Name"].str.strip() != ""]
    .drop_duplicates(subset=["Name"])
    .reset_index(drop=True)
)

# Related-product cards show the rating of the product page they link to.
# Product pages look names up case-insensitively, so for the few names that
# differ only in case, that is the first row of the case-insensitive name.
# (Only this display copy is changed; search still uses train_data's Rating.)
recommendation_data["Rating"] = (
    recommendation_data["Name"].str.lower().map(_first_row_by_name["Rating"])
)

recommendation_vectorizer = TfidfVectorizer(stop_words="english")
recommendation_matrix = recommendation_vectorizer.fit_transform(
    recommendation_data["Tags"]
)


def content_based_recommendations(item_name, top_n=10):
    """Recommend products similar to item_name using TF-IDF + cosine similarity."""
    if not item_name:
        return pd.DataFrame()

    matches = recommendation_data[
        recommendation_data["Name"].str.lower() == str(item_name).lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    item_index = matches.index[0]

    # Compute similarity only for the selected product instead of
    # building a full N x N cosine-similarity matrix on every request.
    similarities = linear_kernel(
        recommendation_matrix[item_index : item_index + 1],
        recommendation_matrix,
    ).ravel()

    # Exclude the selected product itself.
    similarities[item_index] = -1

    top_indices = similarities.argsort()[::-1][:top_n]
    result = recommendation_data.iloc[top_indices].copy()
    result["SimilarityScore"] = similarities[top_indices]

    return result[
        ["Name", "ReviewCount", "Brand", "ImageURL", "FallbackImage", "Rating",
         "Price", "SimilarityScore"]
    ].reset_index(drop=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def truncate(text, length):
    if pd.isna(text):
        return ""
    text = str(text)
    return text[:length] + "..." if len(text) > length else text


@app.template_test("rated")
def is_rated(value):
    """True if a product has a real rating.

    Ratings are on a 1-5 scale; build_data.py (like the notebook) fills
    missing ratings with 0, so 0 or an empty value means "not rated".
    """
    try:
        return float(value) > 0  # NaN > 0 is False too
    except (TypeError, ValueError):
        return False


@app.template_filter("price")
def format_price(value):
    """Format a USD price for display, e.g. 14.95 -> "$14.95"."""
    if value is None or pd.isna(value):
        return "Price unavailable"
    return f"${float(value):,.2f}"


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    products = trending_products.head(8).to_dict(orient="records")

    return render_template(
        "index.html",
        trending_products=products,
        truncate=truncate,
    )


@app.route("/main")
def main():
    query = request.args.get("query", "").strip()

    if query:
        matched = search_products(query, top_n=20)

        if matched.empty:
            message = f'No relevant products found for "{query}".'
            products = []
        else:
            message = f'Showing the best matches for "{query}".'
            products = matched.to_dict(orient="records")
    else:
        # Keep the old behavior for an empty search, but make it explicit.
        products = train_data.sample(
            n=min(20, len(train_data)), random_state=random.randint(1, 9999)
        ).to_dict(orient="records")
        message = "Showing random products. Search for a product to get ranked results."

    return render_template(
        "main.html",
        products=products,
        truncate=truncate,
        message=message,
    )


@app.route("/product/<path:product_name>")
def product_page(product_name):
    product_name = clean_name(urllib.parse.unquote(product_name))

    matches = train_data[
        train_data["Name"].str.lower() == product_name.lower()
    ]

    if matches.empty:
        matches = train_data[
            train_data["Name"].str.lower().str.contains(
                product_name.lower(), regex=False, na=False
            )
        ]

    if matches.empty:
        return f"Product '{product_name}' not found.", 404

    product_row = matches.iloc[0]
    dataset_name = product_row["Name"]

    rec = content_based_recommendations(dataset_name, top_n=8)
    recommendations = rec.to_dict(orient="records")

    product = product_row.to_dict()

    return render_template(
        "product.html",
        product=product,
        recommendations=recommendations,
        truncate=truncate,
    )


@app.route("/index")
def indexredirect():
    return redirect(url_for("index"))


@app.route("/recommendations", methods=["POST", "GET"])
def recommendations():
    if request.method == "POST":
        prod = clean_name(request.form.get("prod", "").strip())

        try:
            nbr = int(request.form.get("nbr", 10))
        except (TypeError, ValueError):
            nbr = 10

        nbr = max(1, min(nbr, 50))
        rec = content_based_recommendations(prod, top_n=nbr)

        if rec.empty:
            return render_template(
                "main.html",
                products=[],
                truncate=truncate,
                message=f'No recommendations available for "{prod}".',
            )

        products = rec.to_dict(orient="records")

        return render_template(
            "main.html",
            products=products,
            truncate=truncate,
            message=f'Recommended products similar to "{prod}".',
        )

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
