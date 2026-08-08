from flask import Flask, request, render_template, redirect, url_for
import random
import urllib.parse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Keep product rows usable by the web app.
train_data["Name"] = train_data["Name"].fillna("").astype(str)
train_data["Tags"] = train_data["Tags"].fillna("").astype(str)
train_data["Category"] = train_data["Category"].fillna("").astype(str)
train_data["Brand"] = train_data["Brand"].fillna("").astype(str)

# Local fallback images
RANDOM_IMAGE_URLS = [
    "/static/img/img_1.png",
    "/static/img/img_2.png",
    "/static/img/img_3.png",
    "/static/img/img_4.png",
    "/static/img/img_5.png",
    "/static/img/img_6.png",
    "/static/img/img_7.png",
    "/static/img/img_8.png",
]

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
        ["Name", "ReviewCount", "Brand", "ImageURL", "Rating", "SimilarityScore"]
    ].reset_index(drop=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def truncate(text, length):
    if pd.isna(text):
        return ""
    text = str(text)
    return text[:length] + "..." if len(text) > length else text


def add_fallback_images(products):
    """Ensure every product has an image usable by the template."""
    products = products.copy()
    for product in products:
        image = product.get("ImageURL")
        if not image or pd.isna(image):
            product["ImageURL"] = random.choice(RANDOM_IMAGE_URLS)
    return products


def product_price(product):
    # The supplied dataset does not contain a Price column, so these
    # are UI placeholders from the original project.
    return product.get("Price") if product.get("Price") else random.choice(
        [40, 50, 60, 70, 100, 122, 106]
    )


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    products = trending_products.head(8).to_dict(orient="records")
    products = add_fallback_images(products)

    return render_template(
        "index.html",
        trending_products=products,
        truncate=truncate,
        random_product_image_urls=RANDOM_IMAGE_URLS,
        random_price=random.choice([40, 50, 60, 70, 100, 122, 106]),
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

    products = add_fallback_images(products)

    return render_template(
        "main.html",
        products=products,
        truncate=truncate,
        random_price=random.choice([40, 50, 60, 70, 100, 122, 106]),
        message=message,
    )


@app.route("/product/<path:product_name>")
def product_page(product_name):
    product_name = urllib.parse.unquote(product_name)

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
    recommendations = add_fallback_images(rec.to_dict(orient="records"))

    product = product_row.to_dict()
    product = add_fallback_images([product])[0]

    return render_template(
        "product.html",
        product=product,
        recommendations=recommendations,
        truncate=truncate,
        random_product_image_urls=RANDOM_IMAGE_URLS,
        random_price=product_price(product),
    )


@app.route("/index")
def indexredirect():
    return redirect(url_for("index"))


@app.route("/recommendations", methods=["POST", "GET"])
def recommendations():
    if request.method == "POST":
        prod = request.form.get("prod", "").strip()

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
                random_price=50,
                message=f'No recommendations available for "{prod}".',
            )

        products = add_fallback_images(rec.to_dict(orient="records"))

        return render_template(
            "main.html",
            products=products,
            truncate=truncate,
            random_price=random.choice([40, 50, 60, 70, 100, 122, 106]),
            message=f'Recommended products similar to "{prod}".',
        )

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
