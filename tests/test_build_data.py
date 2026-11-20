"""Building models/clean_data.csv from the TSV (build_data.py)."""

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import build_data

EXPECTED_COLUMNS = [
    "ID", "ProdID", "Rating", "ReviewCount", "Category", "Brand", "Name",
    "ImageURL", "Description", "Tags", "Price", "DescriptionText",
]


def test_build_has_every_tsv_row_and_expected_columns(built, tsv):
    assert len(built) == len(tsv) == 5000
    assert list(built.columns) == EXPECTED_COLUMNS
    assert built["Name"].tolist() == tsv["Product Name"].tolist()


def test_prices_are_copied_from_tsv(built, tsv):
    # Compared as numbers: the CSV writes 8.50 as 8.5.
    built_prices = pd.to_numeric(built["Price"])
    tsv_prices = pd.to_numeric(tsv["Product Price"])
    assert built_prices.equals(tsv_prices)
    assert (built["Price"] == "").sum() == 42  # products with no price


def test_missing_ratings_are_stored_as_zero_and_real_ratings_kept(built, tsv):
    missing = tsv["Product Rating"] == ""
    assert (built.loc[missing, "Rating"].astype(float) == 0).all()
    assert (
        built.loc[~missing, "Rating"].astype(float)
        == tsv.loc[~missing, "Product Rating"].astype(float)
    ).all()


def test_image_field_is_kept_as_in_tsv(built, tsv):
    # The build keeps the raw field; the app picks the single URL.
    assert built["ImageURL"].tolist() == tsv["Product Image Url"].tolist()


# --- spaCy tagging -------------------------------------------------------
# Expected values below are copied from the saved outputs of the original
# Colab run of Recommendation_System.ipynb (which used en_core_web_sm).

NOTEBOOK_BRAND_TAGS = {
    "($100 Value) Peter Thomas Roth Thermal Therapy": "peter, thomas, roth",
    "got2b Kinkier Curling Mousse 8 oz (Pack of 2)": "göt2b",
    "Kokie Professional Matte Lipstick, Hot Berry, 0.14 fl oz": "kokie, cosmetics",
    "L.A. Colors Matte Lipstick, Tender Matte": "colors",
    "OPI Nail Polish, Strawberry Margarita, 0.5 Fl Oz": "opi",
}


@pytest.mark.parametrize("name,brand_tags", NOTEBOOK_BRAND_TAGS.items())
def test_brand_tags_match_notebook(built, name, brand_tags):
    rows = built[built["Name"].str.startswith(name)]
    assert not rows.empty, name
    assert rows["Brand"].iloc[0] == brand_tags


def _notebook_recommendations(built, item_name, top_n):
    """The notebook's own content-based recommendation code (cells 26-31)."""
    matrix = TfidfVectorizer(stop_words="english").fit_transform(built["Tags"])
    sims = cosine_similarity(matrix, matrix)
    idx = built[built["Name"] == item_name].index[0]
    ranked = sorted(enumerate(sims[idx]), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[1 : top_n + 1]]


@pytest.mark.parametrize(
    "item_name,expected_rows",
    [
        # notebook cell 30
        ("OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath",
         [156, 184, 205, 237, 325, 375, 402, 706, 886]),
        # notebook cell 33
        ("Kokie Professional Matte Lipstick, Hot Berry, 0.14 fl oz",
         [3406, 546, 2406, 4050, 4084, 1559, 2873, 3023]),
    ],
)
def test_tags_reproduce_notebook_recommendations(built, item_name, expected_rows):
    # These depend on the Tags of all 5000 rows, so they catch any
    # widespread change in tokenization or stop-word handling.
    assert _notebook_recommendations(built, item_name, len(expected_rows)) == expected_rows


def test_tags_combine_category_brand_and_description(built):
    joined = built["Category"] + ", " + built["Brand"] + ", " + built["Description"]
    assert (built["Tags"] == joined).all()


def test_tags_are_lowercase_words_without_stop_words(built):
    tokens = pd.Series(
        [t for tags in built["Tags"].head(500) for t in tags.split(", ") if t]
    )
    assert tokens.str.isalnum().all()
    assert (tokens == tokens.str.lower()).all()
    assert not tokens.isin(["the", "and", "for", "with"]).any()


# --- readable descriptions ----------------------------------------------

def test_description_text_is_empty_only_when_there_is_no_description(built, tsv):
    raw = tsv["Product Description"].str.strip()
    placeholders = raw.isin(["{}", "@generated"])
    assert ((built["DescriptionText"] == "") == ((raw == "") | placeholders)).all()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Say &quotyes&quot to it", 'Say "yes" to it'),
        ("6.3&quot; long", '6.3" long'),
        ("crÃ¨me brulee", "crème brulee"),
        ("Quick factsÂ\xa0 â€¢ Fantastic", "Quick facts • Fantastic"),
        ("Heading BiorÃ Brilliant", "Heading Bioré Brilliant"),
        ("Wicked Coolâ„¢ Toothpaste", "Wicked Cool™ Toothpaste"),
        ("R&B and P&G stay as they are", "R&B and P&G stay as they are"),
        ("Great scent.  var _mtm = _mtm || []; _mtm.push({});", "Great scent."),
        ("Gel. (function(w,d,s,l,i){w[l]=1})(window);", "Gel."),
        ("  lots   of\xa0\xa0space  ", "lots of space"),
        ("{}", ""),
        ("@generated", ""),
        ("", ""),
    ],
)
def test_clean_description(raw, expected):
    assert build_data.clean_description(raw) == expected


def test_clean_descriptions_leave_no_encoding_damage_or_scripts(built):
    text = built["DescriptionText"]
    assert not text.str.contains(r"Ã|â€|â„|Â|&quot|_mtm|_paq|gtm\.js", regex=True).any()
