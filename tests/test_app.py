"""The Flask pages: prices, descriptions, images, ratings, search, related."""

import os
import subprocess
import sys

import pytest

from conftest import GARBLED_NAMES, PROJECT_DIR
from pages import (
    all_img_srcs, get, product_description, product_image, product_price,
    product_rating, product_title, product_url, related_cards, result_cards,
)

KOKIE = "Kokie Professional Matte Lipstick, Hot Berry, 0.14 fl oz"  # $5.16, unrated
OPI = "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath"  # unrated, no description


def dollars(raw_price):
    return f"${float(raw_price):,.2f}" if raw_price else "Price unavailable"


def first_url(raw_field):
    return raw_field.split("|")[0].strip()


# --- startup ---------------------------------------------------------------

def test_app_refuses_to_start_without_built_data(tmp_path):
    env = dict(os.environ, CLEAN_DATA_PATH=str(tmp_path / "missing.csv"))
    result = subprocess.run(
        [sys.executable, "-c", "import app"], cwd=PROJECT_DIR, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "python build_data.py" in result.stderr


@pytest.mark.parametrize("url", ["/", "/index", "/main", "/main?query=shampoo"])
def test_pages_load(client, url):
    assert client.get(url, follow_redirects=True).status_code == 200


def test_unknown_product_is_404(client):
    assert client.get(product_url("no such product xyz")).status_code == 404


# --- prices ----------------------------------------------------------------

def test_product_page_shows_real_price_in_dollars(client):
    assert product_price(get(client, product_url(KOKIE))) == "$5.16"


def test_product_prices_match_tsv(client, unique_names):
    for _, row in unique_names.sample(40, random_state=0).iterrows():
        page = get(client, product_url(row["display_name"]))
        assert product_price(page) == dollars(row["Product Price"]), row["Product Name"]


def test_product_without_price_says_unavailable(client, unique_names):
    row = unique_names[unique_names["Product Price"] == ""].iloc[0]
    assert product_price(get(client, product_url(row["display_name"]))) == "Price unavailable"


@pytest.mark.parametrize("query", ["shampoo", "razor", "shoelace"])
def test_search_card_prices_match_tsv_and_product_page(client, tsv_first_row, query):
    cards = result_cards(get(client, f"/main?query={query}"))
    assert cards
    for card in cards:
        expected = dollars(tsv_first_row.loc[card["name"].lower(), "Product Price"])
        assert card["p"] == expected, card["name"]
        assert product_price(get(client, product_url(card["name"]))) == expected


def test_repeated_names_look_the_same_on_search_card_and_product_page(client, tsv):
    # Some names appear on several rows with different prices or images.
    # The product page shows the first row, so search cards must too.
    key = tsv["name_key"]
    repeated = tsv[key.duplicated(keep=False)].groupby("name_key").filter(
        lambda g: g["Product Price"].nunique() > 1 or g["Product Image Url"].nunique() > 1
    )
    names = repeated["display_name"].drop_duplicates()
    assert len(names) > 20
    for name in names:
        cards = [c for c in result_cards(get(client, "/main", query_string={"query": name}))
                 if c["name"].lower() == name.lower()]
        assert cards, name
        page = get(client, product_url(cards[0]["name"]))
        assert cards[0]["p"] == product_price(page), name
        assert cards[0]["src"] == product_image(page)[0], name


def test_prices_are_the_same_on_every_load(client):
    search = [[c["p"] for c in result_cards(get(client, "/main?query=lip balm"))] for _ in range(3)]
    page = [product_price(get(client, product_url(KOKIE))) for _ in range(3)]
    assert search[0] == search[1] == search[2]
    assert page[0] == page[1] == page[2]


def test_no_rupee_prices_anywhere(client):
    for url in ["/main?query=shampoo", "/main", product_url(KOKIE)]:
        assert "₹" not in get(client, url)


# --- descriptions ------------------------------------------------------------

def test_product_page_shows_description_not_tags(client, tsv_first_row, built):
    page = get(client, product_url(KOKIE))
    description, missing = product_description(page)
    raw = tsv_first_row.loc[KOKIE.lower(), "Product Description"]
    assert not missing
    assert description == " ".join(raw.split())
    tags = built.loc[built["Name"] == KOKIE, "Tags"].iloc[0]
    assert tags not in page


def test_description_is_readable(client, tsv):
    row = tsv[tsv["Product Description"].str.contains("&quotyes&quot", regex=False)].iloc[0]
    description, _ = product_description(get(client, product_url(row["display_name"])))
    assert 'Say "yes" to gentle ingredients' in description
    assert "&quot" not in description


def test_product_without_description_says_so(client):
    description, missing = product_description(get(client, product_url(OPI)))
    assert missing
    assert description == "No description available for this product."


# --- images ------------------------------------------------------------------

def test_every_image_is_a_single_url(client):
    for url in ["/main?query=shampoo", "/main", product_url(KOKIE), product_url(OPI)]:
        srcs = all_img_srcs(get(client, url))
        assert srcs
        for src in srcs:
            assert "|" not in src and " " not in src, src


def test_images_use_first_tsv_url(client, tsv_first_row):
    page = get(client, product_url(KOKIE))
    src, _ = product_image(page)
    assert src == first_url(tsv_first_row.loc[KOKIE.lower(), "Product Image Url"])
    cards = result_cards(get(client, "/main?query=shampoo")) + related_cards(page)
    assert cards
    for card in cards:
        assert card["src"] == first_url(tsv_first_row.loc[card["name"].lower(), "Product Image Url"])


def test_image_fallback_is_a_local_file_and_the_same_on_every_load(client):
    loads = [get(client, "/main?query=shampoo") for _ in range(3)]
    fallbacks = [[c["fallback"] for c in result_cards(page)] for page in loads]
    assert fallbacks[0] == fallbacks[1] == fallbacks[2]
    for fallback in set(fallbacks[0]) | {product_image(get(client, product_url(OPI)))[1]}:
        assert fallback.startswith("/static/img/")
        assert client.get(fallback).status_code == 200


def test_image_fallback_is_the_same_across_server_restarts(built_csv):
    # Python's built-in hash() changes between runs; the fallback must not.
    script = f"import app; print(app.fallback_image({KOKIE!r}), app.fallback_image({OPI!r}))"
    outputs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, CLEAN_DATA_PATH=str(built_csv), PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_DIR,
                                env=env, capture_output=True, text=True, check=True)
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_product_with_no_usable_url_gets_its_fallback(app_module):
    import pandas as pd
    df = pd.DataFrame({"Name": ["A", "B"], "ImageURL": ["", "not a url | also not"]})
    app_module.prepare_images(df)
    assert df["ImageURL"].tolist() == df["FallbackImage"].tolist()
    assert all(u.startswith("/static/img/") for u in df["ImageURL"])


# --- ratings -----------------------------------------------------------------

def test_unrated_product_says_no_ratings_yet(client):
    page = get(client, product_url(OPI))
    assert product_rating(page) == "No ratings yet"
    assert "0.0" not in page


def test_rated_product_shows_its_rating(client, unique_names):
    row = unique_names[unique_names["Product Rating"] == "4.5"].iloc[0]
    assert product_rating(get(client, product_url(row["display_name"]))) == "⭐ 4.5"


def test_product_page_ratings_match_tsv(client, unique_names):
    for _, row in unique_names.sample(40, random_state=1).iterrows():
        raw = row["Product Rating"]
        expected = f"⭐ {float(raw)}" if raw else "No ratings yet"
        assert product_rating(get(client, product_url(row["display_name"]))) == expected


@pytest.mark.parametrize("name", [OPI, KOKIE])
def test_related_strip_ratings(client, tsv_first_row, name):
    cards = related_cards(get(client, product_url(name)))
    raws = [tsv_first_row.loc[c["name"].lower(), "Product Rating"] for c in cards]
    assert any(raws) and not all(raws), "want a mix of rated and unrated here"
    for card, raw in zip(cards, raws):
        assert card["p"] == (f"⭐ {float(raw)}" if raw else "No ratings yet"), card["name"]


# --- product names with &quot -------------------------------------------------------
# The source dropped the ";" from &quot in 79 product names (12&quot,
# &quotLiv&quotin). They should read with real quotes everywhere, and links,
# search and the recommendations form must keep working for them.

LIV_RAW = "OPI- Nail Lacquer-GelColor - &quotLiv&quotin the Gray -.5 FL OZ"
LIV = 'OPI- Nail Lacquer-GelColor - "Liv"in the Gray -.5 FL OZ'


@pytest.fixture(scope="module")
def quot_names(tsv):
    names = tsv.loc[tsv["Product Name"].str.contains("&quot", regex=False), "display_name"]
    names = names.drop_duplicates().tolist()
    assert len(names) == 79
    return names


def test_names_with_quot_read_properly_and_their_links_work(client, quot_names):
    for name in quot_names:
        # Search by the name finds the product ...
        page = get(client, "/main", query_string={"query": name})
        assert "&amp;quot" not in page and "&quot" not in page.replace("&quot;", "")
        cards = [c for c in result_cards(page) if c["name"] == name]
        assert cards, f"search did not find {name!r}"
        card = cards[0]
        # ... its card reads properly ...
        assert card["alt"] == name
        assert "&quot" not in card["title"]
        assert name.startswith(card["title"].removesuffix("..."))
        # ... and the card's link opens the right product page.
        product = get(client, card["href"])
        assert product_title(product) == name
        assert "&amp;quot" not in product


def test_related_products_with_quot_in_name_link_correctly(client, quot_names):
    seen = 0
    for name in quot_names:
        for card in related_cards(get(client, product_url(name))):
            if '"' in card["name"]:
                seen += 1
                assert "&quot" not in card["name"] + card["alt"] + card["title"]
                assert product_title(get(client, card["href"])) == card["name"]
    assert seen > 20


def test_old_links_with_quot_still_open_the_product(client):
    page = get(client, product_url(LIV_RAW))
    assert product_title(page) == LIV


def test_search_matches_words_that_were_stuck_to_quot(client):
    # Before the fix the search saw "quotliv" instead of "liv" and ranked
    # this product 6th for this query.
    names = [c["name"] for c in result_cards(get(client, "/main?query=Liv in the Gray"))]
    assert names[0] == LIV


@pytest.mark.parametrize("typed", [LIV, LIV_RAW])
def test_recommendations_form_accepts_names_with_quotes(client, typed):
    page = client.post("/recommendations", data={"prod": typed, "nbr": "5"}).get_data(as_text=True)
    cards = result_cards(page)
    assert len(cards) == 5
    assert LIV not in [c["name"] for c in cards]


def test_other_ampersands_in_names_are_unchanged(client, tsv):
    rows = tsv[tsv["Product Name"].str.contains(r"&(?!quot)", regex=True)]
    assert len(rows) > 5
    for name in rows["Product Name"].drop_duplicates().head(10):
        if "&quot" in name:
            continue
        assert product_title(get(client, product_url(name))) == name


# --- product names garbled by an encoding mix-up -------------------------------------

SUPERSTAY_RAW = "SuperStay 24Ã‚Â Liquid Lipstick"
SUPERSTAY = "SuperStay 24 Liquid Lipstick"
GARBLE_MARKS = ("Ã", "â€", "â„", "Â")


def test_garbled_names_are_all_in_the_data(tsv):
    garbled = tsv.loc[tsv["Product Name"].str.contains("|".join(GARBLE_MARKS)), "Product Name"]
    assert set(garbled) == set(GARBLED_NAMES)


@pytest.mark.parametrize("raw,name", GARBLED_NAMES.items())
def test_garbled_names_read_properly_and_their_links_work(client, raw, name):
    page = get(client, "/main", query_string={"query": name})
    cards = [c for c in result_cards(page) if c["name"] == name]
    assert cards, f"search did not find {name!r}"
    card = cards[0]
    assert card["alt"] == name
    assert name.startswith(card["title"].removesuffix("..."))
    product = get(client, card["href"])
    assert product_title(product) == name
    assert "\xa0" not in product_title(product)
    assert not any(mark in product_title(product) for mark in GARBLE_MARKS)


@pytest.mark.parametrize("raw,name", GARBLED_NAMES.items())
def test_old_links_with_garbled_names_still_open_the_product(client, raw, name):
    assert product_title(get(client, product_url(raw))) == name


@pytest.mark.parametrize("typed", [SUPERSTAY, SUPERSTAY_RAW])
def test_recommendations_form_accepts_repaired_or_garbled_name(client, typed):
    page = client.post("/recommendations", data={"prod": typed, "nbr": "5"}).get_data(as_text=True)
    cards = result_cards(page)
    assert len(cards) == 5
    assert SUPERSTAY not in [c["name"] for c in cards]


def test_lipstick_search_shows_superstay_with_a_clean_name(client):
    cards = result_cards(get(client, "/main?query=lipstick"))
    assert cards[0]["name"] == SUPERSTAY
    assert not any(mark in c["alt"] + c["title"] for c in cards for mark in GARBLE_MARKS)


@pytest.mark.parametrize("query,fragment,max_rank", [
    ("Dry-Lock briefs", "Classic Briefs with Dry-Lock Technology", 2),  # was 15th
    ("DermaDry protective underwear", "DermaDry™", 1),  # was 2nd
])
def test_search_matches_words_that_were_garbled(client, query, fragment, max_rank):
    names = [c["name"] for c in result_cards(get(client, "/main", query_string={"query": query}))]
    assert any(fragment in n for n in names[:max_rank]), names[:5]


def test_only_broken_names_change(app_module, tsv):
    # Every name on the site is the TSV name, except the 79 with &quot and the
    # 4 garbled ones; names with real accents, dashes or ™ stay as they are.
    site = app_module.train_data["Name"].tolist()
    assert site == tsv["display_name"].tolist()
    changed = {raw for raw, shown in zip(tsv["Product Name"], site) if raw != shown}
    assert len(changed) == 79 + 4
    untouched_non_ascii = [n for n in site if not n.isascii() and n in set(tsv["Product Name"])]
    assert len(untouched_non_ascii) > 50


# --- search and related products -----------------------------------------------
# Expected results were taken from the original app before any of these
# changes; update them if search or recommendations are changed on purpose.

def test_search_ranking_unchanged(client):
    names = [c["name"] for c in result_cards(get(client, "/main?query=shampoo"))]
    assert len(names) == 20
    assert names[:5] == [
        "Volumizing Therapy Shampoo by Biosilk for Unisex - 7 oz Shampoo",
        "Taliah Waajid Total Body Black Earth Shampoo, 8 oz",
        "Suave Professionals Shampoo Color Protection 12.6oz",
        "Aveda Pure Abundance Volumizing Shampoo, 8.5 oz",
        "White Rain Moisturizing Shampoo, Apple Blossom, 15 Oz",
    ]


def test_search_with_no_match(client):
    page = get(client, "/main?query=qwxzvbnm")
    assert "No relevant products found" in page
    assert result_cards(page) == []


def test_related_products_unchanged(client):
    names = [c["name"] for c in related_cards(get(client, product_url(OPI)))]
    assert names == [
        "OPI GelColor Soak-Off Gel Lacquer Mini GCN51B / 0.25oz - Let Me Bayou a Drink",
        "OPI Nail Lacquer, Teal The Cows Come Home, 0.5 Oz",
        "OPI Nail GelColor + Infinite Shine Polish - SENDING YOU HOLIDAY HUGS J08 + J47",
        "OPI Infinite Shine 2 Polish - ISL P33 - Alpaca My Bags Nail Polish, Women, 0.5oz",
        "OPI Nail Lacquer Polish .5oz/15mL - Ski Slope Sweetie E15",
        "OPI Nail Polish, Machu Peach-U, 0.5 Fl Oz",
        "OPI Nail Lacquer Nail Polish, Go with the Lava Flow",
        "OPI Nail Polish, Nothin Mousie, 0.5 Fl Oz",
    ]


def test_recommendations_form(client):
    resp = client.post("/recommendations", data={"prod": KOKIE, "nbr": "5"})
    cards = result_cards(resp.get_data(as_text=True))
    assert resp.status_code == 200 and len(cards) == 5
    assert KOKIE not in [c["name"] for c in cards]
    assert all("Lip" in c["name"] for c in cards)
