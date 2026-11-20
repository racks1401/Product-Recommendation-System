"""Build models/clean_data.csv from the Walmart TSV in this repo.

This reproduces the preprocessing in Recommendation_System.ipynb
(the "Data Loading and Preprocessing" and "Data Cleaning and Tags
Creations" sections) without Google Drive or Colab.

Tagging uses spaCy's English tokenizer via ``spacy.blank("en")``.
The notebook loaded ``en_core_web_sm``, but its tag function only reads
``token.text``, which is produced by the tokenizer alone; the model's
tagger/parser/NER never change token text. ``spacy.blank("en")`` ships
with spaCy itself, so no model download is needed.

One addition over the notebook: "Product Price" is kept as a Price column
(USD, left empty where the TSV has no price) so the app can show real
prices. It is not used for tagging.

A second addition: DescriptionText, a readable copy of "Product
Description" for the product page. The notebook overwrites Description
with tags, so the original text would otherwise be lost. See
clean_description() for what is fixed. It is not used for tagging.

Usage:
    python build_data.py
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from text_repair import repair_encoding

BASE_DIR = Path(__file__).resolve().parent
TSV_PATTERN = "marketing_sample_for_walmart_com-walmart_com_product_review_*.tsv"
OUTPUT_PATH = BASE_DIR / "models" / "clean_data.csv"

SOURCE_COLUMNS = [
    "Uniq Id",
    "Product Id",
    "Product Rating",
    "Product Reviews Count",
    "Product Category",
    "Product Brand",
    "Product Name",
    "Product Image Url",
    "Product Description",
    "Product Tags",
    "Product Price",
]

COLUMN_NAME_MAPPING = {
    "Uniq Id": "ID",
    "Product Id": "ProdID",
    "Product Rating": "Rating",
    "Product Reviews Count": "ReviewCount",
    "Product Category": "Category",
    "Product Brand": "Brand",
    "Product Name": "Name",
    "Product Image Url": "ImageURL",
    "Product Description": "Description",
    "Product Tags": "Tags",
    "Product Price": "Price",
}

COLUMNS_TO_EXTRACT_TAGS_FROM = ["Category", "Brand", "Description"]


def find_tsv():
    matches = sorted(BASE_DIR.glob(TSV_PATTERN))
    if not matches:
        sys.exit(f"No TSV matching '{TSV_PATTERN}' found in {BASE_DIR}")
    if len(matches) > 1:
        names = "\n  ".join(m.name for m in matches)
        sys.exit(f"Found more than one matching TSV, expected one:\n  {names}")
    return matches[0]


def extract_tags(nlp, texts):
    """Notebook's clean_and_extract_tags, applied to a sequence of texts.

    Lowercase, tokenize with spaCy, keep alphanumeric tokens that are not
    spaCy English stop words, join with ", ".
    """
    lowered = (text.lower() for text in texts)
    return [
        ", ".join(
            token.text
            for token in doc
            if token.text.isalnum() and token.text not in STOP_WORDS
        )
        for doc in nlp.pipe(lowered, batch_size=256)
    ]


# Website tracking snippets (Matomo, Google Tag Manager) that the source
# appended to some descriptions. In every case the snippet runs from its
# opening marker to the end of the text.
TRACKING_SCRIPT = re.compile(r"\(?function\(w,d,s,l,i\)|var _mtm\b|var _paq\b")

# Placeholder values that are not descriptions.
PLACEHOLDER_DESCRIPTIONS = {"{}", "@generated"}

HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z]+);")


def clean_description(text):
    """Turn a raw "Product Description" into readable display text.

    Returns "" when there is no real description.
    """
    text = TRACKING_SCRIPT.split(text, maxsplit=1)[0]
    text = repair_encoding(text)  # see text_repair.py
    # The source dropped the ";" from &quot, e.g. "Say &quotyes&quot".
    # Other "&X" sequences without ";" are real text ("R&B"), so only
    # &quot is repaired; complete entities are decoded normally.
    text = re.sub(r"&quot;?", '"', text)
    text = HTML_ENTITY.sub(lambda m: html.unescape(m.group(0)), text)
    text = " ".join(text.split())  # collapse whitespace incl. non-breaking
    if text in PLACEHOLDER_DESCRIPTIONS or not re.search(r"[^\W\d_]", text):
        return ""
    return text


def build(tsv_path):
    train_data = pd.read_csv(tsv_path, sep="\t")
    train_data = train_data[SOURCE_COLUMNS]

    train_data = train_data.fillna({
        "Product Rating": 0,
        "Product Reviews Count": 0,
        "Product Category": "",
        "Product Brand": "",
        "Product Description": "",
    })

    train_data = train_data.rename(columns=COLUMN_NAME_MAPPING)

    train_data["ID"] = train_data["ID"].str.extract(r"(\d+)").astype(float)
    train_data["ProdID"] = train_data["ProdID"].str.extract(r"(\d+)").astype(float)

    train_data["DescriptionText"] = train_data["Description"].map(clean_description)

    nlp = spacy.blank("en")
    for column in COLUMNS_TO_EXTRACT_TAGS_FROM:
        train_data[column] = extract_tags(nlp, train_data[column])

    train_data["Tags"] = train_data[COLUMNS_TO_EXTRACT_TAGS_FROM].apply(
        lambda row: ", ".join(row), axis=1
    )
    return train_data


def main(output_path=OUTPUT_PATH):
    output_path = Path(output_path)
    tsv_path = find_tsv()
    print(f"Reading   {tsv_path.name}")
    print(f"spaCy     {spacy.__version__} (tokenizer: spacy.blank('en'))")

    train_data = build(tsv_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".csv.tmp")
    train_data.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)

    shown = (
        output_path.relative_to(BASE_DIR)
        if output_path.is_relative_to(BASE_DIR)
        else output_path
    )
    print(f"Rows      {len(train_data)}")
    print(f"Columns   {', '.join(train_data.columns)}")
    print(f"Wrote     {shown}")


if __name__ == "__main__":
    main()
