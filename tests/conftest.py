"""Shared fixtures.

The tests never rely on models/clean_data.csv already existing: the data
is built from the TSV into a temporary folder once per test run, and the
app is pointed at that build through the CLEAN_DATA_PATH variable.
"""

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import build_data  # noqa: E402


# The four names garbled by an encoding mix-up in the source, and how they
# should read. Written out by hand so the tests don't reuse the app's repair.
GARBLED_NAMES = {
    "SuperStay 24Ã‚Â Liquid Lipstick":
        "SuperStay 24 Liquid Lipstick",
    "Natureâ€™s Originâ„¢ Aromatherapy Frankincense Essential Oil, 15 ml":
        "Nature’s Origin™ Aromatherapy Frankincense Essential Oil, 15 ml",
    "(4 Pack) Attends Protective Underwear with DermaDryâ„¢ Technology for Adult Incontinence Care, L, Unisex (18 count)":
        "(4 Pack) Attends Protective Underwear with DermaDry™ Technology for Adult Incontinence Care, L, Unisex (18 count)",
    "(2 pack) Attends Classic Briefs with Dry-LockÂ Technology for Adult Incontinence Care, X-Large, Unisex (20 Briefs)":
        "(2 pack) Attends Classic Briefs with Dry-Lock Technology for Adult Incontinence Care, X-Large, Unisex (20 Briefs)",
}


def display_name(raw_name):
    """How a TSV product name should read on the site: the source's broken
    &quot (no ";") becomes a real quote character, and the four garbled
    names read as in GARBLED_NAMES."""
    name = GARBLED_NAMES.get(raw_name, raw_name)
    return name.replace("&quot;", '"').replace("&quot", '"')


@pytest.fixture(scope="session")
def tsv():
    """The raw TSV as strings, for computing expected values independently.

    display_name is the name the site shows and links to; name_key is its
    lowercase form (product pages match names case-insensitively).
    """
    df = pd.read_csv(
        build_data.find_tsv(), sep="\t", dtype=str, keep_default_na=False
    )
    df["display_name"] = df["Product Name"].map(display_name)
    df["name_key"] = df["display_name"].str.lower()
    return df


@pytest.fixture(scope="session")
def tsv_first_row(tsv):
    """First TSV row per (case-insensitive) product name, i.e. the row a
    product page shows."""
    return tsv.drop_duplicates("name_key").set_index("name_key")


@pytest.fixture(scope="session")
def built_csv(tmp_path_factory):
    """Run the real build (build_data.main) into a temporary folder."""
    out = tmp_path_factory.mktemp("build") / "clean_data.csv"
    build_data.main(output_path=out)
    return out


@pytest.fixture(scope="session")
def built(built_csv):
    return pd.read_csv(built_csv, dtype=str, keep_default_na=False)


@pytest.fixture(scope="session")
def app_module(built_csv):
    os.environ["CLEAN_DATA_PATH"] = str(built_csv)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def unique_names(tsv):
    """TSV rows whose product name appears only once (case-insensitive), so
    the expected value for a product page is unambiguous."""
    counts = tsv["name_key"].value_counts()
    return tsv[tsv["name_key"].map(counts) == 1]
