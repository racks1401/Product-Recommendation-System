"""Repair text that was garbled by an encoding mix-up in the source data.

UTF-8 text in the TSV was at some point decoded as Windows-1252, so e.g.
"crème" appears as "crÃ¨me" and "™" as "â„¢". Used by build_data.py for
descriptions and by app.py for product names; kept separate so the app
does not need to import build_data (and spaCy) just to start.
"""

import re

import ftfy

# Characters that can follow "Ã" when a two-byte UTF-8 letter has been
# mis-decoded as Windows-1252. A lone "Ã" not followed by one of these lost
# its second byte. In this dataset all such cases are "é" (Bioré, Curél,
# Ombré, décor), so restore that before ftfy would guess differently.
_UTF8_CONTINUATION = "".join(
    bytes([b]).decode("cp1252", errors="ignore") for b in range(0x80, 0xC0)
)
LONE_A_TILDE = re.compile("Ã(?![" + re.escape(_UTF8_CONTINUATION) + "])")


def repair_encoding(text):
    """Undo the mis-decoding, e.g. "crÃ¨me" -> "crème", "â€™" -> "’"."""
    text = LONE_A_TILDE.sub("é", text)
    text = ftfy.fix_encoding(text)
    # ftfy leaves rows that mix good and bad text alone; the only sequence
    # it leaves behind in this dataset is a mangled trademark sign.
    text = text.replace("â„¢", "™")
    text = text.replace("Â", "")  # leftover mojibake, never part of a word here
    return text
