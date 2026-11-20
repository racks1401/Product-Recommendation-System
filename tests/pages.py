"""Helpers for reading the app's HTML in tests."""

import html
import re
import urllib.parse

_ONERROR = r"""onerror="this\.onerror=null; this\.src='([^']*)';\""""


def product_url(name):
    return "/product/" + urllib.parse.quote(name, safe="")


def text(fragment):
    """Visible text of an HTML fragment."""
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def get(client, url, **kwargs):
    resp = client.get(url, **kwargs)
    assert resp.status_code == 200, f"{url} returned {resp.status_code}"
    return resp.get_data(as_text=True)


def product_title(page):
    return text(re.search(r"<h2>(.*?)</h2>", page, re.S).group(1))


def product_price(page):
    return text(re.search(r'<div class="price">(.*?)</div>', page, re.S).group(1))


def product_rating(page):
    m = re.search(r'<div class="rating( unrated)?">(.*?)</div>', page, re.S)
    return text(m.group(2))


def product_description(page):
    m = re.search(r'<p class="description( missing)?">(.*?)</p>', page, re.S)
    return text(m.group(2)), bool(m.group(1))


def product_image(page):
    """(src, fallback) of the main product image."""
    m = re.search(
        r'<div class="product-section">\s*<img src="([^"]*)"[^>]*?' + _ONERROR,
        page, re.S,
    )
    return html.unescape(m.group(1)), html.unescape(m.group(2))


def _cards(page, card_class):
    pattern = (
        r'<a href="(/product/[^"]+)">\s*<div class="' + card_class + r'">\s*'
        r'<img src="([^"]*)" alt="([^"]*)"[^>]*?' + _ONERROR
        + r'.*?<h4>(.*?)</h4>\s*(<p[^>]*>.*?</p>)'
    )
    return [
        {
            "href": html.unescape(href),  # the link, exactly as the browser follows it
            "name": urllib.parse.unquote(html.unescape(href)[len("/product/"):]),
            "src": html.unescape(src),
            "alt": html.unescape(alt),
            "title": text(title),  # visible (possibly truncated) name
            "fallback": html.unescape(fallback),
            "p": text(p),
        }
        for href, src, alt, fallback, title, p in re.findall(pattern, page, re.S)
    ]


def result_cards(page):
    """Cards on the search / recommendation results page; 'p' is the price."""
    return _cards(page, "card")


def related_cards(page):
    """Related-product cards on a product page; 'p' is the rating."""
    return _cards(page, "recommend-card")


def all_img_srcs(page):
    return [html.unescape(s) for s in re.findall(r'<img src="([^"]*)"', page)]
