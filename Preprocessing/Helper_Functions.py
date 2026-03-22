import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse
from functools import lru_cache
from sentence_transformers import SentenceTransformer


# URL FEATURE EXTRACTION
def extract_urls_from_text(text):
    """Return all http/https URLs found in the text."""
    return re.findall(r"https?://[^\s]+", str(text))


def url_features(url):
    """
    Extract numeric/boolean features from a single URL.
    Phishing emails commonly use long URLs, IP addresses, and many special chars.
    """
    url = str(url).strip().rstrip(").,;\"'")
    uses_https = url.lower().startswith("https://")
    try:
        parsed = urlparse(url)
        uses_https = parsed.scheme == "https"
    except ValueError:
        parsed = None

    return {
        "url_length": len(url),  # Long URLs are suspicious
        "url_num_dots": url.count("."),  # Subdomain abuse (e.g. login.paypal.evil.com)
        "url_has_ip": bool(
            re.search(r"\d+\.\d+\.\d+\.\d+", url)
        ),  # IP instead of domain name
        "url_num_special_chars": len(re.findall(r"[-@?=]", url)),  # Obfuscation chars
        "url_uses_https": uses_https,  # Some phishing still uses HTTPS
    }


# Known free/abused TLDs heavily associated with spam and phishing
_SUSPICIOUS_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",  # Freenom free TLDs
    "xyz",
    "top",
    "click",
    "work",
    "loan",  # Common spam TLDs
    "win",
    "download",
    "accountant",
    "racing",
    "info",
    "biz",  # High spam rate
    "ru",
    "cn",  # High phishing origin rate
}


# DOMAIN FEATURE EXTRACTION
def get_domain(email):
    """Extract the domain portion from an email address (e.g. 'user@gmail.com' -> 'gmail.com')."""
    if "@" in str(email):
        return str(email).split("@")[-1].replace(">", "").strip().lower()
    return ""


def domain_features(domain: str, prefix: str = "") -> dict:
    """
    Structural features of a domain string used as training features.
    These generalise to unseen domains unlike raw domain-to-integer encoding.

    Args:
        domain: Domain string e.g. 'gmail.com', 'paypa1-alert.ru'
        prefix: Optional prefix for key names e.g. 'sender_' or 'receiver_'
    """
    if not domain:
        return {
            f"{prefix}domain_length": 0,
            f"{prefix}domain_num_dots": 0,
            f"{prefix}domain_has_numbers": False,
            f"{prefix}domain_has_hyphen": False,
            f"{prefix}domain_suspicious_tld": False,
        }

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    return {
        f"{prefix}domain_length": len(domain),
        f"{prefix}domain_num_dots": domain.count("."),
        f"{prefix}domain_has_numbers": bool(re.search(r"\d", domain)),
        f"{prefix}domain_has_hyphen": "-" in domain,
        f"{prefix}domain_suspicious_tld": tld in _SUSPICIOUS_TLDS,
    }


# BODY TEXT FEATURES
def body_text_features(text):
    """
    Compute simple statistical features from email body text.
    These are cheap to calculate and provide strong baseline signals:
    - Phishing emails tend to be longer with more urgency markers (!, ?)
    - High uppercase count is a common spam indicator
    """
    text = str(text)
    return {
        "body_length": len(text),
        "num_exclamations": text.count("!"),
        "num_questions": text.count("?"),
        "num_uppercase": sum(1 for c in text if c.isupper()),
        "num_urls_in_body": len(extract_urls_from_text(text)),
    }


@lru_cache(maxsize=1)
def get_embed_model():
    """
    Load the sentence-transformers embedding model (cached after first call).
    'all-MiniLM-L6-v2' produces 384-dimensional embeddings — small, fast, and
    accurate enough to capture semantic meaning of email body text.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")
