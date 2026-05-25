from __future__ import annotations

import ipaddress
from urllib.parse import urlparse
import pandas as pd


def normalize_url(url: str) -> str:
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url


def is_domain_ip(domain: str) -> int:
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0


def extract_tld(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) < 2:
        return ""
    return parts[-1]


def count_subdomains(domain: str) -> int:
    parts = domain.split(".")

    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def extract_url_features(url: str) -> pd.DataFrame:
    normalized_url = normalize_url(url)
    parsed = urlparse(normalized_url)

    scheme = parsed.scheme
    domain = parsed.netloc.lower()

    domain = domain.split(":")[0]

    path_query = parsed.path + parsed.query
    full_url = normalized_url

    url_length = len(full_url)
    domain_length = len(domain)

    tld = extract_tld(domain)

    no_of_letters = sum(char.isalpha() for char in full_url)
    no_of_digits = sum(char.isdigit() for char in full_url)

    no_of_equals = full_url.count("=")
    no_of_qmark = full_url.count("?")
    no_of_ampersand = full_url.count("&")

    special_chars = [
        char for char in full_url
        if not char.isalnum()
    ]

    no_of_special_chars = len(special_chars)

    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip(domain),
        "TLDLength": len(tld),
        "NoOfSubDomain": count_subdomains(domain),
        "HasObfuscation": 1 if "%" in full_url else 0,
        "NoOfObfuscatedChar": full_url.count("%"),
        "ObfuscationRatio": full_url.count("%") / url_length if url_length > 0 else 0,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": no_of_letters / url_length if url_length > 0 else 0,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": no_of_digits / url_length if url_length > 0 else 0,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_special_chars,
        "SpacialCharRatioInURL": no_of_special_chars / url_length if url_length > 0 else 0,
        "IsHTTPS": 1 if scheme == "https" else 0,
    }

    return pd.DataFrame([features])


if __name__ == "__main__":
    test_url = "https://example.com/login?user=test&id=123"
    df = extract_url_features(test_url)
    print(df.T)