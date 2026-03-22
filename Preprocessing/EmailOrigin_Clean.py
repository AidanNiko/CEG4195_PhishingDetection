import email as email_lib
import re
from pathlib import Path

import pandas as pd

from Preprocessing.Helper_Functions import (
    extract_urls_from_text,
    url_features,
    get_domain,
    domain_features,
    body_text_features,
    get_embed_model,
)


def _parse_raw_email(raw: str) -> tuple[str, str, str]:
    """
    Parse a raw RFC-2822 email string.
    Returns (sender_email, receiver_email, body_text).
    """
    msg = email_lib.message_from_string(raw)

    sender_email = msg.get("From", "")
    receiver_email = msg.get("To", "")

    # Extract plain text body, falling back to HTML (stripped)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                body = part.get_payload(decode=True) or b""
                body = body.decode(errors="replace")
                break
            if ct == "text/html" and not body:
                html = part.get_payload(decode=True) or b""
                body = _strip_html(html.decode(errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            raw_body = payload.decode(errors="replace")
            if msg.get_content_type() == "text/html":
                body = _strip_html(raw_body)
            else:
                body = raw_body

    return sender_email, receiver_email, body.strip()


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_email_origin(
    path: str = None, sample_size: int = None, row_label: int = 1
) -> pd.DataFrame:
    """
    Load email_origin.csv and extract features.

    Args:
        path: Path to email_origin.csv. Defaults to Datasets/email_origin.csv.
        sample_size: Max number of rows to use.
        row_label: Which rows to keep by CSV label column (1=spam, 0=ham/legit).
                   The same value is assigned as the output label.

    Returns:
        DataFrame with the same feature schema as the other cleaners.
    """
    embed_model = get_embed_model()

    dataset_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[1] / "Datasets" / "email_origin.csv"
    )

    # Read only the rows matching row_label — avoids loading 75k rows into memory at once
    chunks = []
    for chunk in pd.read_csv(dataset_path, chunksize=5000):
        chunks.append(chunk[chunk["label"] == row_label])
    filtered_df = pd.concat(chunks, ignore_index=True)

    # Remove extremely short emails (< 100 chars) — they are mostly obfuscated
    # one-liners or encoding artefacts that skew the model toward flagging all
    # short emails as spam, causing false positives on brief legitimate emails.
    filtered_df = filtered_df[filtered_df["origin"].str.len() >= 100].reset_index(
        drop=True
    )

    if sample_size:
        sample_size = min(sample_size, len(filtered_df))
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)

    label_name = "spam" if row_label == 1 else "ham"
    print(f"Email Origin: using {len(filtered_df)} {label_name} rows")

    feature_rows = []
    body_texts = []

    for _, row in filtered_df.iterrows():
        raw = str(row.get("origin", ""))
        sender_email, receiver_email, body = _parse_raw_email(raw)

        sender_domain = get_domain(sender_email)
        receiver_domain = get_domain(receiver_email)

        urls = extract_urls_from_text(body)
        url_feat = (
            url_features(urls[0])
            if urls
            else {
                "url_length": 0,
                "url_num_dots": 0,
                "url_has_ip": False,
                "url_num_special_chars": 0,
                "url_uses_https": False,
            }
        )

        body_feat = body_text_features(body)
        body_texts.append(body)

        features = {
            **url_feat,
            **body_feat,
            **domain_features(sender_domain, prefix="sender_"),
            **domain_features(receiver_domain, prefix="receiver_"),
            "domains_match": (sender_domain == receiver_domain and sender_domain != ""),
            "body_text": body,
            "label": row_label,
        }
        feature_rows.append(features)

    body_embeddings = embed_model.encode(
        body_texts,
        batch_size=256,
        show_progress_bar=True,
    )
    for features, embedding in zip(feature_rows, body_embeddings):
        for i, val in enumerate(embedding):
            features[f"embed_{i}"] = val

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    df = clean_email_origin(sample_size=10)
    non_embed = [c for c in df.columns if not c.startswith("embed_")]
    print("Shape:", df.shape)
    print("Label counts:", df["label"].value_counts().to_dict())
    print("\nSample (non-embed):")
    print(df[non_embed].head(3).to_string())
