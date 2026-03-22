import pandas as pd
from pathlib import Path

from Preprocessing.Helper_Functions import (
    extract_urls_from_text,
    url_features,
    get_domain,
    domain_features,
    body_text_features,
    get_embed_model,
)


# NAZARIO CLEANING
def clean_nazario(path: str = None, sample_size: int = None) -> pd.DataFrame:
    """
    Load the Nazario phishing dataset from a local CSV and extract features.
    This is a small (1,565 row) curated corpus of real phishing emails — all label=1.
    It includes sender/receiver addresses so domain features can be extracted.
    """
    embed_model = get_embed_model()
    dataset_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[1] / "Datasets" / "Nazario.csv"
    )
    df = pd.read_csv(dataset_path)
    if sample_size:
        # Cap at actual dataset size to avoid errors
        sample_size = min(sample_size, len(df))
        df = df.sample(n=sample_size, random_state=42)

    feature_rows = []
    body_texts = []

    for _, row in df.iterrows():
        message = row.get("body", "")

        # URL features from the first URL found in the body
        urls = extract_urls_from_text(message)
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

        # Nazario CSV has explicit sender/receiver columns
        sender_domain = get_domain(row.get("sender", ""))
        receiver_domain = get_domain(row.get("receiver", ""))

        body_feat = body_text_features(message)
        body_texts.append(message)

        # Combine all features into a flat dict — will become one row in the DataFrame
        features = {
            **url_feat,
            **body_feat,
            **domain_features(sender_domain, prefix="sender_"),
            **domain_features(receiver_domain, prefix="receiver_"),
            "domains_match": (sender_domain == receiver_domain and sender_domain != ""),
            "body_text": message,
            "label": row["label"],  # Already 0 or 1 in the CSV
        }

        feature_rows.append(features)

    body_embeddings = embed_model.encode(
        body_texts,
        batch_size=256,
        show_progress_bar=False,
    )
    for features, body_embedding in zip(feature_rows, body_embeddings):
        for i, val in enumerate(body_embedding):
            features[f"embed_{i}"] = val

    df_features = pd.DataFrame(feature_rows)
    return df_features
