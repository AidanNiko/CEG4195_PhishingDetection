from Preprocessing.Helper_Functions import (
    extract_urls_from_text,
    url_features,
    get_domain,
    domain_features,
    body_text_features,
    get_embed_model,
)
from datasets import load_dataset
import pandas as pd
import re
import numpy as np


# EXTRACT BODY FUNCTION
def extract_body(message: str) -> str:
    """Extract main body text from Enron raw message."""
    lines = str(message).splitlines()
    body_lines = []
    skip = False
    for line in lines:
        line = line.strip()
        # Skip headers
        if re.match(
            r"^(From|To|Date|Subject|X-|Message-ID|Cc|Bcc|Mime-Version|Content-Type|Sent|Received):",
            line,
        ):
            continue
        # Skip forwarded blocks
        if line.startswith("----------------") or "Forwarded by" in line:
            skip = True
            continue
        if skip and not line:
            skip = False
            continue
        if not skip:
            body_lines.append(line)
    return " ".join(body_lines).strip()


# ENRON CLEANING
def clean_enron(sample_size: int = None, path: str = None) -> pd.DataFrame:
    """
    Load the Enron email dataset from HuggingFace and extract features.
    All Enron emails are labelled 0 (legitimate) — this corpus is the main
    source of non-phishing training examples.
    """
    embed_model = get_embed_model()
    dataset_name = path if path else "SuccessfulCrab/enron"
    train_ds = load_dataset(dataset_name, split="train")
    if sample_size:
        sample_size = min(sample_size, len(train_ds))
        # Shuffle before selecting so we don't always take the same emails
        train_ds = train_ds.shuffle(seed=42).select(range(sample_size))
    df_raw = train_ds.to_pandas()

    feature_rows = []
    body_texts = []

    for _, row in df_raw.iterrows():
        message = row.get("message", "")
        # Strip email headers and forwarded blocks, leaving only the body
        body = extract_body(message)

        # The Enron dataset stores extracted entities as a list of dicts.
        entities = row.get("extracted_entities", [])
        if isinstance(entities, str):
            import ast

            entities = ast.literal_eval(entities)
        if isinstance(entities, np.ndarray):
            entities = entities.tolist()
        if not isinstance(entities, list):
            entities = []

        email_entities = [ent for ent in entities if ent.get("type") == "email"]
        sender_email = email_entities[0]["text"] if len(email_entities) > 0 else ""
        receiver_email = email_entities[1]["text"] if len(email_entities) > 1 else ""

        sender_domain = get_domain(sender_email)
        receiver_domain = get_domain(receiver_email)

        # URL features — only the first URL is used; most legit Enron emails have none
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

        # Statistical body features
        body_feat = body_text_features(body)
        body_texts.append(body)

        # Combine all features into a flat dict — will become one row in the DataFrame
        features = {
            **url_feat,
            **body_feat,
            **domain_features(sender_domain, prefix="sender_"),
            **domain_features(receiver_domain, prefix="receiver_"),
            "domains_match": (sender_domain == receiver_domain and sender_domain != ""),
            "body_text": body,
            "label": 0,  # Enron = legit
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
