import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=16, max_len=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embs.append(pooled.cpu())

    return torch.cat(embs, dim=0).numpy()


def cosine(a, b):
    # a: (N, d), b: (d,)
    return (a @ b) / (np.linalg.norm(b) + 1e-9)


def load_json_outputs(json_file):
    """
    Extract responses from your JSON structure.
    Handles:
      - single-turn items with top-level 'response'
      - multi-turn items with 'turns' list containing 'response'
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        item_id = item.get("id", "")
        category = item.get("category", "")

        # Case 1: top-level response
        if "response" in item:
            rows.append({
                "id": item_id,
                "category": category,
                "turn_index": None,
                "prompt": item.get("prompt", ""),
                "output": item.get("response", "")
            })

        # Case 2: multi-turn responses
        elif "turns" in item and isinstance(item["turns"], list):
            for i, turn in enumerate(item["turns"]):
                rows.append({
                    "id": item_id,
                    "category": category,
                    "turn_index": i,
                    "prompt": turn.get("prompt", ""),
                    "output": turn.get("response", "")
                })

    return pd.DataFrame(rows)


def score_json(json_file, ref_file, out_csv, model_name, device, batch_size, max_len):
    df = load_json_outputs(json_file)

    if df.empty:
        raise ValueError(f"No responses found in {json_file}")

    refs = [line.strip() for line in open(ref_file, "r", encoding="utf-8") if line.strip()]
    if len(refs) < 5:
        raise ValueError("Provide at least ~5 reference pirate texts in ref_file.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # build pirate style target vector
    ref_embs = embed_texts(refs, tokenizer, model, device, batch_size=batch_size, max_len=max_len)
    target = ref_embs.mean(axis=0)
    target = target / (np.linalg.norm(target) + 1e-9)

    outs = df["output"].fillna("").astype(str).tolist()
    out_embs = embed_texts(outs, tokenizer, model, device, batch_size=batch_size, max_len=max_len)

    df["cosine_style_score"] = cosine(out_embs, target)

    df.to_csv(out_csv, index=False)

    print(f"Saved scored CSV -> {out_csv}")
    print(f"File: {json_file}")
    print("Mean cosine_style_score:", float(df["cosine_style_score"].mean()))
    print(df.groupby("category")["cosine_style_score"].mean().sort_values(ascending=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="JSON file with responses")
    ap.add_argument("--ref_file", required=True, help="txt file: one pirate reference text per line")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    score_json(
        json_file=args.input_json,
        ref_file=args.ref_file,
        out_csv=args.out_csv,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()