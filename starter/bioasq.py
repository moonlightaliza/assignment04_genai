
"""Part D data: biomedical questions + PubMed abstracts (BioASQ, mini version)."""

from pathlib import Path
from datasets import load_from_disk, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
BIOASQ_DIR = ROOT / "bioasq"


def load_corpus():
    ds = load_from_disk(str(BIOASQ_DIR / "corpus"))

    if isinstance(ds, DatasetDict):
        ds = ds[list(ds.keys())[0]]

    return {
        int(r["id"]): r["passage"]
        for r in ds
        if r["passage"]
    }


def load_questions(split="test", corpus=None):
    ds = load_from_disk(str(BIOASQ_DIR / "qa"))

    if isinstance(ds, DatasetDict):
        ds = ds[split]

    out = []

    for r in ds:
        rel = [int(i) for i in r["relevant_passage_ids"]]

        if corpus is not None:
            rel = [i for i in rel if i in corpus]

        if rel:
            out.append({
                "qid": int(r["id"]),
                "question": r["question"],
                "answer": r["answer"],
                "relevant_ids": rel
            })

    return out
