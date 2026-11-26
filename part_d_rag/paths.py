"""Shared paths for Part D. This file lives in part_d_rag/, the repo root is one level up."""
import os
import sys

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
sys.path.insert(0,os.path.join(ROOT,"starter"))

RES=os.path.join(ROOT,"results")
FIG=os.path.join(RES,"figures")
os.makedirs(FIG,exist_ok=True)

#the index is ~100 MB, keep it out of git (.gitignore blocks *.pt); override with RAG_INDEX
INDEX_PATH=os.environ.get("RAG_INDEX",os.path.join(HERE,"index.pt"))


def load_data():
    """Corpus + test questions, with relevant ids not in the corpus dropped (same as check_starter)."""
    from bioasq import load_corpus,load_questions
    corpus=load_corpus()
    questions=load_questions("test",corpus)
    return corpus,questions