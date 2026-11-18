"""Build the retrieval index for one dataset (zero LLM tokens):
  * dense passage + question embeddings (bge-small-en-v1.5, CPU is fine)
  * concept atoms per passage / question via spaCy:
        type 0 = named entity, 1 = noun chunk, 2 = content word (lemma)
  * atom vocabulary embeddings + document frequencies
Output: data/phase1/index/<dataset>.npz (+ <dataset>_vocab.json)

Usage: python build_index.py --dataset musique   (or --dataset all)
"""
import argparse
import json
import time

import numpy as np

from data import DATA, DATASETS, load_corpus, load_questions, passage_string

ENCODER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX = DATA / "index"
WORD_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
EDGE_SKIP_POS = {"DET", "PRON", "PUNCT", "CCONJ", "ADP", "PART", "SPACE"}


def _strip(span):
    """Trim leading/trailing determiners, pronouns, punctuation and stopwords from a span."""
    i, j = 0, len(span)
    while i < j and (span[i].pos_ in EDGE_SKIP_POS or span[i].is_stop):
        i += 1
    while j > i and (span[j - 1].pos_ in EDGE_SKIP_POS or span[j - 1].is_stop):
        j -= 1
    if i >= j:
        return None
    s = span[i:j]
    if not any(t.is_alpha or t.like_num for t in s):
        return None
    txt = s.text.lower().strip()
    return txt if len(txt) >= 2 else None


def extract_atoms(nlp, texts, batch_size=128):
    """Returns list (per text) of dict atom_string -> type (lowest type wins)."""
    out = []
    t0 = time.time()
    for n, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        atoms = {}
        for e in doc.ents:
            s = _strip(e)
            if s:
                atoms[s] = 0
        for c in doc.noun_chunks:
            s = _strip(c)
            if s and s not in atoms:
                atoms[s] = 1
        for t in doc:
            if t.pos_ in WORD_POS and not t.is_stop and (t.is_alpha or t.like_num):
                lem = t.lemma_.lower()
                if len(lem) >= 3 and lem not in atoms:
                    atoms[lem] = 2
        out.append(atoms)
        if n and n % 2000 == 0:
            print(f"    spaCy {n}/{len(texts)}  ({time.time() - t0:.0f}s)")
    return out


def to_csr(per_text, vocab_id):
    indptr, idx = [0], []
    for atoms in per_text:
        idx.extend(sorted(vocab_id[a] for a in atoms))
        indptr.append(len(idx))
    return np.array(indptr, np.int64), np.array(idx, np.int32)


def build(ds, model, nlp):
    titles, texts = load_corpus(ds)
    qs = load_questions(ds, titles, texts)
    passages = [passage_string(t, x) for t, x in zip(titles, texts)]
    questions = [q["question"] for q in qs]
    print(f"[{ds}] {len(passages)} passages, {len(questions)} questions")

    t0 = time.time()
    D = model.encode(passages, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    Q = model.encode([QUERY_PREFIX + q for q in questions], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    print(f"  embeddings done in {time.time() - t0:.0f}s")

    t0 = time.time()
    p_atoms = extract_atoms(nlp, [f"{t}. {x}" for t, x in zip(titles, texts)])
    q_atoms = extract_atoms(nlp, questions)
    print(f"  atoms done in {time.time() - t0:.0f}s")

    vocab, vtype = {}, []
    for atoms in p_atoms + q_atoms:
        for a, ty in atoms.items():
            if a not in vocab:
                vocab[a] = len(vtype)
                vtype.append(ty)
            else:
                vtype[vocab[a]] = min(vtype[vocab[a]], ty)
    strings = [None] * len(vocab)
    for a, i in vocab.items():
        strings[i] = a
    p_ptr, p_idx = to_csr(p_atoms, vocab)
    q_ptr, q_idx = to_csr(q_atoms, vocab)
    df = np.bincount(p_idx, minlength=len(vocab)).astype(np.int32)

    t0 = time.time()
    A = model.encode(strings, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    print(f"  {len(strings)} atom embeddings in {time.time() - t0:.0f}s")

    INDEX.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX / f"{ds}.npz", D=D, Q=Q, A=A, atom_type=np.array(vtype, np.int8), df=df,
             p_ptr=p_ptr, p_idx=p_idx, q_ptr=q_ptr, q_idx=q_idx)
    json.dump({"atoms": strings, "encoder": ENCODER}, open(INDEX / f"{ds}_vocab.json", "w", encoding="utf-8"))
    print(f"  saved {INDEX / (ds + '.npz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    args = ap.parse_args()
    import spacy
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER, device="cpu")
    nlp = spacy.load("en_core_web_sm")
    for ds in DATASETS if args.dataset == "all" else [args.dataset]:
        build(ds, model, nlp)
