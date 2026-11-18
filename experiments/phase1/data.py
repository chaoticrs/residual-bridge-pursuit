"""Load HippoRAG-format multi-hop QA data (MuSiQue, 2Wiki, HotpotQA) and map gold passages
to corpus indices. Mirrors HippoRAG 2's get_gold_docs logic (passage string = title + '\\n' + text)."""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phase1"
RAW = DATA / "raw"
DATASETS = ["musique", "2wikimultihopqa", "hotpotqa"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def passage_string(title: str, text: str) -> str:
    return f"{title}\n{text}"


def load_corpus(name):
    raw = json.load(open(RAW / f"{name}_corpus.json", encoding="utf-8"))
    titles = [r.get("title", "") for r in raw]
    texts = [r.get("text") or r.get("paragraph_text") or "" for r in raw]
    return titles, texts


def gold_strings(sample, name):
    """Gold passage strings for one question (same rules as HippoRAG 2)."""
    if "supporting_facts" in sample:  # hotpotqa, 2wiki
        gold_titles = {sf[0] for sf in sample["supporting_facts"]}
        ctx = [c for c in sample["context"] if c[0] in gold_titles]
        joiner = "" if name.startswith("hotpotqa") else " "
        return [c[0] + "\n" + joiner.join(c[1]) for c in ctx]
    if "contexts" in sample:
        return [c["title"] + "\n" + c["text"] for c in sample["contexts"] if c["is_supporting"]]
    out = []
    for p in sample["paragraphs"]:  # musique
        if p.get("is_supporting") is False:
            continue
        out.append(p["title"] + "\n" + (p["text"] if "text" in p else p["paragraph_text"]))
    return out


def _overlap(a, b):
    sa, sb = set(norm(a).split()), set(norm(b).split())
    return len(sa & sb) / (len(sa | sb) or 1)


def load_questions(name, titles, texts):
    samples = json.load(open(RAW / f"{name}.json", encoding="utf-8"))
    full, by_title = {}, {}
    for i, (t, x) in enumerate(zip(titles, texts)):
        full.setdefault(norm(passage_string(t, x)), i)
        by_title.setdefault(norm(t), []).append(i)

    qs = []
    for j, s in enumerate(samples):
        golds = gold_strings(s, name)
        found = set()
        for g in golds:
            i = full.get(norm(g))
            if i is None:  # fallback: same title, best text overlap
                cands = by_title.get(norm(g.split("\n", 1)[0]), [])
                if cands:
                    i = max(cands, key=lambda c: _overlap(texts[c], g))
            if i is not None:
                found.add(i)
        ans = s.get("answer", s.get("possible_answers", ""))
        qs.append(dict(id=s.get("id", s.get("_id", j)), question=s["question"], answer=ans,
                       gold=sorted(found), n_gold=len(set(golds)),
                       complete=len(found) == len(set(golds)) and len(found) > 0))
    return qs


if __name__ == "__main__":  # quick data sanity check (no models needed)
    for ds in DATASETS:
        if not (RAW / f"{ds}.json").exists():
            print(f"{ds}: missing, run download_data.py")
            continue
        T, X = load_corpus(ds)
        Q = load_questions(ds, T, X)
        ok = sum(q["complete"] for q in Q)
        hops = {}
        for q in Q:
            hops[q["n_gold"]] = hops.get(q["n_gold"], 0) + 1
        print(f"{ds}: {len(T)} passages, {len(Q)} questions, {ok} with all gold mapped, gold-count histogram {hops}")
