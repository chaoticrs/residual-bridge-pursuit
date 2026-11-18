# Phase 1: Residual-Bridge Pursuit on real multi-hop corpora

Background and results are in the [top-level README](../../README.md). Nothing here calls an LLM.

## Setup (one time, ~15 MB of packages plus model downloads)
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
`sentence-transformers` and `scikit-learn` are already installed. bge-small-en-v1.5 (~130 MB) downloads on first use.

## Run (from this folder)
```bash
python download_data.py          # HippoRAG 2 data -> data/phase1/raw/  (a few MB)
python data.py                   # sanity check: gold-passage mapping per dataset
python build_index.py            # embeddings + spaCy atoms -> data/phase1/index/   (~10-20 min on CPU)
python run_phase1.py --dataset musique --limit 300 --methods dense,rbp   # quick look
python run_phase1.py             # full run -> results/phase1/summary.md
```

## Files
| file | role |
|---|---|
| `data.py` | loads HippoRAG-format questions and corpus; maps gold passages (same rules as HippoRAG 2) |
| `download_data.py` | fetches data from HF `osunlp/HippoRAG_2`, with GitHub raw as fallback |
| `build_index.py` | bge-small embeddings; spaCy atoms (entities, noun chunks, content lemmas); atom embeddings and document frequency |
| `retrievers.py` | dense, BM25, hybrid, Rocchio, linear RBP, G3-style diffusion without LLM, **RBP** |
| `run_phase1.py` | dev/test protocol: grid search on the first 200 usable questions, reports the rest; writes `summary.md`, `results.json`, `<ds>_rbp_trace.jsonl` |

## Things to know
- Atoms are encoded without the bge query prefix, so atom-to-passage cosine is symmetric. Passages use none either; questions use the prefix.
- `rbp_linear` is the method Theorem 1 predicts will fail. It is kept as a control.
- The trace JSONL shows, per test question, the need atoms, the new (bridge) atoms and the passage each hop picked. Read it before tuning anything.
