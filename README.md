# Residual-Bridge Pursuit: LLM-free multi-hop retrieval

Multi-hop questions need a **bridge document**: one about something the question never mentions. This repo contains:

- **An argument.** Under the linear-concept model of embeddings, any retriever that ranks only by similarity to the question and the first document can't separate the bridge from irrelevant documents about the same entity. That covers dense top-k, pseudo-relevance feedback, and graph diffusion over cosine similarity.
- **A method, Residual-Bridge Pursuit (RBP).** It builds on AIR's need/new idea (Yadav et al., ACL 2020) and scores documents separately against:
  - *what the question still needs*;
  - *what the first document newly introduced*.
- **A full evaluation** on MuSiQue, 2WikiMultiHopQA and HotpotQA, with no LLM anywhere and everything on CPU.

## Results

FSR@5 = the share of questions with **all** gold passages in the top 5. Each dataset has 800 held-out test questions; settings were tuned on a separate 200. The encoder is bge-small-en-v1.5.

| FSR@5 (test) | Dense | Dense + BM25 | Rocchio PRF | Graph diffusion (no LLM) | **RBP** |
|---|---|---|---|---|---|
| MuSiQue | 17.5 | 18.2 | 16.2 | 14.0 | **30.0** |
| 2WikiMultiHopQA | 40.6 | 41.2 | 43.2 | 36.8 | **56.9** |
| HotpotQA | 73.8 | 71.4 | 75.1 | 68.4 | **86.3** |

- **Paired against dense retrieval:** +12.5, +16.3 and +12.5 points, McNemar p < 1e-17 on every dataset.
- **Ablation:** scoring the two concept sets separately beats AIR's pooled-query rule by +6.0 (MuSiQue) and +6.6 (HotpotQA) points, and ties on 2Wiki. Whether the two scores are combined by product or average makes no real difference.
- **Limitation:** 3- and 4-hop chains are not solved. The traces show the bottleneck is picking the next document (the third pick is correct only ~25% of the time), not deciding when to stop.

All numbers are in [results/phase1/results.json](results/phase1/results.json) and [results/phase1/summary.md](results/phase1/summary.md). The `*_trace.jsonl` files hold per-question traces: need and new concepts, and the passage picked at each hop.

## Reproduce (CPU, about 2–3 hours end to end)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd experiments/phase1
python download_data.py      # HippoRAG 2 benchmark subsets -> data/phase1/raw (~40 MB)
python build_index.py        # embeddings + spaCy concepts   -> data/phase1/index (~670 MB, ~20 min)
python run_phase1.py         # all methods, dev/test protocol -> results/phase1/
python blog_figures.py       # figures + paired statistics   -> blog/figures/
```

`run_phase1.py` resumes from `results/phase1/results.json`: methods already finished are skipped. Delete that file to recompute everything.

## Layout

| Path | What |
|---|---|
| `experiments/rbp_synthetic.py` | Toy model that isolates the mechanism (synthetic embeddings) |
| `experiments/phase1/data.py` | Loads questions and corpora; maps gold passages (same rules as HippoRAG 2) |
| `experiments/phase1/build_index.py` | bge-small embeddings; spaCy concepts (entities, noun phrases, content words) |
| `experiments/phase1/retrievers.py` | Dense, BM25, hybrid, Rocchio, linear residual, graph diffusion, RBP (`rbp`, `rbp2`, tree variant `rbpt`, combination ablations) |
| `experiments/phase1/run_phase1.py` | Dev/test protocol, settings search, per-hop metrics, traces |

## Prior work

The need/new set differences come from **AIR**: Yadav, Bethard & Surdeanu, *Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering*, ACL 2020. The data are the benchmark subsets released with **HippoRAG 2** (Gutiérrez et al.). Related: STEP (Luo & Surdeanu, LREC 2022), GoldEn Retriever (Qi et al., EMNLP 2019), MDR (Xiong et al., ICLR 2021), LIMIT (Weller et al., ICLR 2026), BridgeRAG (2026 preprint).

## License

MIT (see [LICENSE](LICENSE)). The datasets keep their own licenses and are downloaded, not redistributed. The trace files quote question text and passage titles from them under those licenses.
