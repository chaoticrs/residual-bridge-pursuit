"""Phase 1: does Residual-Bridge Pursuit beat LLM-free baselines on real multi-hop corpora?

Protocol
  * questions whose gold passages all map into the corpus are kept
  * first --dev questions = dev split (hyper-parameter grid per method, chosen on --select metric)
  * remaining questions = test split (reported numbers; each method evaluated once with its dev-best config)
Metrics: Recall@2/5/10, FSR@5/10 (FSR = all gold passages retrieved, a.k.a. "both docs").

Kill criterion (fixed before running): RBP < G3-no-LLM on FSR@5 on 2 of 3 datasets.

Usage
  python run_phase1.py                       # all datasets, all methods
  python run_phase1.py --dataset musique --limit 300 --methods dense,rbp   # quick look
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from build_index import INDEX
from data import DATA, DATASETS, load_corpus, load_questions, passage_string
from retrievers import (BM25, RBP, Index, RBPConfig, asks_numeric, dense, g3, g3_transition, rbp_linear,
                        rocchio, rrf)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "phase1"
K = 10
METHODS = ["dense", "bm25", "hybrid", "rocchio", "rbp_linear", "g3", "rbp", "rbp2", "rbpt",
           "rbp2_sum", "rbp2_union", "rbp2_air"]
# AND-vs-union ablation: rbp2 pipeline, only the need/new combination changes (alpha re-tuned on dev for each)
COMBINE = {"rbp2_sum": dict(tnorm="sum"), "rbp2_union": dict(tnorm="union"), "rbp2_air": dict(tnorm="union", air_T=4)}
RBP_METHODS = ("rbp", "rbp2", "rbpt")  # rbp2 = rbp + steps 1-3; rbpt = rbp2 + step 4 (tree, soft-gain stop)


def metrics(ranked, gold):
    g = set(gold)
    out = {}
    for k in (2, 5, 10):
        hit = len(g & set(ranked[:k]))
        out[f"R@{k}"] = hit / len(g)
        if k in (5, 10):
            out[f"FSR@{k}"] = float(hit == len(g))
    return out


def evaluate(fn, qs, by_hops=False):
    acc, cnt, per_q = {}, {}, []
    for q in qs:
        h = len(q["gold"])
        mq = metrics(fn(q), q["gold"])
        per_q.append(mq["FSR@5"])
        for m, v in mq.items():
            keys = [m] + ([f"{m}|{h}hop"] if by_hops and m.startswith("FSR") else [])
            for key in keys:
                acc[key] = acc.get(key, 0.0) + v
                cnt[key] = cnt.get(key, 0) + 1
    out = {m: round(100 * v / cnt[m], 2) for m, v in acc.items()}
    if by_hops:
        out["_fsr5_per_q"] = per_q  # paired tests; dropped from summary tables
    return out


def grid(space):
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, vals))


def make_runner(name, params, ix, bm25, texts_q, g3_cache, title_id=None, base_cache=None):
    """Return fn(question_dict) -> ranked list for a method + params."""
    if name == "dense":
        return lambda q: dense(ix, ix.Q[q["qi"]], K)
    if name == "bm25":
        return lambda q: list(np.argsort(-bm25.scores(texts_q[q["qi"]]))[:K])
    if name == "hybrid":
        return lambda q: rrf(ix.D @ ix.Q[q["qi"]], bm25.scores(texts_q[q["qi"]]))[:K]
    if name == "rocchio":
        return lambda q: rocchio(ix, ix.Q[q["qi"]], K, beta=params["beta"])
    if name == "rbp_linear":
        return lambda q: rbp_linear(ix, ix.Q[q["qi"]], K, beta=params["beta"])
    if name == "g3":
        key = params["lam"]
        if key not in g3_cache:
            g3_cache[key] = g3_transition(ix.D, lam=key)
        T = g3_cache[key]
        return lambda q: g3(ix, T, ix.Q[q["qi"]], K, s=params["s"])
    if name in RBP_METHODS or name in COMBINE:
        r = RBP(ix, RBPConfig(**params), title_id=title_id)
        return lambda q: r.rank(q["qi"], ix.Q[q["qi"]], K, numeric_q=q["numeric"], base_cache=base_cache)
    raise ValueError(name)


def rbpt_space(ds, report):
    """Step 4 grid on top of rbp2's dev-chosen config (its 2-doc base is shared and cached across configs)."""
    if "rbp2" not in report:
        raise SystemExit("rbpt needs rbp2 results first (run with --methods rbp2,rbpt)")
    base = {k: [v] for k, v in report["rbp2"]["params"].items()}
    base["max_hops"] = [2]
    return {**base, "lam": [0.01, 0.02, 0.035, 0.05, 0.08], "cov_lo": [0.4, 0.6],
            "max_docs": [3] if ds == "hotpotqa" else [4]}


def spaces(ds):
    return {
        "dense": {}, "bm25": {}, "hybrid": {},
        "rocchio": {"beta": [0.3, 0.7, 1.0]},
        "rbp_linear": {"beta": [0.5, 1.0]},
        "g3": {"lam": [0.1, 0.4, 0.7], "s": [2, 3]},
        "rbp": {"tnorm": ["min", "product"], "need_agg": ["mean", "min"], "alpha": [0.0, 0.2, 0.4],
                "w_span": [0.5, 0.8], "max_hops": [2, 3] if ds == "musique" else [2]},
        # tnorm / need_agg / w_span fixed to the value the rbp dev grid chose on all 3 datasets
        "rbp2": {"tnorm": ["product"], "need_agg": ["mean"], "w_span": [0.5], "alpha": [0.2, 0.4],
                 "need_max_df": [0.05], "need_lemma_max_df": [0.005, 0.02],
                 "anchor_m": [2], "seed_w": [0.3, 0.6],
                 "drop_numeric_new": [True], "same_title_pen": [0.5],
                 "max_hops": [2, 3] if ds == "musique" else [2]},
    }


def run_dataset(ds, args, done=None, save=lambda r: None):
    titles, texts = load_corpus(ds)
    qs_all = load_questions(ds, titles, texts)
    for i, q in enumerate(qs_all):
        q["qi"] = i
        q["numeric"] = asks_numeric(q["question"])
    tid = {t: j for j, t in enumerate(dict.fromkeys(titles))}
    title_id = np.array([tid[t] for t in titles])
    qs = [q for q in qs_all if q["complete"]]
    if args.limit:
        qs = qs[:args.limit]
    dev, test = qs[:args.dev], qs[args.dev:]
    print(f"\n=== {ds}: {len(qs_all)} questions, {len(qs)} usable; dev {len(dev)} / test {len(test)}")

    ix = Index(INDEX / f"{ds}.npz")
    texts_q = [q["question"] for q in qs_all]
    methods = args.methods.split(",")
    bm25 = BM25([passage_string(t, x) for t, x in zip(titles, texts)]) if {"bm25", "hybrid"} & set(methods) else None
    g3_cache, report, base_caches = {}, dict(done or {}), {}

    for name in methods:
        if name in report:
            print(f"  {name:11s} (resumed) {report[name]['test']}")
            continue
        t0 = time.time()
        best = (None, {})
        if name == "rbpt":
            space = rbpt_space(ds, report)
        elif name in COMBINE:
            space = {**{k: [v] for k, v in report["rbp2"]["params"].items()},
                     **{k: [v] for k, v in COMBINE[name].items()}, "alpha": [0.2, 0.4]}
        else:
            space = spaces(ds)[name]
        configs = list(grid(space))
        bc = base_caches.setdefault(name, {}) if name == "rbpt" else None  # 2-doc base is lam-independent
        for params in configs:
            score = evaluate(make_runner(name, params, ix, bm25, texts_q, g3_cache, title_id, bc), dev)[args.select] if len(configs) > 1 else 0
            if best[0] is None or score > best[0]:
                best = (score, params)
        params = best[1]
        res = evaluate(make_runner(name, params, ix, bm25, texts_q, g3_cache, title_id, bc), test, by_hops=True)
        per_q = res.pop("_fsr5_per_q", None)
        report[name] = dict(params=params, dev_select=best[0], test=res, seconds=round(time.time() - t0, 1),
                            test_fsr5=per_q)
        print(f"  {name:11s} {res}  params={params}  ({time.time() - t0:.0f}s, {len(configs)} configs)")
        save(report)

    vocab = json.load(open(INDEX / f"{ds}_vocab.json", encoding="utf-8"))["atoms"]
    for name in [m for m in RBP_METHODS if m in methods and m in report]:  # error-analysis trace on test
        path = RESULTS / f"{ds}_{name}_trace{f'_limit{args.limit}' if args.limit else ''}.jsonl"
        if path.exists():
            continue
        r = RBP(ix, RBPConfig(**report[name]["params"]), title_id=title_id)
        with open(path, "w", encoding="utf-8") as f:
            for q in test:
                ranked = r.rank(q["qi"], ix.Q[q["qi"]], K, record=True, numeric_q=q["numeric"],
                                base_cache=base_caches.get(name))
                rec = r.c.log[-1]
                f.write(json.dumps(dict(
                    question=q["question"], n_hops=len(q["gold"]), gold=[titles[g] for g in q["gold"]],
                    seeds=[titles[i] for i in rec["seeds"]],
                    chain=[titles[i] for i in rec["chain"]], top5=[titles[i] for i in ranked[:5]],
                    hops=[dict(need=[vocab[a] for a in h["need"]], new=[vocab[a] for a in h["new"]],
                               pick=titles[h["pick"]], **{x: h[x] for x in ("gain", "stopped") if x in h})
                          for h in rec["trace"]],
                    fsr5=metrics(ranked, q["gold"])["FSR@5"]), ensure_ascii=False) + "\n")
    return report


def write_summary(all_reports):
    cols = ["R@2", "R@5", "R@10", "FSR@5", "FSR@10"]
    lines = ["# Phase 1 results (test split)", ""]
    for ds, rep in all_reports.items():
        lines += [f"## {ds}", "", "| method | " + " | ".join(cols) + " | params |",
                  "|---|" + "---|" * len(cols) + "---|"]
        for m, r in rep.items():
            lines.append(f"| {m} | " + " | ".join(f"{r['test'].get(c, 0):.1f}" for c in cols) + f" | `{r['params']}` |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="cap usable questions (smoke tests)")
    ap.add_argument("--select", default="FSR@5")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("results.json" if not args.limit else f"results_limit{args.limit}.json")
    reports = json.load(open(out)) if out.exists() else {}
    for ds in DATASETS if args.dataset == "all" else [args.dataset]:
        if not (INDEX / f"{ds}.npz").exists():
            print(f"{ds}: no index, run build_index.py --dataset {ds}")
            continue
        def save(rep, ds=ds):
            reports[ds] = rep
            json.dump(reports, open(out, "w"), indent=1)
        save(run_dataset(ds, args, reports.get(ds), save))
    write_summary(reports)
    print(f"\nwrote {RESULTS / 'summary.md'}")
