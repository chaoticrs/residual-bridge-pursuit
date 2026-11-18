"""
Synthetic test of Residual-Bridge Pursuit (RBP) vs dense, Rocchio-PRF and a
G3RAG-style geometric-gain diffusion (no LLM filter).

Model (linear concept superposition, i.e. the "linear representation hypothesis"):
  every entity / relation is a random unit vector in R^d.
  fact doc (h, r, t) = normalize(e_h + e_r + e_t + noise)
  2-hop question "r2 of (r1 of h)" = normalize(e_h + e_r1 + e_r2 + noise)
  gold chain: d1 = (h, r1, b)  and  d2 = (b, r2, ans)   -- d2 never mentions h.
Distractors: many other facts about h (similarity trap) and many facts with r2.

Run: python experiments/rbp_synthetic.py
"""
import numpy as np

rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def build(d=384, n_ent=3000, n_rel=40, facts_per_ent=12, noise=0.35):
    E = unit(rng.standard_normal((n_ent, d)))
    R = unit(rng.standard_normal((n_rel, d)))
    docs, triples = [], []
    for h in range(n_ent):
        rels = rng.choice(n_rel, facts_per_ent, replace=False)
        for r in rels:
            t = rng.integers(n_ent)
            v = E[h] + R[r] + E[t] + noise * rng.standard_normal(d) / np.sqrt(d) * np.sqrt(3)
            docs.append(v)
            triples.append((h, r, t))
    return E, R, unit(np.array(docs)), triples


def make_questions(E, R, triples, n_q=400, noise=0.35):
    d = E.shape[1]
    by_head = {}
    for i, (h, r, t) in enumerate(triples):
        by_head.setdefault(h, []).append(i)
    qs = []
    while len(qs) < n_q:
        i1 = rng.integers(len(triples))
        h, r1, b = triples[i1]
        if b not in by_head:
            continue
        i2 = by_head[b][rng.integers(len(by_head[b]))]
        _, r2, _ = triples[i2]
        if r2 == r1:
            continue
        q = E[h] + R[r1] + R[r2] + noise * rng.standard_normal(d) / np.sqrt(d) * np.sqrt(3)
        qs.append((unit(q), {i1, i2}))
    return qs


# ---------------------------------------------------------------- retrievers
def dense(D, q, k):
    return list(np.argsort(-D @ q)[:k])


def rocchio(D, q, k, beta=0.7):
    d1 = int(np.argmax(D @ q))
    q2 = unit(q + beta * D[d1])
    s = D @ q2
    s[d1] = np.inf
    return list(np.argsort(-s)[:k])


def rbp(D, q, k, beta=1.0, beam=3):
    """Residual-Bridge Pursuit.
    hop-2 query = (part of q not explained by d1) + beta * (part of d1 not in q).
    The second term carries the bridge entity, the first carries the unanswered relation."""
    s0 = D @ q
    cands = np.argsort(-s0)[:beam]
    best, best_score = None, -np.inf
    for d1 in cands:
        v1 = D[d1]
        resid = q - (q @ v1) * v1          # unexplained query
        novel = v1 - (v1 @ q) * q          # new info introduced by d1 (bridge)
        q2 = unit(unit(resid) + beta * unit(novel))
        s2 = D @ q2
        s2[d1] = -np.inf
        d2 = int(np.argmax(s2))
        chain_score = s0[d1] + s2[d2]
        if chain_score > best_score:
            best_score, best = chain_score, (d1, q2)
    d1, q2 = best
    s = D @ q2
    s[d1] = np.inf
    out = list(np.argsort(-s)[:k])
    return out


def omp(A, x, n_atoms=3):
    """Orthogonal matching pursuit of x over dictionary rows A. Returns atom ids."""
    sel, r = [], x.copy()
    for _ in range(n_atoms):
        j = int(np.argmax(np.abs(A @ r)))
        sel.append(j)
        S = A[sel].T
        w, *_ = np.linalg.lstsq(S, x, rcond=None)
        r = x - S @ w
    return set(sel)


def rbp_atoms(D, A, q, k, beam=3, n_atoms=3):
    """Residual-Bridge Pursuit with a concept dictionary A.
    atoms(q) - atoms(d1)  = what the question still asks (unanswered relation)
    atoms(d1) - atoms(q)  = what d1 newly introduced (bridge entity, + its tail)
    hop-2 score = soft-AND (min) over the q-only atom and the best d1-only atom."""
    s0 = D @ q
    aq = omp(A, q, n_atoms)
    best = None
    for d1 in np.argsort(-s0)[:beam]:
        a1 = omp(A, D[d1], n_atoms)
        need_q = list(aq - a1)
        new_1 = list(a1 - aq)
        if not need_q or not new_1:
            continue
        Sq = (D @ A[need_q].T).min(1)            # must cover every unanswered q atom
        Sb = (D @ A[new_1].T).max(1)             # must touch at least one new d1 atom
        s2 = np.minimum(Sq, Sb)
        s2[d1] = -np.inf
        score = s0[d1] + s2.max()
        if best is None or score > best[0]:
            best = (score, d1, s2)
    if best is None:
        return dense(D, q, k)
    _, d1, s2 = best
    s2 = s2.copy(); s2[d1] = np.inf
    return list(np.argsort(-s2)[:k])


def g3_graph(D, lo=0.1, hi=0.7, lam=0.4, knn=50):
    """G3RAG-style sparse graph: w_ij = x*sqrt(1-x^2) * exp(-lam*rho_j), x in [lo,hi]."""
    S = D @ D.T
    rho = S.mean(1)
    np.fill_diagonal(S, 0)
    W = np.where((S >= lo) & (S <= hi), S * np.sqrt(np.clip(1 - S**2, 0, 1)), 0.0)
    W = W * np.exp(-lam * rho)[None, :]
    # keep top-knn per row to stay sparse
    idx = np.argpartition(-W, knn, axis=1)[:, :knn]
    Wk = np.zeros_like(W)
    rows = np.arange(W.shape[0])[:, None]
    Wk[rows, idx] = W[rows, idx]
    Wk = Wk / (Wk.sum(1, keepdims=True) + 1e-12)
    return Wk


def g3(D, T, q, k, s=2):
    seeds = np.argsort(-D @ q)[:s]
    p = np.zeros(D.shape[0])
    p[seeds] = 1.0 / s
    p2 = p @ T
    p2[seeds] = np.inf
    return list(np.argsort(-p2)[:k])


def evaluate(name, fn, qs, k=5):
    full, rec = 0, 0.0
    for q, gold in qs:
        got = set(fn(q, k))
        hit = len(gold & got)
        rec += hit / len(gold)
        full += hit == len(gold)
    print(f"{name:28s} Recall@{k} {rec/len(qs):.3f}   both-docs@{k} {full/len(qs):.3f}")


if __name__ == "__main__":
    for noise in (0.35, 0.8):
        E, R, D, triples = build(noise=noise, n_ent=800)
        qs = make_questions(E, R, triples, noise=noise)
        print(f"\n== corpus {D.shape[0]} docs, d={D.shape[1]}, noise={noise}, {len(qs)} two-hop questions")
        T = g3_graph(D)
        evaluate("dense top-k", lambda q, k: dense(D, q, k), qs)
        evaluate("Rocchio PRF (q + d1)", lambda q, k: rocchio(D, q, k), qs)
        evaluate("G3-style diffusion, no LLM", lambda q, k: g3(D, T, q, k), qs)
        evaluate("RBP linear (no dict)", lambda q, k: rbp(D, q, k), qs)
        A = np.vstack([E, R])                      # oracle concept dictionary
        evaluate("RBP-atoms, oracle dict", lambda q, k: rbp_atoms(D, A, q, k), qs)
        # realistic: dictionary atoms are noisy (like embedding an NER span alone)
        An = unit(A + 0.5 * rng.standard_normal(A.shape) / np.sqrt(A.shape[1]))
        evaluate("RBP-atoms, noisy dict", lambda q, k: rbp_atoms(D, An, q, k), qs)
