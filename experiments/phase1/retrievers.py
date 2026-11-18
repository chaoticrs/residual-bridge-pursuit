"""Retrievers for Phase 1. All are LLM-free. Each returns a ranked list of passage indices.

Baselines : dense, bm25, hybrid (RRF of dense + bm25), rocchio (q + d1),
            rbp_linear (Gram-Schmidt residual + novel part of d1; Theorem 1 says this is weak),
            g3 (G3RAG-style geometric-gain one-step diffusion, dense seeds, NO LLM filter).
Ours      : rbp (Residual-Bridge Pursuit over spaCy concept atoms, set difference + t-norm AND).
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def topk(x, k):
    k = min(k, len(x))
    idx = np.argpartition(-x, k - 1)[:k]
    return idx[np.argsort(-x[idx])]


def rrf(*scores, k=60, depth=200):
    fused = {}
    for s in scores:
        for r, i in enumerate(topk(s, depth)):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(fused, key=lambda i: -fused[i])


def fill(head, scores, k):
    """Ranking = head (in order) followed by the best remaining docs by `scores`."""
    out, seen = [], set()
    for i in list(head) + list(topk(scores, k + len(head))):
        i = int(i)
        if i not in seen:
            out.append(i)
            seen.add(i)
        if len(out) == k:
            break
    return out


# --------------------------------------------------------------------------- index
class Index:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.D, self.Q, self.A = z["D"], z["Q"], z["A"]
        self.atom_type, self.df = z["atom_type"], z["df"]
        self.p_ptr, self.p_idx, self.q_ptr, self.q_idx = z["p_ptr"], z["p_idx"], z["q_ptr"], z["q_idx"]
        self.n = len(self.D)
        self.idf = np.log((self.n - self.df + 0.5) / (self.df + 0.5) + 1.0).astype(np.float32)
        self.empty_doc = self.p_ptr[1:] == self.p_ptr[:-1]
        self._cache = {}
        vocab = Path(npz_path).with_name(Path(npz_path).stem + "_vocab.json")
        strings = json.load(open(vocab, encoding="utf-8"))["atoms"]
        self.is_num = np.array([any(ch.isdigit() for ch in s) for s in strings], bool)  # dates, amounts, ordinals

    def p_atoms(self, i):
        return self.p_idx[self.p_ptr[i]:self.p_ptr[i + 1]]

    def q_atoms(self, i):
        return self.q_idx[self.q_ptr[i]:self.q_ptr[i + 1]]

    def span_match(self, atom_ids):
        """(len(atom_ids), n): for each atom, max cosine to any atom of each passage. Cached per atom."""
        atom_ids = [int(a) for a in atom_ids]
        if len(self._cache) > 5000:
            self._cache.clear()
        todo = list(dict.fromkeys(a for a in atom_ids if a not in self._cache))
        if todo:
            S = self.A[todo] @ self.A.T                       # (t, vocab)
            G = S[:, self.p_idx]                              # (t, total passage-atom pairs)
            M = np.maximum.reduceat(G, self.p_ptr[:-1].clip(max=len(self.p_idx) - 1), axis=1)
            M[:, self.empty_doc] = -1.0
            for a, row in zip(todo, M):
                self._cache[a] = row.astype(np.float32)
        return np.stack([self._cache[a] for a in atom_ids])


# --------------------------------------------------------------------------- baselines
class BM25:
    def __init__(self, texts, k1=0.9, b=0.4):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words="english")
        X = self.cv.fit_transform(texts).tocsr().astype(np.float32)
        n = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel()
        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        tf = X.data
        X.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / dl.mean())) * idf[X.indices]
        self.W = X.T.tocsr()  # (V, n)

    def scores(self, query):
        v = self.cv.transform([query])
        v.data[:] = 1.0
        return np.asarray((v @ self.W).todense()).ravel()


def dense(ix, qv, k):
    return list(topk(ix.D @ qv, k))


def rocchio(ix, qv, k, beta=0.7):
    s0 = ix.D @ qv
    d1 = int(np.argmax(s0))
    q2 = qv + beta * ix.D[d1]
    return fill([d1], ix.D @ (q2 / np.linalg.norm(q2)), k)


def rbp_linear(ix, qv, k, beta=1.0):
    s0 = ix.D @ qv
    d1 = int(np.argmax(s0))
    v1 = ix.D[d1]
    resid = qv - (qv @ v1) * v1
    novel = v1 - (v1 @ qv) * qv
    q2 = resid / (np.linalg.norm(resid) + 1e-9) + beta * novel / (np.linalg.norm(novel) + 1e-9)
    return fill([d1], ix.D @ q2, k)


def g3_transition(D, lo=0.1, hi=0.7, lam=0.4, knn=50, chunk=2048):
    """Sparse G3RAG graph: w_ij = x*sqrt(1-x^2)*exp(-lam*rho_j), x in [lo,hi]; row-normalised."""
    n = len(D)
    rho = D @ D.mean(0)
    pen = np.exp(-lam * rho).astype(np.float32)
    rows, cols, vals = [], [], []
    for s in range(0, n, chunk):
        S = D[s:s + chunk] @ D.T
        r = np.arange(S.shape[0])
        S[r, s + r] = 0.0
        W = np.where((S >= lo) & (S <= hi), S * np.sqrt(np.clip(1 - S * S, 0, None)), 0.0) * pen[None, :]
        kk = min(knn, n - 1)
        top = np.argpartition(-W, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(W, top, 1)
        rows.append(np.repeat(s + r, kk))
        cols.append(top.ravel())
        vals.append(v.ravel())
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    rs = np.asarray(T.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, 1e-12)) @ T


def g3(ix, T, qv, k, s=2):
    s0 = ix.D @ qv
    seeds = topk(s0, s)
    p = np.zeros(ix.n, np.float32)
    p[seeds] = 1.0 / s
    p2 = np.asarray(T.T @ p).ravel()  # one diffusion step: p @ T
    p2 = p2 + 1e-6 * mm(s0)          # tie-break leftover slots by dense score
    return fill(list(seeds), p2, k)


# --------------------------------------------------------------------------- ours: RBP
@dataclass
class RBPConfig:
    beam: int = 3                  # hop-1 candidates explored
    max_hops: int = 2              # 2 for HotpotQA/2Wiki; MuSiQue has 2-4
    tau: float = 0.85              # cosine at which two atoms count as "the same concept"
    q_types: tuple = (0, 1, 2)     # atom types used for the question's needs
    new_types: tuple = (0, 1)      # bridge atoms must be entities / noun chunks
    max_df_frac: float = 0.02      # bridge atoms that occur in >2% of passages are uninformative
    max_new: int = 12              # keep the most specific (highest idf) new atoms
    w_span: float = 0.7            # atom->doc score: w*span_match + (1-w)*cos(passage, atom)
    need_agg: str = "mean"         # "mean" (idf-weighted) or "min" (strict AND over needs)
    tnorm: str = "min"             # "min" (Goedel), "product", "luk" (Lukasiewicz); ablations: "sum" (OR-ish mean),
                                   # "union" (AIR-style: one idf-weighted alignment query over need U new)
    alpha: float = 0.2             # weight of dense query score mixed into hop scores
    # --- long-chain fixes (defaults = original Phase-1 RBP, so old configs reproduce exactly) ---
    need_max_df: float = 1.0       # step 1: drop need atoms occurring in > this fraction of passages
    need_lemma_max_df: float = 1.0 # step 1: stricter cap for bare content words (type 2: "win", "game")
    anchor_m: int = 0              # step 2: extra hop-1 seeds = best passage for each of the m rarest
                                   #         question entities / noun chunks
    seed_w: float = 0.0            # step 2: hop-1 score = (1-w)*dense + w*idf-weighted need coverage
    drop_numeric_new: bool = False # step 3: no digit-bearing bridge atoms unless the question asks for one
    same_title_pen: float = 1.0    # step 3: multiply hop score of passages sharing a chain passage's title
    lam: float = float("inf")      # step 4: min marginal soft-coverage gain to add a passage (inf = off)
    max_docs: int = 4              # step 4: hard cap on chain/tree size
    cov_lo: float = 0.5            # step 4: atom->passage score where soft coverage starts (1.0 at tau)
    air_T: int = 10 ** 6           # "union" only: add new atoms only if |need| <= air_T (AIR uses T=2..4)
    log: list = field(default_factory=list, repr=False)


def _tnorm(a, b, kind):
    if kind == "min":
        return np.minimum(a, b)
    if kind == "product":
        return a * b
    if kind == "sum":
        return 0.5 * (a + b)
    return np.maximum(0.0, a + b - 1.0)


NUMERIC_Q = re.compile(r"\b(when|year|years|date|century|decade|how many|how much|how old|how long|population)\b", re.I)


def asks_numeric(question):
    return bool(NUMERIC_Q.search(question))


class RBP:
    def __init__(self, ix: Index, cfg: RBPConfig, title_id=None):
        self.ix, self.c = ix, cfg
        self.title_id = title_id  # int per passage; equal ids = same title (MuSiQue has many)

    def _atom_doc(self, atoms):
        span = self.ix.span_match(atoms)
        cos = self.ix.A[atoms] @ self.ix.D.T
        return self.c.w_span * span + (1 - self.c.w_span) * cos

    def _explained(self, atoms, ref_atoms):
        if len(atoms) == 0 or len(ref_atoms) == 0:
            return np.zeros(len(atoms), bool)
        return (self.ix.A[atoms] @ self.ix.A[ref_atoms].T).max(1) >= self.c.tau

    def _need_atoms(self, q_at):
        """Step 1: keep specific question atoms only; generic words never get 'explained' and block stopping."""
        ix, c = self.ix, self.c
        keep = ix.df[q_at] <= max(2, c.need_max_df * ix.n)
        lemma = ix.atom_type[q_at] == 2
        keep &= ~lemma | (ix.df[q_at] <= max(2, c.need_lemma_max_df * ix.n))
        return q_at[keep]

    def _new_atoms(self, doc_atoms, seen_atoms, numeric_q=True):
        ix, c = self.ix, self.c
        a = doc_atoms[np.isin(ix.atom_type[doc_atoms], c.new_types)]
        a = a[ix.df[a] <= max(2, c.max_df_frac * ix.n)]
        if c.drop_numeric_new and not numeric_q:  # step 3: "68 million", "1816", "38th" are not bridges
            a = a[~ix.is_num[a]]
        a = a[~self._explained(a, seen_atoms)]
        return a[np.argsort(-ix.idf[a])][:c.max_new]

    def _hop(self, need, new, s0n):
        c, ix = self.c, self.ix
        if c.tnorm == "union":  # AIR (Yadav et al. 2020) eq. 1 + 5: single query = need (+ new if few needs left)
            q = need if len(need) > c.air_T else np.concatenate([need, new])
            if len(q) == 0:
                return s0n.copy()
            w = ix.idf[q]
            return (1 - c.alpha) * mm((w[:, None] * self._atom_doc(q)).sum(0) / w.sum()) + c.alpha * s0n
        if len(need):
            S = self._atom_doc(need)
            if c.need_agg == "min":
                need_s = S.min(0)
            else:
                w = ix.idf[need]
                need_s = (w[:, None] * S).sum(0) / w.sum()
            need_s = mm(need_s)
        else:  # question fully covered lexically: use the dense query as the need
            need_s = s0n
        new_s = mm(self._atom_doc(new).max(0))
        return (1 - c.alpha) * _tnorm(need_s, new_s, c.tnorm) + c.alpha * s0n

    def _seeds(self, s0, s0n, need0):
        """Step 2: hop-1 candidates and their start scores.
        Candidates = dense top-beam  U  best passage for each of the anchor_m rarest entity/noun-chunk needs
        (the true chain start is usually the most specific thing in the question, e.g. 'Changes Two')."""
        ix, c = self.ix, self.c
        cand = [int(d) for d in topk(s0, c.beam)]
        start = s0n
        if (c.anchor_m or c.seed_w) and len(need0):
            S = self._atom_doc(need0)                                    # (|need|, n), cached per atom
            if c.seed_w:
                w = ix.idf[need0]
                cover = mm((w[:, None] * S).sum(0) / w.sum())
                start = (1 - c.seed_w) * s0n + c.seed_w * cover
            if c.anchor_m:
                ents = np.where(ix.atom_type[need0] <= 1)[0]
                for j in ents[np.argsort(-ix.idf[need0[ents]])][:c.anchor_m]:
                    d = int(np.argmax(S[j]))
                    if d not in cand:
                        cand.append(d)
        return cand, start

    def _base(self, qi, qv, numeric_q):
        """Seeds + 2-hop beam (rbp / rbp2). Returns the state the tree extension starts from, or None."""
        ix, c = self.ix, self.c
        s0 = ix.D @ qv
        s0n = mm(s0)
        q_at = ix.q_atoms(qi)
        q_at = q_at[np.isin(ix.atom_type[q_at], c.q_types)]
        if len(q_at) == 0:
            return dict(s0=s0, empty=True)
        need0 = self._need_atoms(q_at)
        cand, start = self._seeds(s0, s0n, need0)
        best = None
        for d1 in cand:
            chain, score, last = [d1], start[d1], None
            seen = np.concatenate([q_at, ix.p_atoms(d1)])
            need = need0[~self._explained(need0, ix.p_atoms(d1))]
            new = self._new_atoms(ix.p_atoms(d1), q_at, numeric_q)
            trace = []
            for _ in range(c.max_hops - 1):
                if len(new) == 0:
                    break
                s2 = self._hop(need, new, s0n)
                s2 = self._title_pen(s2, chain)
                s2[chain] = -np.inf
                d2 = int(np.argmax(s2))
                trace.append(dict(need=need.tolist(), new=new.tolist(), pick=d2))
                chain.append(d2)
                score += s2[d2]
                last = s2
                need = need[~self._explained(need, ix.p_atoms(d2))]
                new = self._new_atoms(ix.p_atoms(d2), seen, numeric_q)
                seen = np.concatenate([seen, ix.p_atoms(d2)])
            score /= len(chain)
            if best is None or score > best[0]:
                best = (score, chain, last, trace, need, seen)
        _, chain, last, trace, need, seen = best
        return dict(s0=s0, s0n=s0n, q_at=q_at, need0=need0, cand=cand, chain=chain, last=last,
                    trace=trace, need=need, seen=seen, empty=False)

    def _title_pen(self, s, chain):
        if self.title_id is not None and self.c.same_title_pen != 1.0:  # step 3: Thailand -> Thailand
            s = np.where(np.isin(self.title_id, self.title_id[chain]), self.c.same_title_pen * s, s)
        return s

    def _soft_cover(self, need0):
        """C[j, d] in [0,1]: how well passage d covers need atom j (0 below cov_lo, 1 at tau, linear between).
        Absolute scale (not min-max), so marginal gains are comparable across questions -> usable for stopping."""
        S = self._atom_doc(need0)
        return np.clip((S - self.c.cov_lo) / (self.c.tau - self.c.cov_lo), 0.0, 1.0)

    def _extend(self, b, numeric_q):
        """Step 4: grow the 2-doc chain into a tree. Each new passage must (a) link to the bridge frontier of
        ANY chosen passage (tree, not path) and (b) add soft coverage of the question's residual needs.
        Stop when the idf-weighted marginal coverage gain < lam (prize-collecting / OMP residual rule)."""
        ix, c = self.ix, self.c
        chain, last, trace, need, seen = list(b["chain"]), b["last"], list(b["trace"]), b["need"], b["seen"]
        need0 = b["need0"]
        if len(need0) == 0 or not np.isfinite(c.lam) or len(chain) >= c.max_docs:
            return chain, last, trace
        C = self._soft_cover(need0)
        w = ix.idf[need0] / ix.idf[need0].sum()
        cov = C[:, chain].max(1)
        frontier = np.unique(np.concatenate(
            [self._new_atoms(ix.p_atoms(d), np.concatenate([b["q_at"]] + [ix.p_atoms(e) for e in chain if e != d]),
                             numeric_q) for d in chain]))
        frontier = frontier[np.argsort(-ix.idf[frontier])][:2 * c.max_new]
        while len(chain) < c.max_docs and len(frontier):
            s = self._title_pen(self._hop(need, frontier, b["s0n"]), chain)
            s[chain] = -np.inf
            d = int(np.argmax(s))
            gain = float(w @ np.clip(C[:, d] - cov, 0.0, None))
            trace.append(dict(need=need.tolist(), new=frontier.tolist(), pick=d, gain=round(gain, 4)))
            if gain < c.lam:
                trace[-1]["stopped"] = True
                break
            chain.append(d)
            last = s
            cov = np.maximum(cov, C[:, d])
            need = need[~self._explained(need, ix.p_atoms(d))]
            nd = self._new_atoms(ix.p_atoms(d), seen, numeric_q)
            seen = np.concatenate([seen, ix.p_atoms(d)])
            frontier = np.union1d(frontier[~self._explained(frontier, ix.p_atoms(d))], nd)
            frontier = frontier[np.argsort(-ix.idf[frontier])][:2 * c.max_new]
        return chain, last, trace

    def rank(self, qi, qv, k, record=False, numeric_q=True, base_cache=None):
        """base_cache: dict shared across configs that differ only in extension params (lam, max_docs, cov_lo)."""
        if base_cache is not None and qi in base_cache:
            b = base_cache[qi]
        else:
            b = self._base(qi, qv, numeric_q)
            if base_cache is not None:
                base_cache[qi] = b
        if b["empty"]:
            return list(topk(b["s0"], k))
        chain, last, trace = self._extend(b, numeric_q)
        s0n = b["s0n"]
        tail = s0n if last is None else np.where(np.isfinite(last), 0.5 * mm(np.nan_to_num(last, neginf=0)) + 0.5 * s0n, 0)
        if record:
            self.c.log.append(dict(q=int(qi), chain=chain, trace=trace, seeds=b["cand"]))
        return fill(chain, tail, k)
