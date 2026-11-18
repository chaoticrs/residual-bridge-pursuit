# Phase 1 results (test split)

## musique

| method | R@2 | R@5 | R@10 | FSR@5 | FSR@10 | params |
|---|---|---|---|---|---|---|
| dense | 39.1 | 51.2 | 58.8 | 17.5 | 25.9 | `{}` |
| bm25 | 34.4 | 44.3 | 51.5 | 10.6 | 15.8 | `{}` |
| hybrid | 40.0 | 52.7 | 61.0 | 18.2 | 27.2 | `{}` |
| rocchio | 36.3 | 45.1 | 53.6 | 16.2 | 24.9 | `{'beta': 1.0}` |
| rbp_linear | 36.3 | 45.1 | 53.6 | 16.2 | 24.9 | `{'beta': 1.0}` |
| g3 | 39.1 | 46.6 | 49.6 | 14.0 | 17.6 | `{'lam': 0.1, 's': 3}` |
| rbp | 42.8 | 57.2 | 63.9 | 30.0 | 36.8 | `{'tnorm': 'product', 'need_agg': 'mean', 'alpha': 0.2, 'w_span': 0.5, 'max_hops': 2}` |
| rbp2 | 43.9 | 57.7 | 64.4 | 30.0 | 37.5 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbpt | 43.9 | 58.0 | 64.7 | 29.4 | 37.0 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'lam': 0.08, 'cov_lo': 0.4, 'max_docs': 4}` |
| rbp2_sum | 45.7 | 59.5 | 66.3 | 30.9 | 39.2 | `{'tnorm': 'sum', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_union | 40.6 | 54.4 | 62.5 | 25.6 | 34.2 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_air | 42.9 | 56.8 | 63.4 | 24.0 | 31.8 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'air_T': 4}` |

## 2wikimultihopqa

| method | R@2 | R@5 | R@10 | FSR@5 | FSR@10 | params |
|---|---|---|---|---|---|---|
| dense | 64.4 | 70.8 | 74.0 | 40.6 | 46.0 | `{}` |
| bm25 | 57.7 | 67.8 | 72.4 | 36.6 | 43.1 | `{}` |
| hybrid | 64.1 | 71.4 | 74.3 | 41.2 | 46.1 | `{}` |
| rocchio | 58.8 | 70.6 | 77.8 | 43.2 | 54.1 | `{'beta': 0.7}` |
| rbp_linear | 57.5 | 68.8 | 75.7 | 41.4 | 52.0 | `{'beta': 1.0}` |
| g3 | 64.4 | 69.2 | 73.4 | 36.8 | 43.8 | `{'lam': 0.4, 's': 2}` |
| rbp | 63.2 | 79.6 | 83.9 | 57.0 | 63.0 | `{'tnorm': 'product', 'need_agg': 'mean', 'alpha': 0.2, 'w_span': 0.5, 'max_hops': 2}` |
| rbp2 | 63.8 | 79.2 | 83.2 | 56.9 | 62.2 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbpt | 63.8 | 78.8 | 83.0 | 55.8 | 62.0 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'lam': 0.02, 'cov_lo': 0.4, 'max_docs': 4}` |
| rbp2_sum | 70.2 | 81.7 | 85.1 | 59.1 | 64.5 | `{'tnorm': 'sum', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_union | 65.3 | 80.8 | 84.5 | 58.4 | 64.4 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_air | 64.0 | 78.1 | 82.0 | 54.5 | 60.6 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.6, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'air_T': 4}` |

## hotpotqa

| method | R@2 | R@5 | R@10 | FSR@5 | FSR@10 | params |
|---|---|---|---|---|---|---|
| dense | 72.3 | 85.8 | 90.9 | 73.8 | 83.0 | `{}` |
| bm25 | 55.8 | 71.7 | 84.3 | 49.4 | 69.9 | `{}` |
| hybrid | 66.7 | 85.2 | 91.9 | 71.4 | 84.4 | `{}` |
| rocchio | 70.4 | 86.2 | 91.8 | 75.1 | 84.5 | `{'beta': 0.3}` |
| rbp_linear | 65.4 | 83.1 | 90.1 | 68.9 | 81.8 | `{'beta': 1.0}` |
| g3 | 72.3 | 82.8 | 84.8 | 68.4 | 71.9 | `{'lam': 0.1, 's': 3}` |
| rbp | 83.2 | 91.6 | 94.6 | 84.6 | 90.1 | `{'tnorm': 'product', 'need_agg': 'mean', 'alpha': 0.4, 'w_span': 0.5, 'max_hops': 2}` |
| rbp2 | 84.3 | 92.4 | 95.6 | 86.2 | 92.2 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.4, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.3, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbpt | 84.3 | 92.4 | 95.7 | 86.1 | 92.2 | `{'tnorm': 'product', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.4, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.3, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'lam': 0.02, 'cov_lo': 0.6, 'max_docs': 3}` |
| rbp2_sum | 84.4 | 93.0 | 95.9 | 87.5 | 92.9 | `{'tnorm': 'sum', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.3, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_union | 75.0 | 90.9 | 94.6 | 83.5 | 90.4 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.2, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.3, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2}` |
| rbp2_air | 75.3 | 89.2 | 93.3 | 79.6 | 87.2 | `{'tnorm': 'union', 'need_agg': 'mean', 'w_span': 0.5, 'alpha': 0.4, 'need_max_df': 0.05, 'need_lemma_max_df': 0.005, 'anchor_m': 2, 'seed_w': 0.3, 'drop_numeric_new': True, 'same_title_pen': 0.5, 'max_hops': 2, 'air_T': 4}` |
