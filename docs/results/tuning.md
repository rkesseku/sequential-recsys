# Does tuning help? Optuna search for the LightGBM re-ranker

Search: TPE (seed 0); trial 0 = default recipe; no pruning; search features from SASRec seed 0, LightGBM seed 0. Data: validation stage only: candidates/features from training histories, labels = validation film; test targets never read during the search. Split: validation-stage users split at random (seed 0): 4832 fit users, 1208 tune users. Objective: NDCG@10 of the re-ordered candidates on the tune users (missing validation film = miss), ranker trained on the fit users with the repo routine (10% of fit users with a positive for early stopping, patience 50). Final: ranker trained on all validation-stage users (repo routine), scored once on the test targets; seed s = SASRec seed s features + LightGBM seed s. Bootstrap: 1000 paired resamples of users, tuned minus default. Threads: 13. Total time 6.6 min.

100 trials in 3.7 min. Tune-user NDCG@10 (search seed): default 0.1754 (38 trees), best 0.1872 (trial 33, 117 trees). Re-checked over LightGBM seeds [0, 1, 2]: default 0.1738, tuned 0.1849.

| Parameter | Default | Tuned | Range searched |
|---|---|---|---|
| num_leaves | 31 | 11 | 7-255 |
| learning_rate | 0.05 | 0.06613 | 0.01-0.3 |
| min_child_samples | 50 | 8 | 5-500 |
| feature_fraction | 0.9 | 0.7244 | 0.4-1.0 |
| bagging_fraction | 0.8 | 0.6588 | 0.4-1.0 |
| bagging_freq | 1 | 3 | 1-10 |
| lambda_l2 | 0 | 3.417 | 0.0-100.0 |
| n_estimators | 600 | 1392 | 200-2000 |

Test (6,040 users when run on MovieLens-1M), per seed:

| Seed | Default NDCG@10 | Tuned NDCG@10 | Default HR@10 | Tuned HR@10 | Trees (default / tuned) | Tuned - default NDCG@10 [95% CI] |
|---|---|---|---|---|---|---|
| 0 | 0.1751 | 0.1761 | 0.3045 | 0.3051 | 47 / 18 | +0.0010 [-0.0019, +0.0039] |
| 1 | 0.1755 | 0.1732 | 0.3045 | 0.3012 | 15 / 3 | -0.0023 [-0.0058, +0.0015] |
| 2 | 0.1746 | 0.1773 | 0.3028 | 0.3056 | 43 / 21 | +0.0028 [+0.0000, +0.0057] |

Mean over seeds: default NDCG@10 0.1750 ± 0.0005, tuned 0.1755 ± 0.0021 (mean ± SD). Tuned minus default on each user's seed-averaged NDCG@10: +0.0005 [-0.0013, +0.0025].
