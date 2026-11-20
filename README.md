# sequential-recsys

[![CI](https://github.com/rkesseku/sequential-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/sequential-recsys/actions/workflows/ci.yml)

**Sequential recommendation on MovieLens-1M, from candidate retrieval to learned re-ranking, evaluated the strict way.**
Four retrievers (popularity, item-kNN, EASE and a SASRec-style transformer) are compared on one
chronological split. Every user's held-out next film is ranked against the *whole* catalogue rather
than a sample of 100 negatives, and a LightGBM LambdaRank model re-ranks the merged top-100
candidates. Every number below comes from `seqrec run-all` and is stored in
[`docs/results/metrics.json`](docs/results/metrics.json).

## Why this exists

Recommender papers and portfolios often report inflated accuracy because of three shortcuts:
ranking the true item against 100 sampled negatives, ignoring interaction order, or tuning on the
test set. This repo avoids all three. The questions it answers:

1. How much does **modelling the order of a user's history** buy over strong order-agnostic baselines
   such as EASE, a well-known linear model that is hard to beat?
2. Does a **second-stage learned ranker**, the standard retrieval → ranking architecture used in
   production, add anything measurable on top of the best retriever?
3. What happens to **catalogue coverage and popularity bias** as accuracy improves?

## Data

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) (Harper & Konstan, 2015):
1,000,209 ratings from 6,040 users on 3,706 films. Every rating is treated as an implicit
interaction. **The GroupLens licence does not allow redistribution, so the data is not in this
repo.** Download `ml-1m.zip` from GroupLens yourself and unzip it to `data/raw/ml-1m/`.

## Protocol

- **Split.** Per user, interactions are sorted by timestamp (ties keep file order). The last item is
  the test target and the second-to-last is the validation target. The rest is training data,
  averaging 163.6 items per user.
- **Ranking.** Each model scores all 3,706 films. Films the user has already seen are removed, and
  ties are broken pessimistically against the target. Metrics: HR@10, NDCG@10, Recall@K (equal to HR@K with
  one target), coverage@10 (share of the catalogue appearing in anyone's top 10) and head share@10
  (share of top-10 slots given to the 10% most-interacted films).
- **No test leakage.** All models are fit on training interactions only. Hyperparameters (item-kNN
  `k`, EASE λ) and SASRec early stopping use validation NDCG@10. At test time each model reads the
  training history plus the validation item.
- **Uncertainty.** 95% percentile bootstrap intervals over users (1,000 resamples), a paired
  bootstrap for model differences, and three SASRec seeds.

## Models

| Model | What it is |
|---|---|
| Popularity | Most-interacted films in training data |
| ItemKNN | Cosine item-item similarity on the binary matrix, shrinkage 10, top-k neighbours (k tuned: 50) |
| EASE | Closed-form item-item autoencoder (Steck, 2019), λ tuned on validation (250) |
| SASRec | Causal self-attention over the last 100 items (Kang & McAuley, 2018): 2 pre-LN blocks, d=64, 1 head, dropout 0.2, shared input/output embeddings, **full softmax cross-entropy** instead of the paper's single sampled negative, Adam lr 2e-3, at most 120 epochs, early stopping (patience 3 evaluations, every 2 epochs; every seed stopped early, after 38–56 epochs) |
| Re-ranker | LightGBM LambdaRank over the top 100 of the merged EASE + SASRec lists. 10 features: EASE, SASRec and ItemKNN score and rank, log popularity, log recent popularity, log history length, genre affinity. Trained on the **validation** targets, evaluated on test |

## Results (test set, 6,040 users)

| Model | NDCG@10 [95% CI] | HR@10 [95% CI] | Recall@50 | Coverage@10 | Head share@10 |
|---|---|---|---|---|---|
| Popularity | 0.0180 [0.0156, 0.0206] | 0.0369 [0.0326, 0.0416] | 0.139 | 5.4% | 100% |
| ItemKNN (k=50) | 0.0403 [0.0364, 0.0443] | 0.0805 [0.0735, 0.0876] | 0.252 | 37.7% | 82.3% |
| EASE (λ=250) | 0.0530 [0.0491, 0.0574] | 0.1079 [0.1005, 0.1159] | 0.320 | 38.1% | 81.5% |
| SASRec (seed 0) | 0.1666 [0.1593, 0.1744] | 0.2907 [0.2803, 0.3030] | 0.558 | 64.8% | 46.8% |
| **Re-ranker** | **0.1751 [0.1676, 0.1831]** | **0.3045 [0.2930, 0.3169]** | 0.569* | 60.5% | 53.7% |

\*The re-ranker can only reorder its 100 candidates. The test target was among them for 64.5% of
users, slightly below SASRec's own Recall@100 of 0.668, because half of each candidate list comes
from EASE.

**Paired differences in NDCG@10** (bootstrap over users):

| Comparison | Difference [95% CI] |
|---|---|
| SASRec − EASE | +0.1135 [+0.1061, +0.1216] |
| Re-ranker − SASRec | +0.0085 [+0.0047, +0.0123] |
| EASE − ItemKNN | +0.0128 [+0.0092, +0.0166] |

**Across SASRec seeds** (0, 1, 2), test NDCG@10 is 0.1662 ± 0.0015 (mean ± SD) and HR@10 is
0.2913 ± 0.0017, so the per-user intervals above are not an artefact of one lucky seed. With the
epoch cap raised from 40 to 120, every seed was halted by validation early stopping (seed 0 after 56
epochs with its best at epoch 50, seed 1 after 48 with its best at 42, and seed 2 after 38 with its best at 32). Under the old
40-epoch cap, seed 0 was still improving when it was cut off. Training it to convergence raised its
test NDCG@10 from 0.1653 to 0.1666.

What the numbers say:

- **Order matters a lot here.** The sequential model triples the NDCG@10 of the best order-agnostic
  model. A film fan's next pick depends strongly on what they watched most recently, and EASE and
  ItemKNN discard that signal.
- **The learned ranker adds a small but statistically clear gain** (+0.0085, about 5% relative).
  It did not shrink when SASRec was trained to convergence. In fact it grew, from +0.0066 against the
  40-epoch model. The gain is about five times the seed-to-seed SD of SASRec, but it is measured
  against one SASRec seed. The ranker relies mostly on SASRec's rank and score (together 67% of the trees' total gain),
  with EASE, recency and genre signals as minor corrections. The ranker's own ceiling is the 64.5%
  candidate recall.
- **Better accuracy came with less popularity concentration.** SASRec gives 47% of top-10 slots to
  the head of the catalogue (EASE: 81%) and covers 65% of the catalogue (EASE: 38%). The re-ranker
  trades a little of that back: head share rises to 53.7% and coverage falls to 60.5%.

![Accuracy](docs/results/accuracy.png)
![Recall@K](docs/results/recall_curve.png)
![SASRec training](docs/results/sasrec_training.png)
![Accuracy vs popularity](docs/results/accuracy_vs_popularity.png)

**Compute.** Everything ran on CPU with 2 threads on a shared 14-core machine. SASRec took 35–95
minutes of wall-clock time per seed (38–56 epochs, with per-epoch time varying with machine load). With
cached checkpoints, the rest of `run-all` (tuning, EASE, ItemKNN, features, re-ranker, bootstrap)
took 108 s.

## Does tuning help?

The re-ranker above uses one fixed LightGBM recipe. `seqrec tune` asks whether a hyperparameter
search improves it. It runs a 100-trial Optuna TPE search (Akiba et al., 2019) with sampler seed 0, with the
fixed recipe entered as trial 0. The search covers `num_leaves`, `learning_rate`, `min_child_samples`,
`feature_fraction`, `bagging_fraction`/`bagging_freq`, `lambda_l2` and the `n_estimators` cap
(with early stopping as in the main pipeline).

**Validation without touching test.** The ranker is *trained* on the validation films, so the
same films cannot also be used to score its hyperparameters. A separate time-based split was not
carved out, because that would mean retraining SASRec on shorter histories. Instead, the 6,040
validation-stage users are split once at random into 4,832 *fit* users and 1,208 *tune* users. Each
trial trains the ranker on the fit users (same routine, same 10% early-stopping hold-out) and is
scored by NDCG@10 on the tune users. Every candidate, feature and label in the search comes from
training histories and validation films. Test targets are never read (a unit test checks that corrupting
them leaves every trial unchanged). Afterwards both recipes are trained on all validation-stage
users and scored once on test for seeds 0, 1 and 2 (seed *s* means features from SASRec seed *s*
and LightGBM seed *s*). The difference, tuned minus default, is assessed with a paired bootstrap
over users (1,000 resamples).

| Recipe | Validation NDCG@10 | Test NDCG@10, seed 0 | Seed 1 | Seed 2 | Test NDCG@10, mean ± SD | Test HR@10, mean |
|---|---|---|---|---|---|---|
| Default | 0.1738 | 0.1751 | **0.1755** | 0.1746 | 0.1750 ± 0.0005 | 0.3039 |
| Tuned | **0.1849** | **0.1761** | 0.1732 | **0.1773** | **0.1755** ± 0.0021 | **0.3040** |

Bold marks the best value in each column (highest). Validation NDCG@10 is on the 1,208 tune users,
averaged over LightGBM seeds 0-2. These users also selected the tuned recipe, so its validation
value is optimistic. Test values are for 6,040 users, and the seed-0 default is the headline re-ranker
(0.1751).

**Tuned minus default, test NDCG@10** (95% paired bootstrap interval over users) is +0.0010
[−0.0019, +0.0039] for seed 0, −0.0023 [−0.0058, +0.0015] for seed 1 and +0.0028 [+0.0000, +0.0057]
for seed 2. On each user's NDCG@10 averaged over the three seeds it is +0.0005 [−0.0013, +0.0025].

Chosen recipe: `num_leaves` 11, `learning_rate` 0.066, `min_child_samples` 8, `feature_fraction`
0.72, `bagging_fraction` 0.66 with `bagging_freq` 3, `lambda_l2` 3.4, `n_estimators` cap 1,392
(default: 31, 0.05, 50, 0.9, 0.8 / 1, 0, 600). The search took 3.7 minutes (100 trials of 1-9 s each)
and the whole study 6.6 minutes with 13 LightGBM threads. Details are in
[`docs/results/tuning.md`](docs/results/tuning.md).

**Reading.** Tuning does not help on the test set. The search raised tune-user NDCG@10 by about 0.011, but
on the test set the tuned recipe is better for two seeds and worse for one, and its seed-averaged gain of
+0.0005 has an interval that includes zero. It is also less stable across seeds (SD 0.0021 against
0.0005). One visible reason: on the full validation stage the tuned recipe early-stopped after only
3-21 trees (default: 15-47), so the final model depends more on the noisy 10% early-stopping hold-out
than on the searched parameters. The headline numbers therefore stay with the fixed recipe, and the
conclusion that the re-ranker adds +0.0085 over SASRec is unchanged.

## Limitations

- **One dataset, one split type.** Leave-last-out per user does not hold out a global time window,
  so other users' later interactions can inform training. This is the standard protocol for this
  benchmark but is not a deployment simulation. A global temporal split would be stricter.
- **Training budget.** SASRec is early-stopped on validation NDCG with patience 3 (6 epochs). All
  three seeds stopped well before the 120-epoch cap, but a larger patience or a learning-rate
  schedule might still find slightly better checkpoints. Runs are on CPU with multithreaded
  kernels, so re-training a seed reproduces its numbers only approximately.
- **SASRec variant.** It uses full softmax cross-entropy and `torch.nn` pre-LN blocks instead of the
  paper's exact sampled-BCE setup, so the numbers are not directly comparable to sampled-metric
  results in the literature.
- **Re-ranker fairness.** The ranker is trained on validation targets with the seed-0 SASRec,
  which was itself early-stopped on those same validation targets. Its test evaluation is clean,
  but its training signal is slightly optimistic.
- **Implicit feedback only.** Ratings are binarised, and there are no user or item side features
  apart from genres.

## How to run

```bash
pip install -e ".[dev]"            # CPU torch is enough; dev includes optuna
ruff check src tests && pytest     # synthetic data only, no downloads
# put MovieLens-1M in data/raw/ml-1m/ first
seqrec train-sasrec --seed 0 --threads 2 --verbose    # optional: train/cache one seed at a time
seqrec run-all --threads 2 --verbose                  # tunes, trains/loads 3 seeds, writes docs/results/
seqrec plots                                          # redraw figures from metrics.json
seqrec tune --threads 13 --n-jobs 13                 # Optuna study -> docs/results/tuning.*
```

SASRec checkpoints are cached in `results/cache/` (gitignored), keyed on the seed and a hash of the
training config, so `run-all` reuses them only when the settings match. On some
Windows conda environments, numpy (MKL) and PyTorch load two OpenMP runtimes. If you see
`OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` for the session. CI on Linux does not need this.

The project brief is at [`docs/brief/brief.pdf`](docs/brief/brief.pdf).

## Layout

```
src/seqrec/
  data.py       loading, chronological split, synthetic logs for tests
  metrics.py    full-catalogue ranks, HR/NDCG, coverage, popularity, bootstrap
  models.py     Popularity, ItemKNN, EASE
  sasrec.py     SASRec-style transformer with early stopping
  rerank.py     candidate merging, features, LightGBM LambdaRank
  tuning.py     Optuna study for the re-ranker -> tuning.json, tuning.md
  pipeline.py   end-to-end experiment -> metrics.json
  plots.py, cli.py
tests/          23 tests on tiny synthetic interaction logs
```

## References

- T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama. Optuna: A Next-generation Hyperparameter
  Optimization Framework. *KDD*, 2019.
- F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context. *ACM TiiS*, 2015.
- W.-C. Kang and J. McAuley. Self-Attentive Sequential Recommendation. *ICDM*, 2018.
- H. Steck. Embarrassingly Shallow Autoencoders for Sparse Data. *WWW*, 2019.

## License

MIT (code). MovieLens data is subject to the GroupLens licence and is not included.
