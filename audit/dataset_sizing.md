# Dataset sizing — how big the train/val label artifact must be

**Purpose.** Audit finding I1: `run_info_matching`'s budget ladder requests `m*N` training
rows for `m` in 1..5, and `subsample_train` caps at `train_ds.n_rows`. If the frozen TRAIN
split holds fewer than `5N` rows the top rungs train on bit-identical data, the Gamma-RMSE
curve is flat by construction, and `plateau_reached=True` is indistinguishable from a real
information plateau. Fix batch 1 makes the cap loud; this memo makes it never fire.

**Answer up front.** Generate **`--n-param-points 512`** (with `--n-skt 64`, seed 42). That
yields a measured-basis TRAIN split of **≈23 760 rows = 5.80N**, a **+16 %** margin over the
`5N = 20 480` requirement. The repo default of 448 also passes, but by **+2.3 % (472 rows)** —
inside the uncertainty of the retention estimate, so it is not a safe choice.

Cost of the recommendation: **≈16 min single-core** and **≈2.9 MB** on disk. Both are small
enough that the margin is effectively free; the reason not to take an even larger margin is
scientific, not economic (see [Margin](#margin-why-16-and-not-more)).

---

## 0. Correction to the task framing: `make_labels` is not the production entry point

The task asks for a `make_labels --n-points`. `make_labels.generate_labels` does produce the
label artifact, but the **frozen train/val artifact is built by
[`make_datasets.py train_val`](../make_datasets.py#L135)**, which wraps `make_labels` in
chunking + resume and — decisively for this question — is the only path that writes the
`split` key. Three consequences:

1. A bare `make_labels` artifact has **no `split` array**, so `ArmDataset._resolve_split`
   ([train_pinn.py:284-292](../train_pinn.py#L284-L292)) falls back to an 80/20
   by-parameter-point split with `rng [seed, 21]` **and raises a `RuntimeWarning`**. The rule is
   numerically identical to `make_datasets`', so the row counts below hold either way — but the
   split is then not recorded in the artifact, and it silently depends on `train.py --seed`
   matching the generation seed.
2. `make_datasets.generate_train_val` **already enforces this memo's budget** at
   [make_datasets.py:258-279](../make_datasets.py#L258-L279): floors `(5N, N)` from
   `info_budget(pinn_raw)` — 20 480 and 4 096 at today's `n_price_points` — and on failure it
   raises with the `n_param_points` you should have used. Running `make_labels` directly
   **bypasses that check** — exactly the failure mode I1 is about.
3. The default `--n-skt` differs: 16 in `make_labels`, **64** in `make_datasets`. Every number
   here assumes 64.

Both command lines are given in [§6](#6-the-command-line). Use the `make_datasets` one.

---

## 1. Required train rows

```
N            = pinn_config.yaml  shared.n_price_points   = 4096
required     = 5 * N                                     = 20480 TRAIN rows
also required= N (VAL floor, make_datasets.info_budget)   = 4096 VAL rows
```

`5N` is the contract's cap (`information_matching.cap: 5N`, contract line 282) mirrored by
`pinn_config.yaml info_matching.cap_multiplier: 5` and enforced on the config ladder by
`saturation_sweep_configs`.

> **~~Fragility worth a separate ticket (not fixed here).~~ RESOLVED — fix batch 4 ITEM 2.**
> `make_datasets.py:70` used to hardcode `N_INFO = 4096` as a bare literal rather than reading
> `shared.n_price_points`, with no test tying them, so a change to `n_price_points` would have
> left the budget check validating against the stale 4096 and I1 would have come back silently
> (the same single-source-of-truth class as audit C1). The floors are now
> `make_datasets.info_budget(pinn_raw)` = `(5N, N)` with `N` read from the config, guarded by a
> parity test and a mutation test (`test_make_datasets.py`). The `5 * N` and `N` figures below
> are still today's values — they are now *derived* from `pinn_config.yaml` rather than
> re-typed beside it.

---

## 2. The retention chain, stage by stage

Rows are counted from generation to `ArmDataset(..., "train").n_rows`. Measured values are
marked ✅, assumptions ⚠️.

| # | Stage | Where | Factor |
|---|---|---|---|
| 1 | Parameter sampling: Feller floor 0.40 + 10 % anchor excision | `train_pinn.sample_hypercube_params` | **×1.000** ✅ — the sampler *resamples until `n` are accepted* ([train_pinn.py:157-179](../train_pinn.py#L157-L179)), so rejection costs **zero delivered points**. Verified: `sample_hypercube_params(n=448, seed=42)` returns exactly 448. |
| 2 | Rows per parameter point | `make_labels.py:143-150`, `build_arm_labels:330-333` | **× n_skt = 64** ✅ — `(S, K, tau)` are **`n_skt` uniform triples drawn once and shared** by every parameter point (`np.tile`/`np.repeat`), **not** a grid product. The contract `grid: {S.n: 41, K.n: 33, tau.n: 16}` applies to the *anchor grids* (`generate_anchor_grids`), never to the hypercube label artifact. |
| 3 | Oracle mask `mask_any` | `make_labels.py:183`, applied in `build_arm_labels:329-335` | **×0.911** ✅ measured — see [§3](#3-the-oracle-mask-measured) |
| 4 | Train/val split, **by parameter point** | `make_datasets.py:181-186`; fallback `train_pinn.py:288` | **×0.80** (train) / ×0.20 (val) ✅ — `val_param_frac` default `0.20`; the contract does **not** declare it (`splits:` covers eval holdouts only), so 0.20 is the code default in both producer and consumer |
| 5 | Any further filter before `ArmDataset` | — | **×1.000** ✅ **measured: none.** R09 §4 walks the chain and gets `ArmDataset(...,"train").n_rows == 1811`, exactly the chain's prediction. No NaN drop, no dedup, no `n_price_points` subsampling — `n_price_points` is consumed **only** by `saturation_sweep_configs`; every non-info-matching arm trains on the whole split. |

**Chain as a product:**

```
train_rows = n_param_points × (1 − val_param_frac) × n_skt × keep
           = n_param_points × 0.80 × 64 × 0.911
           = n_param_points × 46.65

val_rows   = n_param_points × 0.20 × 64 × 0.911   = n_param_points × 11.66
```

Because the split is by parameter point and the mask is heavily clustered *within* a point
(band points mask ~52 %, plain points ~7 %), the true train retention differs from the global
`keep` by which points land where. R09c composes each split with its own category counts;
those are the numbers in the table in [§4](#4-candidate-sizes).

---

## 3. The oracle mask (measured)

**No historical retention rate exists anywhere in the repo** — no `data/` directory, no
manifest, no recorded `mask_rate` in any doc, test or artifact. Nothing was assumed: the rate
was measured by running `make_labels.generate_labels` **unchanged, with production leg
kwargs**, on points drawn from the actual production hypercube.

The mask is not homogeneous. `cross_validate` masks a row when any pair of legs disagrees by
more than `tol_rel = 1e-3 × scale`, so **more legs means more chances to disagree**, and the
contract's fourth-leg band clause (Feller ratio in `[0.40, 0.60]` → ADI) creates a category
that masks an order of magnitude harder:

`audit/repro/r09b_retention_by_category.py` → `r09b_output.txt`:

```
  [cf+fd] 160 points, 10240 rows, 91 s
    mask rate        : 0.0685  +/- 0.0034 (SE clustered by parameter point)
    per greek        : price 0.0000, delta 0.0549, gamma 0.0562, vega 0.0517, vanna 0.0591

  [cf+fd+mc] 24 points, 1536 rows, 41 s
    mask rate        : 0.0749  +/- 0.0076 (SE clustered by parameter point)

  [band (ALL production band points)] 20 points, 1280 rows, 461 s
    legs seen        : ['adi+cf+fd', 'adi+cf+fd+mc']
    mask rate        : 0.5211  +/- 0.0522 (SE clustered by parameter point)
    per greek        : price 0.0930, delta 0.1305, gamma 0.1023, vega 0.3945, vanna 0.3906
```

Composed at the **exact** production mix (the band set and the MC subset are deterministic
functions of `(n_param_points, seed)`, so the weights are counted, not estimated):

```
  cf+fd      weight 0.8571 x mask 0.0685 = 0.05868
  cf+fd+mc   weight 0.0982 x mask 0.0749 = 0.00735
  band       weight 0.0446 x mask 0.5211 = 0.02326
  composed mask rate  : 0.0893 +/- 0.0030
  composed RETENTION  : 0.9107  (95% lower bound 0.9048)
```

Two things to note, because they set the margin:

- **The ADI band carries 26 % of all masking on 4.5 % of the points.** The disagreement is
  concentrated in **vega (0.39) and vanna (0.39)** — ADI vs CF/FD near the Feller boundary,
  which is precisely the regime the contract added the fourth leg for. This is expected
  behaviour of the oracle, not a bug, but it means the band share is the single biggest lever
  on artifact size.
- **A single small probe biases the estimate.** The first probe
  (`r09_dataset_retention.py`, 40 points) happened to draw 4 band points — 10 % of the probe
  against the sample's 4.46 % — and reported retention 0.8941 instead of 0.9107. That is why
  the estimate above is composed per category rather than taken from one run.

---

## 4. Candidate sizes

`audit/repro/r09c_candidate_sizes.py` → `r09c_output.txt`. Band membership and the
`rng [seed, 21]` split are computed **exactly** for each candidate at the production seed 42;
only the three per-category mask rates come from §3.

```
 n_pts  band  bnd_tr  keep_tr  train_rows    margin  val_rows   hours  verdict
------------------------------------------------------------------------------
   384    24      21   0.8999       17682   -13.7%      4500    0.22     FAIL
   416    22      18   0.9065       19320    -5.7%      4828    0.22     FAIL
   448    20      13   0.9145       20952    +2.3%      5158    0.21     PASS
   480    28      21   0.9062       22271    +8.7%      5516    0.27     PASS
   512    26      23   0.9055       23760   +16.0%      5990    0.26     PASS
   544    28      21   0.9090       25308   +23.6%      6291    0.28     PASS
   576    35      27   0.9045       26686   +30.3%      6617    0.33     PASS
   608    36      31   0.9021       28058   +37.0%      7121    0.34     PASS
   640    36      25   0.9088       29780   +45.4%      7306    0.35     PASS
```

- **Binding constraint is TRAIN, always.** Val needs `n ≥ 352`, train needs `n ≥ 440`; the
  `5N` train floor is ~1.25× stricter than the `N` val floor at `val_param_frac = 0.20`.
- **Implied minimum `--n-param-points` = 440** at the point estimate (443 at the 95 % lower
  retention bound). Rounded up to `make_datasets`' `CHUNK_MAX = 32` chunk boundary: **448**.
- The repo default **448 clears the bar by 472 rows (+2.3 %)**. That is not comfortable: the
  ±0.0030 SE on composed retention alone is ±0.33 % of rows, and any drift in the CF/FD leg
  settings (grid resolution, FD step) moves the plain-category rate directly.

Sensitivity to the one variable that actually moves (band share):

```
  w_band     keep        448        512        544
   0.020   0.9218      21121       24189       25663
   0.045   0.9105      20862       23892       25349
   0.060   0.9037      20707       23714       25160
   0.080   0.8947      20499       23477       24909
   0.100   0.8857      20292*      23240       24657
   0.120   0.8766      20085*      23003       24405
   0.150   0.8631      19775*      22647       24028
  (* = below the 5N = 20480 requirement)
```

**448 fails at a band share of 10 %; 512 survives past 15 %.** Across seeds 42–51 the band
share at n=448 ranges 0.0379–0.0670, so 448 is not *likely* to fail at seed 42 — but it has no
room for a reseed, a leg-kwargs change, or a `val_param_frac` change.

---

## 5. Margin: why +16 % and not more

Compute and disk do **not** constrain this decision:

- **Wall clock ≈ 0.26 h single-core** at n=512 (measured on this machine: CF+FD 0.57 s/point,
  +MC 1.71 s/point, +ADI **23.05 s/point** at `n_skt=64`). ADI is 60 % of the total on 5 % of
  the points. Generation is chunked (≤32 points) and resumable, so it can be split or restarted.
- **Disk ≈ 2.85 MB** — see [§7](#7-disk-footprint). Negligible.

So margin is chosen on scientific grounds. The constraint pushing *down*:

> Every non-info-matching arm trains on the **whole** train split (`n_price_points` is consumed
> only by `saturation_sweep_configs`), while the info-matched baseline's selected rung uses
> `m*N ≤ 5N`. Oversizing the artifact therefore widens the gap between the ladder's top rung
> and the budget every other arm actually gets, which muddies the "info-matched" framing.

Train split as a multiple of N: 448 → 5.12N · **512 → 5.80N** · 544 → 6.18N · 576 → 6.52N.

**512 is the balance point:** it keeps the train split under 6N while giving the `5N` ladder a
16 % cushion — enough to absorb the retention SE (±0.33 %), a reseed (worst case over seeds
42–51 at n=512 is 23 528 rows, still +15 %), a band share up to 15 %, and a moderate change to
the FD/CF leg settings. It is also `16 × CHUNK_MAX`, so the chunked generator produces exactly
16 full parts with no ragged tail.

If you want a harder guarantee against a `val_param_frac` change or a future `n_price_points`
increase, **544** (+23.6 %, 6.18N, 0.28 h) is the next chunk-aligned step. I would not go past
576 for the reason above.

---

## 6. The command line

**Use this** — production path, writes `split`, enforces the budget, chunked and resumable:

```bash
python make_datasets.py train_val \
  --contract heston_benchmark_v6.yaml \
  --pinn-cfg pinn_config.yaml \
  --out-dir data/staging/train_val_v6 \
  --seed 42 \
  --n-param-points 512 \
  --n-skt 64 \
  --mc-subset-frac 0.10 \
  --val-param-frac 0.20
```

Expect on stdout:

```
budget check: retained TRAIN rows ~23760 (need >= 20480 = 5N), retained VAL rows ~5990
              (need >= 4096); mask rate ~0.089 -> PASS
```

`--out-dir` must not contain `frozen` (asserted). Promotion to `data/frozen/<tag>` + git tag
stays a human step, after `mask_neutrality_report.md` is reviewed — the run prints that
reminder itself.

**The literal `make_labels` equivalent**, for completeness only — it writes **no `split` key**
and runs **no budget check**, so it re-opens I1:

```bash
python make_labels.py \
  --contract heston_benchmark_v6.yaml \
  --pinn-cfg pinn_config.yaml \
  --n-points 512 \
  --n-skt 64 \
  --seed 42 \
  --mc-subset-frac 0.10 \
  --out-dir data/staging/labels_v6
```

If you use it, `train.py` must be run with `--seed 42` so `ArmDataset`'s fallback split
reproduces the same partition, and the `RuntimeWarning` about the missing `split` key must be
treated as expected rather than noise.

---

## 7. Disk footprint

`train_val_labels.npz` is uncompressed `np.savez`. Columns are 5 label quantities ×
{`consensus_` f8, `uncertainty_` f8, `mask_` bool} plus `mask_any` bool, over
`n_param_points × n_skt` rows; the small per-point vectors (`params` 5×f8, `feller_ratio`,
`adi_points`, `split`) and the shared 3×64 `(S, K, tau)` are rounding error.

| n_param_points | rows | consensus | uncertainty | masks | **total npz** |
|---|---|---|---|---|---|
| 448 | 28 672 | 1.15 MB | 1.15 MB | 0.17 MB | **2.49 MB** |
| **512** | **32 768** | **1.31 MB** | **1.31 MB** | **0.20 MB** | **2.85 MB** |
| 544 | 34 816 | 1.39 MB | 1.39 MB | 0.21 MB | **3.02 MB** |

`parts/` holds the same data again as ≤32-point chunks (≈1× more, plus per-part manifests)
and can be deleted after the merge verifies — budget ~6 MB transient at n=512. The anchor
grids (`generate_anchor_grids`, TASK B) are a separate artifact and are not sized here.

Storage is not a consideration at this scale. Compute is: **≈16 min at n=512**, of which
**≈10 min is the ADI leg** on the 26 Feller-band points.

---

## 8. Measured vs assumed

**Measured (executed on this machine, real code paths, outputs in `audit/repro/`):**

| Quantity | Value | Where |
|---|---|---|
| Rows per parameter point = `n_skt`, shared `(S,K,tau)`, not a grid product | 64 | code read + R09 §1 |
| Sampler delivers exactly `n` points (Feller/excision cost 0) | ×1.000 | R09 §1 |
| Mask rate, `cf+fd` category | 0.0685 ± 0.0034 (160 points) | R09b |
| Mask rate, `cf+fd+mc` category | 0.0749 ± 0.0076 (24 points) | R09b |
| Mask rate, ADI-band category | 0.5211 ± 0.0522 (**all 20** production band points) | R09b |
| Composed retention at the production mix | 0.9107 ± 0.0030 | R09b |
| Band / MC / split membership per candidate `n` | exact counts, seed 42 | R09c |
| No filter between `build_arm_labels` and `ArmDataset` | `n_rows` == chain prediction, exactly | R09 §4 |
| Per-leg wall clock at `n_skt=64` | CF 0.06 s, FD 0.46 s, MC(200k) 0.98 s, ADI 23.5 s | timing run in §5 |

**Assumed (stated, not measured):**

1. **The three per-category mask rates transfer across `n_param_points`.** Measured on the
   n=448 sample; a different `n` draws different Latin-hypercube points. The band *count* is
   computed exactly per candidate, but its 0.5211 *rate* is carried over. The 20-point band
   census has SE 0.0522, which is why the sensitivity table in §4 is the operative check
   rather than the point estimate.
2. **`val_param_frac = 0.20`.** Both `make_datasets`' default and `ArmDataset`'s fallback, but
   **the contract does not declare it**. If a human sets it differently, every number here
   moves — the chain in §2 is the formula to redo.
3. **Production leg kwargs = `oracle.py` defaults** (`heston_greeks_mc` 200 000 paths,
   250 steps/yr; CF/FD/ADI grid defaults). The probes passed no `leg_kwargs`. If the freeze run
   changes any leg's resolution, the CF-vs-FD agreement rate changes and this must be re-measured.
4. **`mc_subset_frac = 0.10`.** The `make_datasets` default; not contract-declared. Its effect
   is small either way: the MC category masks only 0.0064 more than the plain one, so at 10 %
   weight it adds ~0.0006 to the composed mask rate.
5. **`n_skt = 64`.** The `make_datasets` default, not contract-declared. `make_labels`' own
   default is 16 — using it would need 4× the parameter points and 4× the ADI cost.
6. ~~**The 5N floor stays 4096-based.** `make_datasets.N_INFO` is a hardcoded literal, not read
   from `pinn_config.yaml`, and no test locks them (see §1).~~ **No longer an assumption** —
   fix batch 4 ITEM 2 made the floors read `shared.n_price_points`, with a test locking them
   (see §1). The floor now tracks the config by construction; what remains assumed is only
   that `n_price_points` itself stays 4096.
7. **Single-core timings.** Measured on this machine; a different host moves the wall-clock
   column but nothing else.

**Reproduce:**

```bash
python audit/repro/r09_dataset_retention.py  <scratch_dir> 40      # ~2 min  — end-to-end chain
python audit/repro/r09b_retention_by_category.py <scratch_dir> 160 24  # ~10 min — per-category rates
python audit/repro/r09c_candidate_sizes.py                          # ~1 min  — composition + sizing
```

Nothing was generated into `data/`; the probes wrote to a scratch directory only, and
`generate_labels` refuses any path containing `frozen` regardless.
