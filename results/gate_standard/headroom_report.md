# Oracle-headroom gate report

Contract `oracle_headroom_gate` (runs_before: all_training). The
spread between the oracle hedge and the delta-corrupted oracle hedge
is the CEILING on any effect the project can show on the primary
metric (misspecified delta-only CVaR95).

- corruption mode: **field**  (PRIMARY: frozen smooth RFF field)
- cell: combined perturbation, magnitude 1.0, baseline train regime, T' = 0.17, freq = 252/yr, n_paths = 10000, seeds = [42, 43, 44, 45, 46]
- rms(Gamma_oracle) on the reference cloud: **0.0435638**
- absolute sigma_gamma targets: pilot: 0.00404684
- sigma ladder (pilot, contract `oracle_headroom_gate.sigma_rel_ladder`): decision rungs 0.0928946; DIAGNOSTIC-ONLY rungs none — swept, reported and plotted, NEVER an input to the DECISION scan  **(LADDER OVERRIDDEN on the command line — this is NOT the contract ladder)**

## Spread over seeds (spread_rel = (cvar_noisy - cvar_oracle) / cvar_oracle)

| arm | role | sigma_rel | sigma_gamma (NOMINAL) | sigma_gamma_effective | sigma_delta_effective | tc | spread_rel mean | seed std | CI-excl-0 frac | t_ex mean | clipped_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot | decision | 0.09289 | 0.004047 | 0.004511 | 0.0333 | 0.0 | +1.0082 | 0.0145 | 1.00 | 0.1978 | 0.0633 |
| pilot | decision | 0.09289 | 0.004047 | 0.004511 | 0.0333 | 0.01 | +0.3754 | 0.0090 | 1.00 | 0.1978 | 0.0633 |
| pilot | decision | 0.09289 | 0.004047 | 0.004511 | 0.0333 | 0.02 | +0.1597 | 0.0046 | 1.00 | 0.1978 | 0.0633 |

`sigma_gamma` is NOMINAL: the field is calibrated to it BEFORE the [-0.05, 1.05] delta clip. `sigma_gamma_effective` (gamma units: std of d/dS of the delivered post-clip delta error) and `sigma_delta_effective` (delta units: std of that error) are what was actually DELIVERED. sigma_gamma_pilot is a gamma rmse, so the pilot point is compared against `sigma_gamma_effective` — comparing it against the delta-error std would be a units error (contract `effective_sigma_reporting.compare_pilot_against`).

`clipped_frac` is the fraction of delta evaluations on which the [-0.05, 1.05] delta clip BOUND. The field amplitude is calibrated on the UNCLIPPED field, and the clip does NOT attenuate that corruption — it SATURATES it. Where the clip binds, the corrupted hedger holds a position that is FLAT in S, so its gamma error there is the ORACLE's own -Gamma: a small calibrated gamma error is REPLACED by a larger uncalibrated one. Measured across the contract's DECISION rungs (AM3-1) the DELIVERED gamma scale EXCEEDS its nominal label — 1.21x at sigma_rel 0.10, 1.41x at 0.15, 1.54x at 0.20 — and only falls below it once the clip binds nearly everywhere (0.80 and above), outside any rung this contract decides on. A binding clip therefore does NOT make the gate conservative: comparing the pilot against the NOMINAL sigma is ANTI-conservative in exactly that band. A value of 0 means the sigma axis is exact; otherwise read the delivered scale off `sigma_gamma_effective` below rather than assuming its direction. The region of validity exists not because the mapping is biased in a known direction but because a saturated, bang-bang hedger is a STRUCTURALLY DIFFERENT OBJECT from the smooth-Greek-error PINN the gate stands in for (contract region_of_validity.interpretation).

## Pilot point vs the DELIVERED corruption

- sigma_gamma_pilot (NOMINAL target): **0.00404684**
- compared against `sigma_gamma_effective` (contract `oracle_headroom_gate.effective_sigma_reporting.compare_pilot_against`): **0.00451082**
- sigma_delta_effective (delta units, NOT the comparison target): 0.033303
- clipped_frac on the pilot arm: 0.0633 (region_of_validity.clipped_frac_max = 0.25)
- region of validity: **INSIDE** — the spread maps back to a sigma

## DECISION (per tc tier)

Smallest DECISION-ELIGIBLE sigma_rel with mean spread_rel >= 0.1 (the
pre-registered relative CVaR95 threshold, contract
`oracle_headroom_gate.spread_threshold_rel`) AND the paired per-path
bootstrap 95% CI excluding 0 in every seed. DIAGNOSTIC rungs (none) are EXCLUDED from this scan by the contract, however large their
spread: above `region_of_validity.clipped_frac_max` the spread is no
longer a monotone reading of a gamma error of the labelled size.

- tc = 0.0: sigma_rel = 0.09289 (sigma_gamma = 0.004047)
- tc = 0.01: sigma_rel = 0.09289 (sigma_gamma = 0.004047)
- tc = 0.02: sigma_rel = 0.09289 (sigma_gamma = 0.004047)

The three readings are DISTINCT and must stay distinct: a cleared rung,
`NONE` (no swept arm cleared the threshold — a no-go reading), and
`INCONCLUSIVE` (the pilot sits outside `region_of_validity`, so the
measurement cannot be mapped back to a sigma at all: neither a pass nor
a no-go, and it authorizes no training spend).

The go/no-go is a HUMAN decision, not this script's (contract
`oracle_headroom_gate.decision_rule`). If plausible pilot-fit
sigma_gamma sits below the passing level, RETUNE before M4 — levers
in order: (1) rebalancing frequency, (2) misspecification severity.
The gate runs BEFORE all training (runs_before: all_training).

Full-size gate command (human-launched):

    python gate_headroom.py --mode field --out-dir results/gate

Pilot-calibrated point once the pilot fit exists — read the float from
the runlog rather than retyping it (`train.py --pilot` also prints a
deliberately-reproduced PRE-FIX value, which must never reach the gate):

    python gate_headroom.py --sigma-gamma-from-runlog <run>/runlog.json
