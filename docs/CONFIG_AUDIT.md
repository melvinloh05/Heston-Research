# CONFIG_AUDIT — pinn_config.yaml & hedging_config.yaml vs heston_benchmark_v6.yaml

Audit date: 2026-07-13. **Diff in §5 approved and applied 2026-07-13**; tests extended per §6 and
`python -m pytest -q` passes (21/21). Contract (`heston_benchmark_v6.yaml`) is the single source of truth and
READ-ONLY; where it and a config disagree, the config changes. Verdicts below are derived by
reading the code that actually consumes each key: `SobolevPINN.load_arm` / `PINNConfig`
(pinn_config.yaml) and `Hedging_backtest.resolve_config` / `_run_sweep` (hedging_config.yaml).

**Sourcing rule used throughout.** `_run_sweep` reads some values straight from the contract
(`bm[...]`) and some from the engine YAML (`eng[...]`). A contract value that the engine already
reads from `bm` (transaction-cost tiers, Bates sweep, Merton params, `train_params`,
`perturbations`) must **NOT** be duplicated into `hedging_config.yaml` — the header of that file
says "contract has none of these", and a second copy is a live divergence hazard. Keys the task
asks to *stage for P5* (`horizon.T_prime`, `pnl_convention`, `risk.loss_definition`) are **not yet
read by any engine code**, so staging them in the engine supplement is safe and is where the P5
engine will look.

---

## 1. pinn_config.yaml — arms (methods §143–164 of contract)

Arch is identical across arms (`shared:` block only; no arm overrides `n_layers`/`width`/
`activation`) → the identical-ansatz invariant holds. `status` cannot be a `PINNConfig` field
(`load_arm` does `PINNConfig(**d)`), so arm status/role is documented in comments — expected.

| Contract method (§methods) | pinn_config arm | current flags | verdict |
|---|---|---|---|
| `baseline0_feedforward` (use_pde:false, labels none, info-matched) | `feedforward` | `use_pde:false, use_bc:false`, no labels | **MATCH** |
| `baseline1_standard_pinn` (use_pde:true, labels none) | `standard_pinn` | `{}` → use_pde:true (default), no labels | **MATCH** |
| `ladder_rung0_price` (use_pde:true, no labels) | `rung0_price_only` | `{}` | **MATCH** |
| `ladder_rung1_delta` (D) | `rung1_delta` | `supervise_delta` | **MATCH** |
| `ladder_rung2_delta_gamma` (D,G; **V off**) | `rung2_delta_gamma` | D on, G on, **V off** | **MATCH** |
| `ladder_rung3_dgv` = `sobolev_pinn` (MAIN, D,G,V) | `rung3_delta_gamma_vega` | D,G,V | **MATCH** |
| `sobolev_sans_pde` (use_pde:false OR λ_pde:0; D,G,V) | `sobolev_sans_pde` | `<<rung3, use_pde:false` | **MATCH** |
| `lambda_pde_zero` *if kept separate, document as identical to sobolev_sans_pde* | `lambda_pde_zero` | `<<rung3, lambda_pde:0.0` — comment says **"Distinct from lambda_pde_zero"** | **MISMATCH (doc)** |
| dose `sigma_000` (oracle, σ=0, D+V labels TRUE) | `sigma_000` | `*rung3` (label_source=oracle default, σ=0) | **MATCH** |
| dose `sigma_010/025/050` | `sigma_010/025/050` | `<<rung3, gamma_label_noise_sigma:0.10/0.25/0.50` | **MATCH** |
| dose `shuffled` | `shuffled` | `<<rung3, label_source:shuffled` | **MATCH** |
| dose `bs_gamma` | `bs_gamma` | `<<rung3, label_source:bs_gamma` | **MATCH** |
| `gradient_penalty_only` (gp:true, label_source:none) | `gradient_penalty_only` | `label_source:none, gradient_penalty:true` (does NOT inherit rung3 — correct, else `__post_init__` asserts) | **MATCH** |
| `info_matched_baseline` | `info_matched_baseline` | `{}` (standard_pinn flags; swept by info_matching) | **MATCH** |
| `optional_vanna_arm` (supervise_vanna:true, λ_vanna>0, status schedule_permitting) | `optional_vanna_arm` | `<<rung3, supervise_vanna:true, lambda_vanna:1.0`; status in comment | **MATCH** |
| Gamma-only view (= rung2, measures Vega marginal) | `gamma_only` | `*rung2` | **MATCH** (extra alias, fine) |

**Only defect: `lambda_pde_zero` documentation.** Per the loss gate
`if cfg.use_pde and cfg.lambda_pde != 0.0` (SobolevPINN.py:209), `sobolev_sans_pde` (use_pde
false) and `lambda_pde_zero` (use_pde true, λ_pde 0) **both** skip the PDE term and produce an
**identical loss dict** (no `"pde"` key). The current comment calls them "Distinct", contradicting
the contract instruction that a separately-kept `lambda_pde_zero` be documented as identical. Fix
is a comment-only change (they stay separate arms for provenance: factorial-OFF cell vs Sakuma
twin-net A2 endpoint).

## 2. pinn_config.yaml — hypercube_sampling (contract §training_parameterization.sampling)

| Contract clause | key | current | verdict |
|---|---|---|---|
| `method: latin_hypercube` | `hypercube_sampling.method` | `latin_hypercube` | **MATCH** |
| `kappa [1.0, 4.0]` | ranges.kappa | `[1.0, 4.0]` | **MATCH** |
| `theta [0.02, 0.12]` | ranges.theta | `[0.02, 0.12]` | **MATCH** |
| `xi [0.20, 0.60]` | ranges.xi | `[0.20, 0.60]` | **MATCH** |
| `rho [-0.80, -0.20]` | ranges.rho | `[-0.80, -0.20]` | **MATCH** |
| `v0 [0.01, 0.12]` | ranges.v0 | `[0.01, 0.12]` | **MATCH** |
| Feller reject `< 0.40` | `feller_min` | `0.40` | **MATCH** |

## 3. pinn_config.yaml — shared / ansatz

| Contract clause | key | current | verdict |
|---|---|---|---|
| inputs `[S,K,tau,kappa,theta,xi,rho,v0]` | `shared.inputs` | identical | **MATCH** |
| identical arch across arms (no per-arm n_layers/width/activation) | `shared` only | no arm overrides these | **MATCH** |

**pinn_config.yaml summary: 1 MISMATCH (doc-only, `lambda_pde_zero`), 0 MISSING.**

---

## 4. hedging_config.yaml — engine block

Read by `resolve_config` + `_run_sweep`. Contract-owned values that `_run_sweep` reads from `bm`
are marked "MATCH (contract-owned)" and are **deliberately absent** from hedging_config.

| Contract clause | hedging_config key | current | verdict |
|---|---|---|---|
| instrument S0=100 (§hedging_simulation.instrument) | `contract.S0` | `100.0` | **MATCH** |
| instrument K=100 | `contract.K` | `100.0` | **MATCH** |
| instrument tau0=0.25 (engine `contract.T` is the tau0 fallback) | `contract.T` | **`0.5`** | **MISMATCH** |
| hedge horizon T′=0.17 (§horizon; engine reads from P5) | `horizon.T_prime` | **absent** | **MISSING** |
| rebalancing daily, freq 252 (dt≈0.003968) | `rebalancing.frequency_per_year` | `252` | **MATCH** |
| rebalancing fixed_across_arms | `rebalancing` comment | "FIXED — …varied lever" | **MATCH** (documented; wording strengthened in diff) |
| paths n_paths_per_cell = 10000 (production) | `simulation.n_paths` | **`20000`** | **MISMATCH** |
| smoke override documented | — | no explicit note | **MISSING (doc)** |
| TC tiers `[0.00,0.01,0.02]` | read from `bm.hedging_simulation.transaction_costs.tiers` (line 498) | contract value | **MATCH (contract-owned — do NOT duplicate)** |
| xi_up: xi 0.30→0.45 ⇒ shift +0.15 | `misspecification.directions.xi_up.shift_at_m1` | `0.15` | **MATCH** |
| rho_down: rho −0.50→−0.80 ⇒ shift −0.30 | `…rho_down.shift_at_m1` | `-0.30` | **MATCH** |
| combined = both legs | `…combined.legs` | `[xi_up, rho_down]` | **MATCH** |
| magnitudes `[0.0, 0.5, 1.0]` | `misspecification.magnitudes` | **`[0.0, 0.25, 0.5, 0.75, 1.0]`** | **MISMATCH** |
| Bates sweep λ_j`[0,0.1,0.25,0.5]`, μ_j −0.10, σ_j`[0.05,0.10,0.15]` + "λ_j rows first; σ_j grid only where gap visible" | read from `bm.models.bates.jump_severity_sweep` (line 532); "σ_j only when λ>0" encoded at line 534 | contract value + engine behavior | **MATCH (contract-owned + engine-encoded)** |
| Merton σ0.20 λ_j0.25 μ_j−0.10 σ_j0.10 | read from `bm.models.merton.params` (line 549) | contract value | **MATCH (contract-owned)** |
| smoothing/no-trade-band on standard-PINN | `smoothing_baseline.applies_to` | `[standard_pinn]` | **MATCH** |
| pnl_convention.initial_premium = θ_train oracle for all arms (P5) | `pnl_convention.initial_premium` | **absent** | **MISSING** |
| pnl_convention.terminal_mark = true-DGP price at T′ (P5) | `pnl_convention.terminal_mark` | **absent** | **MISSING** |
| CVaR level 0.95 | `risk.cvar_level` | `0.95` | **MATCH** |
| loss definition "−PnL" | `risk.loss_definition` | **absent** | **MISSING** |
| seeds from benchmark meta; hedging_config must NOT redefine | (none) | correctly absent (`resolve_config` derives) | **MATCH** |

**hedging_config.yaml summary: 3 MISMATCH (`contract.T`, `simulation.n_paths`,
`misspecification.magnitudes`), 4 MISSING (`horizon.T_prime`, `pnl_convention.initial_premium`,
`pnl_convention.terminal_mark`, `risk.loss_definition`) + 1 doc note (smoke override).**

> Note on `contract.T`: `_run_sweep` overrides T with instrument `tau0`=0.25 (contract wins,
> line 503–506), so production already runs at 0.25; but the hedging *tests* read
> `eng["contract"]["T"]` directly in `_setup`, so today they run at 0.5. Setting it to 0.25
> aligns tests with production and removes the misleading value. (One test comment "T = 0.5" will
> be updated; its assertions are magnitude-relative and still hold.)

---

## 5. Proposed minimal diff (awaiting approval — config edits need sign-off)

### pinn_config.yaml (doc-only)
```diff
   # --- v6 completed factorial, 4th cell (DML-no-PDE): rung3 labels, NO residual machinery.
-  #     Distinct from lambda_pde_zero: sobolev_sans_pde has use_pde false (pure factorial
-  #     OFF cell); lambda_pde_zero keeps use_pde true at weight 0.0 (Sakuma twin-net, A2 sweep).
+  #     use_pde:false. Loss is IDENTICAL to lambda_pde_zero below — the PDE term is skipped when
+  #     (not use_pde) OR (lambda_pde==0.0), so both arms emit the same loss dict (no "pde" term).
+  #     Kept as separate arms only for provenance: sobolev_sans_pde = factorial OFF cell;
+  #     lambda_pde_zero = Sakuma twin-net A2 sweep endpoint.
   sobolev_sans_pde:        {<<: *rung3, use_pde: false}
   ...
-  # --- A2 PDE-weight zero point (valid, tested config) ---
+  # --- A2 PDE-weight zero point (Sakuma twin-net); loss identical to sobolev_sans_pde above ---
   lambda_pde_zero:         {<<: *rung3, lambda_pde: 0.0}
```

### hedging_config.yaml
```diff
 contract:                      # the hedged instrument
   S0: 100.0
   K: 100.0
-  T: 0.5                       # within contract tau grid [0.04, 1.0]
+  T: 0.25                      # = contract instrument tau0 (option initial maturity)
   option_type: european_call

+horizon:                       # STAGED for P5: engine will hedge to T' < tau0, liquidate at T'
+  T_prime: 0.17                # A&T construction (contract hedging_simulation.horizon.T_prime)
+
 simulation:
-  n_paths: 20000
+  n_paths: 10000               # production per-cell (contract paths.n_paths_per_cell);
+                               # smoke/tests override via _cfg(n_paths=...) in test_hedging_backtest.py
   scheme: qe_andersen
   psi_c: 1.5                   # QE quadratic/exponential switching threshold

 rebalancing:
-  frequency_per_year: 252      # FIXED — Greek accuracy is the only varied lever
+  frequency_per_year: 252      # daily (dt=1/252≈0.003968); FIXED ACROSS ARMS — Greek accuracy is the only varied lever
   charge_final_unwind: true
 ...
 misspecification:              # hedge-params = train-params + m * shift_at_m1
-  # magnitude=1.0 == v6 confirmatory endpoint (xi 0.45 / rho -0.80); magnitudes
-  # in (0,1) interpolate the severity sweep. PROVISIONAL pending contract
-  # open-item-4 sign-off — do NOT mark final.
-  magnitudes: [0.0, 0.25, 0.5, 0.75, 1.0]
+  # magnitude=1.0 lands EXACTLY on the v6 confirmatory endpoint (xi 0.45 / rho -0.80);
+  # 0.5 is the mid severity point. shift_at_m1 chosen so m=1 hits the contract targets.
+  magnitudes: [0.0, 0.5, 1.0]
   directions:
     xi_up:    {param: xi,  shift_at_m1: 0.15}    # xi  0.30 -> 0.45 at magnitude 1
     rho_down: {param: rho, shift_at_m1: -0.30}   # rho -0.50 -> -0.80 at magnitude 1
     combined: {legs: [xi_up, rho_down]}
 ...
 risk:
   cvar_level: 0.95
+  loss_definition: "-PnL"      # CVaR95 = mean worst 5% of L = -PnL (contract cvar_convention)
   bootstrap_B: 2000
   gap_floor: 1.0e-6            # |cvar_baseline - cvar_oracle| below this -> gap_closed undefined ("")
+
+pnl_convention:                # STAGED for P5 engine read; mirrors contract hedging_simulation.pnl_convention
+  initial_premium: theta_train_oracle_for_all_arms
+  terminal_mark: true_dgp_price_at_T_prime
```

### Explicitly NOT changed (contract-owned; duplicating would violate SoT)
- TC tiers, Bates sweep, Merton params, `train_params`, `perturbations` — engine reads these from
  `bm[...]`. They stay in the contract only.
- Seeds — derived in `resolve_config` from `bm.meta`; must stay out of hedging_config.

## 6. Post-approval test plan
- `test_sobolev_pinn.py::test_config_coverage…`: assert (a) every contract method above maps to an
  existing arm that builds + trains, (b) `sobolev_sans_pde` loss dict has no `"pde"` term,
  (c) `rung2_delta_gamma` has vega OFF / gamma ON, (d) `optional_vanna_arm` has `supervise_vanna`
  and a finite loss.
- `test_hedging_backtest.py`: assert `resolve_config` exposes `engine.horizon.T_prime == 0.17`,
  the three directions with exact shifts (`xi_up +0.15`, `rho_down −0.30`, `combined` legs), and
  `benchmark…transaction_costs.tiers == [0.0, 0.01, 0.02]`. Update the stale `T = 0.5` comment.
- `python -m pytest -q` (all must pass).
