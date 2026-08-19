"""Contract `lambda_selection.robustness_row` — the unmet registered commitment.

    "The confirmatory contrast (rung3 vs standard_pinn) is ADDITIONALLY reported at the
     rung3-sourced lambda_pde, so the sourcing choice can be shown to be immaterial to
     the headline result (or shown not to be)."
     status: robustness_result_not_a_second_confirmatory_test

Stage 1 of the registered staged protocol, re-run with the source arm overridden to
rung3. The CONTRACT IS NOT TOUCHED: `lambda_pde.source_arm` still reads standard_pinn and
still governs the headline. This is the additionally-reported row the contract asks for,
and per its own `status` it is NOT a second confirmatory test and cannot replace one.

Everything else is held identical to the registered stage 1: same candidates, same
LockedTestSet guard, same validation-only scoring, same step budget, same seed.
"""
# repo-local module; run from the repo root
import json
from dataclasses import replace

import yaml
from SobolevPINN import load_arm
from train_pinn import ArmDataset, LockedTestSet, TrainConfig, select_lambdas, train_model
import train as T

CONTRACT, PINN_CFG = "heston_benchmark_v6.yaml", "pinn_config.yaml"
DATA = "data/frozen/v6-labels-20260812/train_val/train_val_labels.npz"
ANCHORS = "data/frozen/v6-labels-20260812/anchors"
SEED, STEPS, DEVICE = 42, 20000, "cpu"

lam_sel = yaml.safe_load(open(CONTRACT))["lambda_selection"]
registered_arm = str(lam_sel["lambda_pde"]["source_arm"])
robustness_arm = str(lam_sel["lambda_gamma"]["source_arm"])       # rung3
cands = [float(x) for x in lam_sel["lambda_pde"]["candidates"]]
print(f"registered source arm : {registered_arm}")
print(f"robustness source arm : {robustness_arm}")
print(f"candidates            : {cands}", flush=True)

tcfg = replace(TrainConfig.from_dict(yaml.safe_load(open(PINN_CFG)).get("training", {})),
               steps=STEPS)
ranges, anchors, feller_min, excise = T.pde_sampling_spec(CONTRACT, PINN_CFG)
guard = LockedTestSet(ANCHORS)

cfg = load_arm(PINN_CFG, robustness_arm)
train_ds = ArmDataset(DATA, cfg, "train", seed=SEED)
val_ds = ArmDataset(DATA, cfg, "val", seed=SEED)


def score_pde(lp, _lg, _lv):
    model, best_state, _, _ = train_model(
        replace(cfg, lambda_pde=lp), train_ds, val_ds, tcfg, SEED, device=DEVICE,
        pde_ranges=ranges, pde_anchors=anchors, feller_min=feller_min,
        excise_rel_radius=excise, early_stop=False)
    model.load_state_dict(best_state)
    s = T._val_greek_score(model, val_ds, DEVICE)
    print(f"  lambda_pde={lp}: val score={s:.8f}", flush=True)
    return s


res = select_lambdas(cands, [1.0], [1.0], fit_and_val_score=score_pde, test_set=guard)
out = {"robustness_row": True,
       "lambda_pde_rung3_sourced": float(res["lambda_pde"]),
       "lambda_pde_registered": None,
       "source_arm": robustness_arm,
       "registered_source_arm": registered_arm,
       "candidates": cands, "seed": SEED, "steps": STEPS,
       "status": "robustness_result_not_a_second_confirmatory_test"}
out["lambda_pde_registered"] = float(yaml.safe_load(open("lambdas_selected.yaml"))["lambda_pde"])
json.dump(out, open("results/robustness_row_lambda_pde.json", "w"), indent=2)
print(json.dumps(out, indent=2))
