"""R12 — is the oracle mask arm-independent? Verified by EXECUTION, not by reading.

The property that protects the confirmatory contrast is: eval_greeks scores every
arm on the SAME set of grid points. This script proves it by running the real
eval_greeks.eval_arm_on_regime on four deliberately different networks and
capturing, from inside each call, the exact input tensor the model was queried on.

The four models: standard_pinn @ init seed 0, standard_pinn @ init seed 1,
rung3_delta_gamma_vega @ init seed 2 (a DIFFERENT arm config), and that same
rung3 model with every weight scaled x50 (a wildly different function, so the
comparison cannot be vacuously true because the models happen to agree).

Note on the seeds: arms differ only by LOSS FLAGS, so two arms built at the same
torch seed are bit-identical networks (contract: "identical architecture,
identical ansatz across arms"). The rung3 model is therefore given its own init
seed, otherwise the "models genuinely differ" guard would fire on that identity
rather than on anything about the mask.

Checks:
  1. the captured (S, K, tau, kappa, theta, xi, rho, v0) tensors are BITWISE equal
     across all four models, on the full grid and on each holdout slice;
  2. n_unmasked is identical;
  3. the models genuinely differ (rmse differs), so (1) is a real invariance;
  4. the selection is a pure function of the artifact: it equals
     flatnonzero(~mask_any & restrict) recomputed independently from the npz.

Usage: python audit/repro/r12_mask_arm_independence.py <anchors_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_greeks as EG
from SobolevPINN import SobolevPINN, load_arm

CONTRACT = "heston_benchmark_v6.yaml"
PINN_CFG = "pinn_config.yaml"


def make_model(arm: str, seed: int, scale: float = 1.0) -> SobolevPINN:
    torch.manual_seed(seed)
    m = SobolevPINN(load_arm(PINN_CFG, arm))
    if scale != 1.0:
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(scale)
    m.eval()
    return m


def instrument(model):
    """Capture the input tensor eval_arm_on_regime queries the model with."""
    box = {}
    orig = model.greeks_eval

    def wrapped(x, *a, **kw):
        box["x"] = x.detach().clone()
        return orig(x, *a, **kw)

    model.greeks_eval = wrapped
    return box


def main(anchors_dir: str) -> None:
    contract = yaml.safe_load(open(CONTRACT))
    regimes = list(contract["splits"]["heldout_greek_and_hedging"])
    models = {
        "standard_pinn/init0": make_model("standard_pinn", 0),
        "standard_pinn/init1": make_model("standard_pinn", 1),
        "rung3/init2": make_model("rung3_delta_gamma_vega", 2),
        "rung3/init2 x50": make_model("rung3_delta_gamma_vega", 2, scale=50.0),
    }
    boxes = {k: instrument(m) for k, m in models.items()}

    all_ok = True
    for regime in regimes:
        npz = str(Path(anchors_dir) / f"{regime}_grid.npz")
        slices = {"full": None, **EG._slice_masks(npz)}
        for slname, restrict in slices.items():
            res, xs = {}, {}
            for k, m in models.items():
                res[k] = EG.eval_arm_on_regime(m, npz, regime, restrict=restrict)
                xs[k] = boxes[k]["x"]
            ref_k = next(iter(models))
            ref = xs[ref_k]
            same_x = all(torch.equal(ref, xs[k]) for k in models)
            same_n = len({res[k]["gamma"]["n_unmasked"] for k in models}) == 1
            rmses = {k: res[k]["gamma"]["rmse"] for k in models}
            differ = len({round(v, 12) for v in rmses.values()}) == len(models)

            d = np.load(npz)
            keep = ~np.asarray(d["mask_any"], bool).ravel()
            sel = keep & (np.asarray(restrict, bool).ravel() if restrict is not None else True)
            idx = np.flatnonzero(sel)
            S_ax, K_ax, T_ax = (np.asarray(d[f"{a}_axis"], float) for a in ("S", "K", "tau"))
            Sg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")[0]
            expect_S = Sg.ravel()[idx]
            got_S = ref[:, 0].to(torch.float64).numpy()
            pure = (got_S.size == expect_S.size
                    and np.allclose(got_S, expect_S, rtol=0, atol=1e-5))

            ok = same_x and same_n and differ and pure
            all_ok &= ok
            print(f"{regime:>16s} / {slname:<5s}: n_unmasked="
                  f"{res[ref_k]['gamma']['n_unmasked']:>6d}  "
                  f"identical_inputs={same_x}  identical_n={same_n}  "
                  f"models_differ={differ}  equals_~mask_any&restrict={pure}  "
                  f"-> {'OK' if ok else 'FAIL'}")
            print("      gamma rmse per model: " +
                  ", ".join(f"{k}={v:.4g}" for k, v in rmses.items()))

    print(f"\nARM-INDEPENDENCE: {'CONFIRMED' if all_ok else 'VIOLATED'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
