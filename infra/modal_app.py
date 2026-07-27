"""infra/modal_app.py — Modal GPU dispatch for the v6 training grid.

One remote function shells train.py on an L40S per (arm, seed); a local driver
maps the roadmap arm table over it, retries a failed run once, and pulls
best.pt + runlog.json back to results/grid/<arm>/s<seed>/.

MONEY SPEND IS HUMAN-APPROVED (CLAUDE.md autonomy dial): dispatch_grid is
DRY-RUN by default — it prints the dispatch plan and a GPU-hr x rate cost
estimate and exits. Actual dispatch happens ONLY with --launch (CLI) /
launch=True (API), which is the point where `modal` is first imported. When
modal is not installed the dry run still works: the import is lazy, confined to
_build_modal_app(), so nothing at module scope touches it.

No secrets in code: Modal reads its token from the environment / .env only.

    python infra/modal_app.py                 # dry run: plan + cost, no spend
    python infra/modal_app.py --launch \
        --data data/frozen/train_val_labels.npz --lambdas lambdas_selected.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# roadmap arm table — rung0/std + rung3 at 10 seeds (the confirmatory contrast
# arms), everything else at the default 5. Seeds are global_seed + 0..(n-1)
# (contract meta.global_seed / seeds_confirmatory_cell / thresholds.seeds.default).
# ---------------------------------------------------------------------------
HIGH_SEED_ARMS = ("rung0_price_only", "standard_pinn", "rung3_delta_gamma_vega")
GRID_ARMS = (
    # supervision ladder (A1)
    "rung0_price_only", "rung1_delta", "rung2_delta_gamma", "rung3_delta_gamma_vega",
    # price-only / matched-info controls
    "standard_pinn", "info_matched_baseline",
    # residual x supervision factorial off-cell + gradient-penalty arm
    "sobolev_sans_pde", "gradient_penalty_only",
    # A6 gamma-label-noise dose-response (sigma_000 == rung3, not re-trained)
    "sigma_010", "sigma_025", "sigma_050", "bs_gamma", "shuffled",
)
BASE_SEED = 42            # contract meta.global_seed
N_SEEDS_HIGH = 10         # meta.seeds_confirmatory_cell
N_SEEDS_DEFAULT = 5       # acceptance_thresholds.seeds.default

# ---------------------------------------------------------------------------
# cost model (all overridable from the CLI; L40S rate is an estimate, not a bill)
# ---------------------------------------------------------------------------
GPU_TYPE = "L40S"
MODAL_L40S_USD_PER_HR = 1.95     # Modal on-demand L40S list price; override --rate
EST_MINUTES_PER_RUN = 25.0       # wall-clock budget for one matched-epochs fit; --minutes
TIMEOUT_S = 2 * 60 * 60          # per-run hard cap on the remote function

_DEFAULT_DATA = "data/frozen/train_val_labels.npz"
_DEFAULT_LAMBDAS = "lambdas_selected.yaml"
_DEFAULT_OUT = "results/grid"

# ---------------------------------------------------------------------------
# grid + cost (pure; no modal, no side effects)
# ---------------------------------------------------------------------------

def seeds_for(arm: str) -> list[int]:
    """global_seed + 0..(n-1); n = 10 for the confirmatory arms, else 5."""
    n = N_SEEDS_HIGH if arm in HIGH_SEED_ARMS else N_SEEDS_DEFAULT
    return [BASE_SEED + i for i in range(n)]


def build_grid(arms: tuple[str, ...] | list[str] = GRID_ARMS) -> list[tuple[str, int]]:
    """The full (arm, seed) dispatch list for `arms` (default = roadmap table)."""
    return [(arm, seed) for arm in arms for seed in seeds_for(arm)]


def estimate_cost(grid: list[tuple[str, int]],
                  minutes_per_run: float = EST_MINUTES_PER_RUN,
                  usd_per_hr: float = MODAL_L40S_USD_PER_HR) -> dict:
    """GPU-hr x rate estimate for a grid (one GPU per run, no overlap credit)."""
    gpu_hours = len(grid) * minutes_per_run / 60.0
    return {"n_runs": len(grid), "gpu_hours": gpu_hours,
            "usd_per_hr": usd_per_hr, "usd": gpu_hours * usd_per_hr,
            "minutes_per_run": minutes_per_run, "gpu_type": GPU_TYPE}


def format_plan(grid: list[tuple[str, int]], cost: dict) -> str:
    """Human-readable dispatch plan + cost line (what the dry run prints)."""
    by_arm: dict[str, list[int]] = {}
    for arm, seed in grid:
        by_arm.setdefault(arm, []).append(seed)
    lines = [f"[dispatch_grid] DRY RUN — {cost['n_runs']} runs on {cost['gpu_type']} "
             f"({N_SEEDS_HIGH} seeds for {'/'.join(HIGH_SEED_ARMS)}; "
             f"{N_SEEDS_DEFAULT} otherwise)"]
    for arm, seeds in by_arm.items():
        lines.append(f"  {arm:<26} {len(seeds):>2} seeds  "
                     f"{seeds[0]}..{seeds[-1]}")
    lines.append(f"[cost] {cost['gpu_hours']:.1f} GPU-hr x ${cost['usd_per_hr']:.2f}/hr "
                 f"= ${cost['usd']:.2f}  (est {cost['minutes_per_run']:.0f} min/run)")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# remote body — shells train.py; returns artifacts as bytes so the driver can
# pull them back without a shared volume. Kept module-level (no modal) so it is
# importable and unit-testable; _build_modal_app just wraps it in @app.function.
# ---------------------------------------------------------------------------

def _train_remote_body(arm: str, seed: int, data_ref: str, lambdas_ref: str,
                       repo_dir: str = "/repo", device: str = "cuda") -> dict:
    """Run one matched-epochs fit via train.py and return its artifacts.

    Returns {arm, seed, status, returncode, stderr_tail, [best_pt], [runlog_json]}.
    status == 'ok' iff train.py exited 0 AND wrote best.pt.
    """
    workdir = Path("/tmp/grid") / arm / f"s{seed}"
    cmd = [sys.executable, "train.py", "--arm", arm, "--seed", str(seed),
           "--data", data_ref, "--lambdas", lambdas_ref,
           "--out", str(workdir), "--device", device, "--matched-epochs"]
    proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    best, runlog = workdir / "best.pt", workdir / "runlog.json"
    result = {"arm": arm, "seed": int(seed), "returncode": proc.returncode,
              "stderr_tail": proc.stderr[-2000:],
              "status": "ok" if proc.returncode == 0 and best.exists() else "failed"}
    if best.exists():
        result["best_pt"] = best.read_bytes()
    if runlog.exists():
        result["runlog_json"] = runlog.read_text()
    return result


def _write_artifacts(out_root: str, result: dict) -> str | None:
    """Persist a remote result's best.pt + runlog.json under out_root/<arm>/s<seed>/."""
    dest = Path(out_root) / result["arm"] / f"s{result['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    if "best_pt" in result:
        (dest / "best.pt").write_bytes(result["best_pt"])
    if "runlog_json" in result:
        (dest / "runlog.json").write_text(result["runlog_json"])
    return str(dest) if ("best_pt" in result or "runlog_json" in result) else None

# ---------------------------------------------------------------------------
# modal wiring (LAZY: `import modal` lives here and nowhere else)
# ---------------------------------------------------------------------------

def _build_modal_app():
    """Construct the Modal App + train_remote function. Imports modal on call."""
    import modal

    app = modal.App("heston-v6-grid")
    repo = Path(__file__).resolve().parent.parent
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch", "numpy", "pyyaml", "scipy")
             .add_local_dir(str(repo), remote_path="/repo"))

    @app.function(gpu=GPU_TYPE, timeout=TIMEOUT_S, image=image)
    def train_remote(arm: str, seed: int, data_ref: str, lambdas_ref: str) -> dict:
        return _train_remote_body(arm, seed, data_ref, lambdas_ref)

    return app, train_remote


def _safe_get(handle, key: tuple[str, int]) -> dict:
    """Block on a spawned call; convert any raised error into a failed result."""
    try:
        return handle.get()
    except Exception as exc:                                    # noqa: BLE001
        return {"arm": key[0], "seed": key[1], "status": "failed",
                "returncode": None, "stderr_tail": f"{type(exc).__name__}: {exc}"}


def _launch(grid: list[tuple[str, int]], data_ref: str, lambdas_ref: str,
            out_root: str, cost: dict) -> dict:
    """Dispatch the grid on Modal, retry each failure once, pull artifacts back."""
    app, train_remote = _build_modal_app()
    gathered: dict[tuple[str, int], dict] = {}
    with app.run():
        handles = {key: train_remote.spawn(key[0], key[1], data_ref, lambdas_ref)
                   for key in grid}
        for key, h in handles.items():
            gathered[key] = _safe_get(h, key)
        retry = [key for key, r in gathered.items() if r.get("status") != "ok"]
        if retry:
            print(f"[dispatch_grid] retrying {len(retry)} failed run(s) once")
            rehandles = {key: train_remote.spawn(key[0], key[1], data_ref, lambdas_ref)
                         for key in retry}
            for key, h in rehandles.items():
                gathered[key] = _safe_get(h, key)
        for key in grid:
            _write_artifacts(out_root, gathered[key])
    ok = [k for k, r in gathered.items() if r.get("status") == "ok"]
    failed = [k for k, r in gathered.items() if r.get("status") != "ok"]
    print(f"[dispatch_grid] done: {len(ok)} ok, {len(failed)} failed "
          f"-> {out_root}/<arm>/s<seed>/")
    if failed:
        print("  failed: " + ", ".join(f"{a}/s{s}" for a, s in failed))
    return {"launched": True, "cost": cost, "out_root": out_root,
            "ok": ok, "failed": failed,
            "results": {f"{a}/s{s}": gathered[(a, s)].get("status") for a, s in grid}}

# ---------------------------------------------------------------------------
# driver + CLI
# ---------------------------------------------------------------------------

def dispatch_grid(grid: list[tuple[str, int]] | None = None, *, launch: bool = False,
                  data_ref: str = _DEFAULT_DATA, lambdas_ref: str = _DEFAULT_LAMBDAS,
                  out_root: str = _DEFAULT_OUT,
                  minutes_per_run: float = EST_MINUTES_PER_RUN,
                  usd_per_hr: float = MODAL_L40S_USD_PER_HR) -> dict:
    """Map the grid over train_remote. DRY-RUN unless launch=True.

    Dry run prints the plan + GPU-hr x rate estimate and returns without ever
    importing modal; launch=True dispatches, retries failures once, and pulls
    best.pt + runlog.json back to out_root/<arm>/s<seed>/.
    """
    grid = build_grid() if grid is None else list(grid)
    cost = estimate_cost(grid, minutes_per_run, usd_per_hr)
    print(format_plan(grid, cost))
    if not launch:
        print("[dispatch_grid] dry run only; pass --launch to spend GPU "
              "(money = human approval per CLAUDE.md)")
        return {"launched": False, "grid": grid, "cost": cost}
    return _launch(grid, data_ref, lambdas_ref, out_root, cost)


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Modal GPU dispatch for the v6 training grid")
    ap.add_argument("--launch", action="store_true",
                    help="actually dispatch (spends money; default is dry run)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm subset (default = roadmap table)")
    ap.add_argument("--data", default=_DEFAULT_DATA, help="frozen train/val label artifact")
    ap.add_argument("--lambdas", default=_DEFAULT_LAMBDAS)
    ap.add_argument("--out-root", default=_DEFAULT_OUT)
    ap.add_argument("--rate", type=float, default=MODAL_L40S_USD_PER_HR,
                    help="USD per GPU-hr for the cost estimate")
    ap.add_argument("--minutes", type=float, default=EST_MINUTES_PER_RUN,
                    help="estimated wall-clock minutes per run")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = _build_cli().parse_args(argv)
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else GRID_ARMS
    grid = build_grid(arms)
    return dispatch_grid(grid, launch=args.launch, data_ref=args.data,
                         lambdas_ref=args.lambdas, out_root=args.out_root,
                         minutes_per_run=args.minutes, usd_per_hr=args.rate)


if __name__ == "__main__":
    main()
