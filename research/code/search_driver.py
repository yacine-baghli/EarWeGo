"""
NESTED-CV SUCCESSIVE-HALVING SEARCH DRIVER for a model family.

WHAT THIS IS FOR. Every new family has hyperparameters. Tuning them on the outer
fold's validation ears and then reporting that same fold is the single easiest way to
manufacture a fake 0.05 mm gain -- and 0.05 mm is the size of the effects this
programme is chasing. So selection happens in a NESTED grouped CV: the driver splits
only the CURRENT OUTER FOLD'S TRAINING SUBJECTS into inner folds, and every trial is
trained and scored entirely inside them.

THE BOUNDARY, WHICH IS THE POINT OF THIS FILE
---------------------------------------------
This driver SELECTS. It NEVER REPORTS. It never trains on the outer fold's full
training set, never predicts an outer-fold validation ear, and never writes a file that
screen_compare.py or cv_verdict.py will pick up. It is not "the outer evaluation with
extra steps"; the two are separate runs by construction:

  * every trial is launched with TRAIN_EARS / VAL_EARS drawn from the inner split, which
    puts train_family.py in explicit-split mode: ALIAS is forced off, the output tag is
    an hpo_* name, and the report carries split_mode="explicit_inner";
  * the driver ASSERTS that no outer-fold validation SUBJECT appears anywhere in any
    inner split, and asserts every child report came back as explicit_inner;
  * the last thing the driver prints is the exact command to run the winning config
    once through the frozen outer fold. That command is a separate, deliberate step.

    FAMILY=dgcnn FOLD=0 K=12 E0=80 python3 search_driver.py
    python research/code/search_driver.py     # <- no FAMILY set: runs the smoke test

SUCCESSIVE HALVING. K configs are sampled by RANDOM SEARCH over the family's documented
SEARCH_SPACE grid (never exhaustive enumeration -- the grids are large and most axes are
inert). All K run at the rung-0 epoch budget; the best ceil(n/ETA) survive; the budget is
multiplied by ETA; repeat. Every config and every rung -- including the ones dropped and
the ones that crashed -- goes into one machine-readable JSON, so nothing is a silent cap.

Each rung is a FRESH run at its own budget, not a resumed checkpoint. The LR schedule is
cosine over EPOCHS, so a resumed 180-epoch score would not be comparable to a native
180-epoch score, and the whole point of the rungs is comparability with the final budget.
The cost is that rung-0 work is thrown away; the benefit is that the rung-2 number means
what it says.

WHAT THE INNER LOOP DELIBERATELY DOES NOT DO. The dense-SSM blend is OFF for every trial.
The per-fold SSM (dense_ssm_f<FOLD>.npz) is built from the OUTER fold's training ears,
which include the inner-loop validation ears -- using it to score an inner trial would
leak. Surface projection is fold-independent and can be enabled with HPO_FULL_EVAL=1;
selection on the raw number is the cheap default. Either way, the alpha/kuse of the blend
are NOT tunable here and must be fixed a priori or tuned in a separate nested pass.


ENVIRONMENT
-----------
  FAMILY        (required for a real run) key into train_family.REGISTRY
  FOLD      0   the OUTER fold whose TRAINING subjects are searched
  SEARCH_SEED 0 RNG for config sampling AND the inner split
  K         12  configs sampled at rung 0
  ETA       3   halving factor; each rung keeps ceil(n/ETA) and multiplies the budget
  E0        80  epoch budget at rung 0
  NRUNGS    3   number of rungs -> budgets E0 * ETA**r
  RUNGS     ""  explicit comma-separated budgets, overrides E0/ETA/NRUNGS for the ladder
  INNER_K   3   grouped inner folds inside the outer training subjects
  INNER_USE 1   inner folds each trial is scored on (always folds 0..INNER_USE-1, so
                every config and every rung is scored on the SAME ears). 1 is cheap and
                noisy; raising it costs linearly and is the honest fix for a search whose
                rung-0 ordering looks like a coin flip.
  TRIAL_SEED 0  SEED handed to every trial: held CONSTANT so configs are compared under
                the same initialisation stream, not against seed noise
  SELECT    raw ordered_MLE_mm | full -> ordered_MLE_full_mm (needs HPO_FULL_EVAL=1)
  RUNNER    subprocess  one fresh interpreter per trial, so RNG/CUDA/module state cannot
                carry between configs. `inproc` skips ~5 s of startup per trial but shares
                that state -- smoke tests and debugging only.
  HPO_FULL_EVAL 0  run surface projection inside trials (never the SSM blend)
  HPO_TTA   2   fresh samples averaged at each trial's final evaluation
  OUT       $WORK/hpo_<FAMILY>_o<FOLD>_s<SEARCH_SEED>.json
  SPACE_JSON ""  override the family's SEARCH_SPACE with this JSON object
  DRY       0   build and assert the splits, print the plan, launch nothing
  WORK/DATA/TRIS/... are inherited by every trial (see train_family.py)
"""
import os, sys, json, math, time, subprocess, contextlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import train_family as TF

TRAINER = os.path.join(_HERE, "train_family.py")


def sample_configs(space, k, rng):
    """Random search over the documented grid, de-duplicated."""
    keys = sorted(space)
    seen, out = set(), []
    for _ in range(400 * max(k, 1)):
        if len(out) >= k:
            break
        c = {kk: space[kk][rng.randint(len(space[kk]))] for kk in keys}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def inner_splits(subj, outer_val_subj, inner_k, seed):
    """Grouped inner folds over ONLY the outer fold's training subjects."""
    pool = np.setdiff1d(np.unique(subj), outer_val_subj)
    perm = np.random.RandomState(seed).permutation(pool)
    out = []
    for p in np.array_split(perm, inner_k):
        p = np.asarray(p)
        tr_s = np.setdiff1d(pool, p)
        out.append(dict(val_subj=p, train_subj=tr_s,
                        val=np.where(np.isin(subj, p))[0],
                        train=np.where(np.isin(subj, tr_s))[0]))
    return pool, out


def assert_nested(subj, outer_val_subj, outer_val_ears, pool, splits):
    """The one property that makes the whole search legitimate. Loud, not implicit."""
    ov = set(np.asarray(outer_val_subj).tolist())
    assert not (set(pool.tolist()) & ov), "searchable pool contains outer-fold val subjects"
    for i, s in enumerate(splits):
        ts, vs = set(s["train_subj"].tolist()), set(s["val_subj"].tolist())
        assert not (ts & vs), f"inner fold {i}: train and val share a SUBJECT"
        assert not ((ts | vs) & ov), \
            f"inner fold {i}: {len((ts | vs) & ov)} outer-fold val subject(s) leaked in"
        assert ts | vs == set(pool.tolist()), f"inner fold {i} does not cover the pool"
        ears = set(s["train"].tolist()) | set(s["val"].tolist())
        assert not (ears & outer_val_ears), f"inner fold {i}: outer-fold val EARS leaked in"
        assert all((subj[s["val"]] == k).sum() == 2 for k in vs), \
            f"inner fold {i}: a subject's two ears were split apart"


def _apply_env(env):
    """Make os.environ exactly `env`; return the previous mapping for _restore_env."""
    old = dict(os.environ)
    for k in list(os.environ):
        if k not in env:
            del os.environ[k]
    os.environ.update(env)
    return old


def _restore_env(old):
    for k in list(os.environ):
        if k not in old:
            del os.environ[k]
    os.environ.update(old)


def trial_env(family, cfg, budget, outer_fold, split, tag, work):
    """Every CFG_* is DROPPED and rebuilt from `cfg`. A stray CFG_LR exported in the
    caller's shell would otherwise override a swept axis in every single trial, and
    nothing downstream would show it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CFG_")}
    env.update(FAMILY=family, FOLD=str(outer_fold),
               SEED=str(os.environ.get("TRIAL_SEED", "0")),
               EPOCHS=str(budget), WORK=work, TAG=tag, ALIAS="0",
               VARIANT=f"{family}_hpo",
               TRAIN_EARS=json.dumps([int(x) for x in split["train"]]),
               VAL_EARS=json.dumps([int(x) for x in split["val"]]),
               FULL_EVAL=os.environ.get("HPO_FULL_EVAL", "0"),
               # the outer-fold SSM has seen the inner-val ears; point SSM at a name that
               # cannot exist so the blend is skipped and the report says why
               SSM=f"{work}/__no_ssm_during_inner_cv__.npz",
               TTA=os.environ.get("HPO_TTA", "2"))
    for k, v in cfg.items():
        env[f"CFG_{k.upper()}"] = json.dumps(v)
    return env


def run_trial(family, cfg, budget, outer_fold, inner_i, split, tag, work, select,
              runner="subprocess"):
    """One training run in explicit-split mode. Returns (score, info).

    runner="subprocess" (default): a fresh interpreter per trial, so torch/numpy RNG
    state, the CUDA context and any module-level state cannot carry between configs.
    runner="inproc": calls train_family.main() in this process. Skips ~5 s of interpreter
    and torch startup per trial and reloads DATA every time, but the trials then share
    RNG and device state -- smoke tests and debugging only, never a real search.
    """
    env = trial_env(family, cfg, budget, outer_fold, split, tag, work)
    lp = f"{work}/{tag}.log"
    t0 = time.time()
    if runner == "inproc":
        old = _apply_env(env)
        with open(lp, "w") as fh, contextlib.redirect_stdout(fh):
            TF.main()
        _restore_env(old)
        rc, err = 0, ""
    else:
        p = subprocess.run([sys.executable, "-u", TRAINER], env=env, capture_output=True,
                           text=True, cwd=os.getcwd())
        rc, err = p.returncode, (p.stderr or p.stdout or "")
        open(lp, "w").write(p.stdout + ("\n--- stderr ---\n" + p.stderr if p.stderr else ""))
    info = {"tag": tag, "budget": budget, "inner_fold": inner_i, "runner": runner,
            "wall_s": round(time.time() - t0, 1), "returncode": rc, "log": lp}
    jp = f"{work}/{tag}.json"
    if rc != 0 or not os.path.exists(jp):
        info["error"] = err[-1200:]
        return None, info
    j = json.load(open(jp))
    assert j["split_mode"] == "explicit_inner", \
        f"{tag} came back as split_mode={j['split_mode']!r} -- a trial ran an OUTER fold"
    assert set(j["val_ear_index"]) == set(int(x) for x in split["val"]), \
        f"{tag} validated on ears the driver did not ask for"
    key = "ordered_MLE_mm" if select == "raw" else "ordered_MLE_full_mm"
    info.update(runtime_s=j["runtime_s"], params=j["params"], score_key=key,
                ordered_MLE_mm=j["ordered_MLE_mm"],
                ordered_MLE_full_mm=j["ordered_MLE_full_mm"], report=jp)
    assert j[key] is not None, f"SELECT={select} but {tag} reported {key}=null"
    return float(j[key]), info


def main():
    FAMILY = os.environ["FAMILY"]
    FOLD = int(os.environ.get("FOLD", "0"))
    SS = int(os.environ.get("SEARCH_SEED", "0"))
    K = int(os.environ.get("K", "12"))
    ETA = int(os.environ.get("ETA", "3"))
    INNER_K = int(os.environ.get("INNER_K", "3"))
    INNER_USE = int(os.environ.get("INNER_USE", "1"))
    SELECT = os.environ.get("SELECT", "raw")
    RUNNER = os.environ.get("RUNNER", "subprocess")
    WORK = os.environ.get("WORK", "scratch")
    DATA = os.environ.get("DATA", f"{WORK}/screen_data_2048.npz")
    DRY = os.environ.get("DRY", "0") == "1"
    OUT = os.environ.get("OUT", f"{WORK}/hpo_{FAMILY}_o{FOLD}_s{SS}.json")
    if os.environ.get("RUNGS"):
        LADDER = [int(x) for x in os.environ["RUNGS"].split(",")]
    else:
        E0, NR = int(os.environ.get("E0", "80")), int(os.environ.get("NRUNGS", "3"))
        LADDER = [E0 * ETA ** r for r in range(NR)]
    assert INNER_USE <= INNER_K
    assert SELECT in ("raw", "full"), f"SELECT={SELECT!r} is not raw|full"
    # caught here rather than after the first trial has burned its whole epoch budget:
    # with FULL_EVAL off every child reports ordered_MLE_full_mm=null and nothing scores.
    assert SELECT != "full" or os.environ.get("HPO_FULL_EVAL") == "1", \
        "SELECT=full needs HPO_FULL_EVAL=1, else every trial reports ordered_MLE_full_mm=null"
    os.makedirs(WORK, exist_ok=True)

    NE = int(np.load(DATA, allow_pickle=True)["clouds"].shape[0])
    subj, parts = TF.frozen_folds(NE)
    fold_note = TF.verify_folds(subj, parts)
    outer_val_subj = parts[FOLD]
    outer_val_ears = set(np.where(np.isin(subj, outer_val_subj))[0].tolist())
    pool, splits = inner_splits(subj, outer_val_subj, INNER_K, SS + 991 * FOLD)
    assert_nested(subj, outer_val_subj, outer_val_ears, pool, splits)
    used = splits[:INNER_USE]

    cls = TF.resolve_family(FAMILY)
    space = json.loads(os.environ["SPACE_JSON"]) if os.environ.get("SPACE_JSON") \
        else getattr(cls, "SEARCH_SPACE", None)
    assert space, (f"family {FAMILY} exposes no SEARCH_SPACE and SPACE_JSON is unset -- "
                   f"there is nothing to search")
    grid = int(np.prod([len(v) for v in space.values()]))
    rng = np.random.RandomState(SS * 7 + 101)
    configs = sample_configs(space, K, rng)

    print(f"NESTED GROUPED CV -- hyperparameter search INSIDE outer fold {FOLD}")
    print(f"  data {DATA}  [{fold_note}]")
    print(f"  outer fold {FOLD} validation: {len(outer_val_subj)} SUBJECTS / "
          f"{len(outer_val_ears)} ears -- NEVER touched by this driver")
    print(f"  searchable pool: {len(pool)} training SUBJECTS")
    print(f"  inner grouped folds: {INNER_K} x ~{len(pool)//INNER_K} subjects; "
          f"scoring every trial on inner fold(s) {list(range(INNER_USE))}")
    for i, s in enumerate(used):
        print(f"    inner {i}: {len(s['train_subj'])} train / {len(s['val_subj'])} val "
              f"SUBJECTS ({len(s['train'])} / {len(s['val'])} ears)")
    print("  leakage assertions PASS: no outer-fold val subject or ear in any inner split")
    print(f"\nspace ({grid} grid points): "
          + " | ".join(f"{k} {space[k]}" for k in sorted(space)))
    print(f"sampled {len(configs)} unique config(s) by random search (seed {SS})")
    print(f"ladder: {LADDER} epochs, eta={ETA}, select on {SELECT}, runner {RUNNER}\n")

    log = {"family": FAMILY, "outer_fold": FOLD, "search_seed": SS, "eta": ETA,
           "ladder": LADDER, "k_sampled": len(configs), "grid_points": grid,
           "select": SELECT, "trial_seed": int(os.environ.get("TRIAL_SEED", "0")),
           "runner": RUNNER, "data": DATA, "fold_check": fold_note, "space": space,
           "inner": {"k": INNER_K, "used": INNER_USE, "seed": SS + 991 * FOLD,
                     "pool_subjects": [int(x) for x in pool],
                     "outer_val_subjects": [int(x) for x in outer_val_subj],
                     "folds": [{"i": i, "val_subjects": [int(x) for x in s["val_subj"]],
                                "n_train_ears": len(s["train"]), "n_val_ears": len(s["val"])}
                               for i, s in enumerate(used)]},
           "configs": [{"id": i, "cfg": c} for i, c in enumerate(configs)],
           "rungs": [], "dry_run": DRY,
           "boundary": ("SELECTION ONLY. Every trial ran in explicit-split mode on inner "
                        "folds of outer fold %d's TRAINING subjects. No outer-fold "
                        "validation ear was predicted and no screen_*/fam_* report was "
                        "written. The outer-fold evaluation is a separate run." % FOLD),
           "inner_ssm_note": ("dense-SSM blend disabled in every trial: the per-fold SSM "
                              "is built from the outer fold's training ears, which include "
                              "the inner-loop validation ears")}

    if DRY:
        n, plan = len(configs), []
        for r, budget in enumerate(LADDER):
            keep = max(1, math.ceil(n / ETA)) if r + 1 < len(LADDER) else n
            plan.append({"rung": r, "budget_epochs": budget, "n_configs": n,
                         "n_trials": n * INNER_USE, "n_keep": keep,
                         "n_dropped": n - keep, "epoch_trainings": n * INNER_USE * budget})
            print(f"RUNG {r}: budget {budget:5d} ep | {n:3d} config(s) x {INNER_USE} inner "
                  f"fold(s) = {n * INNER_USE:3d} trial(s) -> keep {keep}, DROP {n - keep}")
            n = keep
        print(f"PLAN TOTAL: {sum(p['n_trials'] for p in plan)} trials, "
              f"{sum(p['epoch_trainings'] for p in plan)} epoch-trainings. "
              f"DRY=1, nothing launched.")
        log["plan"], log["best"] = plan, None
        json.dump(log, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
        return log

    alive = list(range(len(configs)))
    best = None
    for r, budget in enumerate(LADDER):
        rung = {"rung": r, "budget_epochs": budget, "n_in": len(alive), "trials": []}
        scored = []
        for cid in alive:
            per, infos = [], []
            for i, s in enumerate(used):
                tag = f"hpo_{FAMILY}_o{FOLD}_i{i}_c{cid:03d}_r{r}"
                sc, info = run_trial(FAMILY, configs[cid], budget, FOLD, i, s, tag,
                                     WORK, SELECT, RUNNER)
                per.append(sc); infos.append(info)
            ok = [x for x in per if x is not None]
            score = float(np.mean(ok)) if len(ok) == len(per) and ok else None
            rung["trials"].append({"config_id": cid, "cfg": configs[cid],
                                   "score_mm": None if score is None else round(score, 4),
                                   "per_inner_mm": [None if x is None else round(x, 4)
                                                    for x in per],
                                   "runs": infos})
            scored.append((cid, score))
            tail = "FAILED" if score is None else f"{score:.4f}"
            print(f"  rung{r} c{cid:03d}  {tail:>8s}  "
                  f"{json.dumps(configs[cid], sort_keys=True)}"
                  + ("" if score is not None else f"  see {infos[-1].get('log')}"), flush=True)

        good = sorted([x for x in scored if x[1] is not None], key=lambda x: x[1])
        dead = [c for c, s in scored if s is None]
        n_keep = max(1, math.ceil(len(alive) / ETA)) if r + 1 < len(LADDER) else len(good)
        keep = [c for c, _ in good[:n_keep]]
        drop = [c for c, _ in good[n_keep:]] + dead
        rung.update(n_keep=len(keep), n_dropped=len(drop), kept=keep, dropped=drop,
                    failed=dead,
                    ranking=[{"config_id": c, "score_mm": None if s is None else round(s, 4)}
                             for c, s in scored])
        log["rungs"].append(rung)
        print(f"RUNG {r}: budget {budget:5d} ep | {len(alive):3d} config(s) -> keep "
              f"{len(keep)}, DROPPED {len(drop)}"
              + (f" (failed: {dead})" if dead else "")
              + f"  ids kept {keep}\n", flush=True)
        if good:
            best = {"config_id": good[0][0], "cfg": configs[good[0][0]],
                    "score_mm": round(good[0][1], 4), "budget_epochs": budget,
                    "rung": r, "select": SELECT}
        alive = keep
        if not alive:
            break

    log["best"] = best
    json.dump(log, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  (per-trial stdout in {WORK}/hpo_{FAMILY}_o{FOLD}_i*_c*_r*.log)")
    if best is None:
        print("NO config produced a score. Nothing selected.")
        return log
    cj = json.dumps(best["cfg"], sort_keys=True)
    print(f"\nBEST c{best['config_id']:03d}  {best['score_mm']:.4f} mm  at "
          f"{best['budget_epochs']} epochs (inner CV, {SELECT})  {cj}")
    print(f"\nBOUNDARY: this driver did NOT evaluate outer fold {FOLD}. Selection is done.")
    print("Report it as a separate, deliberate run through the frozen outer fold:")
    print(f"  FAMILY={FAMILY} FOLD={FOLD} SEED=0 EPOCHS={best['budget_epochs']} \\\n"
          f"    CFG_JSON='{cj}' python3 train_family.py")
    print("Then repeat for FOLD=0..4 and adjudicate with cv_verdict.py. The inner-CV "
          "score above is NOT an estimate of outer-fold performance.")
    return log


# ------------------------------------------------------------------ smoke test
def smoke():
    print("=" * 78)
    print("SMOKE -- nested-CV successive halving end to end on synthetic data")
    tmp = os.environ.get("SMOKE_DIR",
                         os.path.join(TF.tempfile.gettempdir(), "search_driver_smoke"))
    dp, tp = TF.fake_bundle(tmp)[:2]
    env = dict(FAMILY="fake", FOLD="0", SEARCH_SEED="0", K="4", ETA="3", RUNGS="1,2",
               INNER_K="2", INNER_USE="1", WORK=tmp, DATA=dp, TRIS=tp, FULL_EVAL="0",
               HPO_FULL_EVAL="0", EVAL_EVERY="1", HPO_TTA="1", TRIAL_SEED="0",
               RUNNER="inproc",       # the ladder itself: 6 trials, no interpreter starts
               CFG_LR="9.9",          # a stray CFG_* that trial_env MUST drop
               SPACE_JSON=json.dumps({"width": [8, 16, 24], "bs": [4, 8]}))
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    log = main()

    # the default runner is a subprocess; prove that path too, once
    subj, parts = TF.frozen_folds(20)
    pool, splits = inner_splits(subj, parts[0], 2, 0)
    sc, info = run_trial("fake", {"width": 8, "bs": 4}, 1, 0, 0, splits[0],
                         "hpo_subproc_check", tmp, "raw", "subprocess")
    for k, v in keep.items():
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert sc is not None, f"subprocess runner failed: {info.get('error')}"
    print(f"  subprocess runner: {info['tag']} rc={info['returncode']} "
          f"score {sc:.4f} wall {info['wall_s']}s")
    sub = json.load(open(f"{tmp}/hpo_subproc_check.json"))
    assert sub["config"]["lr"] == TF.TRAIN_DEFAULTS["lr"], \
        f"stray CFG_LR leaked into a trial: lr={sub['config']['lr']}"
    print(f"  stray CFG_LR=9.9 in the parent env was dropped (trial lr "
          f"{sub['config']['lr']})")

    ov = set(log["inner"]["outer_val_subjects"])
    for f in log["inner"]["folds"]:
        assert not (set(f["val_subjects"]) & ov)
    assert len(log["rungs"]) == 2
    assert log["rungs"][0]["n_in"] == 4 and log["rungs"][0]["n_dropped"] == 2
    assert log["rungs"][1]["n_in"] == 2
    assert log["best"] is not None and log["best"]["score_mm"] > 0
    for r in log["rungs"]:
        for t in r["trials"]:
            for run in t["runs"]:
                j = json.load(open(run["report"]))
                assert j["split_mode"] == "explicit_inner"
                assert not (set(j["val_ear_index"]) &
                            set(np.where(np.isin(TF.frozen_folds(20)[0],
                                                 log["inner"]["outer_val_subjects"]))[0]
                                .tolist())), "a trial saw an outer-fold val ear"
    assert not [p for p in os.listdir(tmp) if p.startswith(("screen_", "fam_"))], \
        "the driver wrote a report file that screen_compare/cv_verdict would pick up"
    print("  every trial report is explicit_inner; no outer-fold val ear was predicted;")
    print("  no screen_*/fam_* file written by the driver")
    print("SMOKE PASS")
    print("=" * 78)


if __name__ == "__main__":
    main() if os.environ.get("FAMILY") else smoke()
