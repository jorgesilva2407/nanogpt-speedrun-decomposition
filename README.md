# flagged — records #11–#18 of modded-nanogpt as one flagged trainer

`train_flagged.py` reproduces record #11 with every flag off (`configs/r11.yaml`) and record #18 with every flag on
(`configs/r18.yaml`); `configs/r12.yaml` … `r17.yaml` are the intermediate records. The git history replays the records
one commit per record (tags `rec-11` … `rec-18`), so the history is the audit trail.

Nothing in this repository has been executed with torch or on a GPU. What was verified statically, and how, is in
`gates/rNN.txt` and `CHANGES/rNN.md`; everything else is listed in `DEFERRED_RUNTIME_CHECKS.md`.

## Inspecting what each record changed

| question | command |
|---|---|
| what did record N add to the trainer? | `git diff rec-12 rec-13 -- train_flagged.py` (any consecutive pair of tags) |
| the whole replay at once | `git log --oneline --decorate rec-11..rec-18` |
| every upstream hunk of record N, classified | `ledger/rNN.yaml` (with `diffs/rNN-vs-r(N-1).diff`) |
| every hunk that maps to one switch, across records | `python -I tools/show_flag.py value_embeds` (or `config:logit_softcap`) |
| flag × record matrix, coupled hyperparameters, unbundled bundles | `reports/changes.md` |
| every hunk side by side, labelled by ledger id and category | open `reports/changes.html` (self-contained, filterable) |
| arm → flag → dependency table | `python -I tools/arm_table.py --markdown` |
| static gates for record N | `python -I tools/gate.py NN` → `gates/rNN.txt` |

Regenerate the reports with `python -I tools/changes_report.py`. Tools are pure Python (PyYAML for the YAML files);
they never import torch.

## How to run the study (documented, not executed)

**0. Environment.** Build the pinned image (`launch/Dockerfile` = upstream #18's; pins unchanged) and download data:
```bash
docker build -f launch/Dockerfile -t modded-nanogpt-flagged:$(git rev-parse --short HEAD) .
export DOCKER_TAG=modded-nanogpt-flagged:$(git rev-parse --short HEAD)
python data/cached_fineweb10B.py 18
```
Run the feasibility spike first (`SPIKE.md`, S1–S4); the measured σ decides the seed counts in `experiments/study.yaml`.

**1. Generate the matrix** (never hand-list configurations):
```bash
python -I experiments/matrix.py            # writes experiments/runs/<exp_id>/config.json and experiments/matrix.json
python -I experiments/matrix.py --dry-run --assume-resolved all   # preview RQ3b counts
```
Today: 25 configurations, 113 runs. Leave-one-out is withheld for A1a, A1b, A3, A4, A5 until the group decides
`OPEN_DECISIONS.md` D1. RQ3b configurations appear once `study.yaml: rq3b.resolved_arms` lists the arms resolved in RQ1.

**2. One run** (atomic; refuses a dirty tree; results/<exp_id>/seed<k>/ gets metrics.jsonl, timing.jsonl,
meta.json, DONE):
```bash
python launch/run_one.py --exp experiments/runs/loo_A7 --seed 0 --host lab-a100 --nproc 1
python launch/run_one.py --exp experiments/runs/loo_A7 --seed 0 --host lab-a100 --nproc 1 --dry-run   # print only
```
A run without `DONE` is moved aside and re-run from step 0 — there is no resume path, by design.

**3. A sweep on one host** (an arm's comparison pair and all its seeds stay on one host; the fixed points it needs are
added automatically):
```bash
launch/run_sweep.sh --host lab-a100 --nproc 1 --arm A7 --arm A3
launch/run_sweep.sh --host rented-h100 --nproc 4 --arm A6b      # overflow at arm granularity; baselines replicated
```
`experiments/hosts.json` records which host owns which arm; a second host is refused.

**4. Throughput** (lab A100 only, otherwise-idle node; prompts and records the answer; never pooled across hosts):
```bash
python launch/throughput.py --exp experiments/runs/fp_r18 --host lab-a100 --nproc 1        # 300 steps by default
```
Jobs to run: `experiments/matrix.json → throughput_jobs`. The cross-hardware check (`cross_hardware` in
`study.yaml`: fp_r18 seed 0 on lab-a100 and rented-h100) is a named experiment analysed separately.

**5. Tables and analyses** (CPU only):
```bash
python -I analysis/tables.py               # reports/table.md: one row per arm (prediction, Δ steps LOO/AOI ± 95% CI,
                                           # Δ tokens/s, Shapley wall-clock split, LOO−AOI gap as an interaction bound)
python -I analysis/mde.py                  # MDEs (edit sigma after S2)
```
`analysis/persistence.py` gives steps-to-threshold at 3.40/3.35/3.30/3.28 (RQ3a; no runs needed; grid and
interpolated crossing; loss-at-budget fallback), `analysis/decompose.py` the dTime identity with the cross term split
symmetrically. Tests: `python -m pytest tools/tests experiments/tests launch/tests analysis/tests`.

**Validation cadence** is the `measurement:` section of each config (v8: every 125 steps, every 25 in the final third,
full set). `study.yaml: validation.mode: subset20` switches to a fixed 20% subset every 5 steps in the final third —
supported, not adopted (`OPEN_DECISIONS.md` D3).

## Repository map
| path | contents |
|---|---|
| `train_flagged.py` | the trainer; `FLAGS`/`REQUIRES` at the top |
| `configs/rNN.yaml` | presets (every flag, every hyperparameter, measurement) |
| `ledger/`, `diffs/`, `CHANGES/` | per-record classification, the diffs it covers, per-record write-ups (+ `infra.md`, `arm-muon.md`) |
| `gates/` | static gate outputs; `tools/` produces them |
| `experiments/` | `arms.yaml`, `study.yaml`, `matrix.py`, generated `runs/`, `matrix.json` |
| `launch/` | `run_one.py`, `run_sweep.sh`, `plan.py`, `throughput.py`, `Dockerfile`, `env.lock` |
| `analysis/` | `load`, `mde`, `persistence`, `decompose`, `tables` (+ tests) |
| `prereg/predictions.yaml` | proposal §4, verbatim (tag `prereg-v1`) |
| `SPIKE.md`, `DEFERRED_RUNTIME_CHECKS.md`, `OPEN_DECISIONS.md` | what must be run first, what is unverified, what is undecided |
