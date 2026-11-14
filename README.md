# What do the speedrun's tricks actually buy?

**Separating sample efficiency from throughput in the NanoGPT speedrun, records #11–#18.**

Every [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) record reports one number: wall-clock time to
reach 3.28 validation loss on 8×H100. That number adds together two things with different scientific status:

* **sample efficiency**: the model needs *fewer tokens* to reach the target. This is a claim about learning, and it
  should transfer to other hardware;
* **throughput**: each token is processed *faster*. This is an optimisation for one hardware target.

This repository takes the main-track records #11–#18 (Nov 2024 – Jan 2025), puts each mechanism they introduced
behind an independent switch in **one trainer**, and builds everything needed to measure each mechanism twice:
removed from the full record #18 configuration, and added alone to the record #11 configuration. The study design is
the group's proposal (v8, kept outside this repository). References like "proposal §4" point to it.

> **Status (2026-11-14): the study is finished.** The H100 sweep (28 learning runs, 21 throughput runs, US$ 78.59) and a lab-A100 complement
> (24 learning runs, a separate dataset) have run; every number below comes from them. What was executed against the pre-registered plan,
> and every deviation: [`DEVIATIONS.md`](DEVIATIONS.md). The final table: [`reports/table.md`](reports/table.md). The data behind it:
> [`saved_results/`](saved_results). Reproduce it in one command (no GPU, no network): section **Reproducing the results**.

---

## Research questions

* **RQ1, decomposition.** For each mechanism, how much of its published gain is *fewer steps* to 3.28 and how much
  is *faster steps*?
* **RQ2, order dependence.** Does a mechanism's gain differ between leave-one-out (remove it from #18) and
  add-one-in (add it to #11)? If yes, the leaderboard's per-record, additive accounting does not hold.
* **RQ3, persistence.** For mechanisms with a real sample-efficiency effect, does the advantage last as the compute
  budget changes, or does it only show up early in training? RQ3a reads loss thresholds 3.40/3.35/3.30/3.28 from
  curves we already log. RQ3b reruns resolved arms at 0.5× and 0.75× of the step budget.

**Hypotheses** (registered in [`prereg/predictions.yaml`](prereg/predictions.yaml), git tag `prereg-v1`):

* **H1:** for at least two arms, the prediction from the mechanism's primary source fails inside this co-tuned
  stack.
* **H2:** leave-one-out and add-one-in disagree by more than seed variance for at least one arm.

## Results in one page

All numbers are in [`reports/table.md`](reports/table.md); every row can be rebuilt from [`saved_results/`](saved_results). Δ steps = steps to
validation loss 3.28 (negative = fewer steps); intervals are 95% Welch t intervals; no p-values.

* **Most of the gain is sample efficiency, not speed.** Throughput barely improves (only dropping an attention layer, A6b, +5%) and gets
  worse with long context and block windows (−11% to −15%). The mechanisms registered as throughput gains (A1a FlexAttention, A5, A5b) were not.
* **Resolved sample-efficiency effects (Δ steps, add-one-in):** A1b long context + document masking −489 ± 116, A4 U-net value embeddings −221 ± 145,
  A5 −565 ± 129, A4b −473 ± 150, A6 −219 ± 116; removing A2 costs +96 ± 41, A6 +60 ± 41, A7 +79 ± 32 (leave-one-out). A5b (record #16 MFU tweaks) is null in
  both directions (+17 ± 240, −5 ± 94). The positive control passes: AdamW needs at least 1.55× the steps of Muon.
* **H1 (the source's prediction fails for ≥ 2 arms): supported** (A3 value embeddings, A7 softcap, A1a). **H2 (leave-one-out and add-one-in
  disagree beyond seed noise for ≥ 1 arm): supported only with a caveat**: A4b and A6 disagree, but their add-one-in configurations are bundles of
  prerequisites, so the gap mixes interaction with the bundle; the clean pairs (A2, A5b, A6b, A7) show no resolved disagreement.
* **RQ3:** the effects of the value-embedding arms are flat across loss thresholds 3.40/3.35/3.30 (RQ3a, a screening instrument); at 0.5×/0.75×
  of the budget (RQ3b, add-one-in only, one seed) the advantage of A3 and A6 shrinks monotonically but stays, within single-run noise (≈ 0.006 loss).
* **Two GPU models.** The H100 sweep left gaps; the lab A100 filled them under a protocol and pass rule written before its first run
  ([`lab_a100/PROTOCOL.md`](lab_a100/PROTOCOL.md)). The transfer check passed (baseline offset +0.0002 ± 0.0093 inside ±0.010; the bridge arm gives
  −218 steps on the A100 against −219 on the H100). Every Δ uses a baseline from its own host and the two datasets are never pooled; throughput is the
  H100's only. This deviates from the single-GPU-model rule of the proposal and is a stated limitation.
* **Limits to keep in mind:** 1–3 seeds per configuration (arms with one seed have no interval), several add-one-in arms are bundles
  ([`DEVIATIONS.md`](DEVIATIONS.md) 3.1), 300-step single-run throughput, one slice of the speedrun history.

---

## How it works

### 1. Recovering what each record actually ran

The upstream git history is not a clean list of "one commit per record". For example, commit `551ec61` *adds #11's
log* but still holds #10's code, and records #15 and #17 were never committed at all. However, every committed
record log **begins with the full training script that produced it**. So each record's snapshot is that embedded
source, extracted byte-exact:

* [`records/rNN/log_source/`](records) holds the snapshot for record NN.
* `records/rNN/ANCHOR.txt` holds its SHAs and hashes.
* [`records/MANIFEST.md`](records/MANIFEST.md) records every snapshot's relation to the nearest commit and its
  confidence. All eight are high confidence.
* [`EXPLORATION.md`](EXPLORATION.md) is the full investigation: history shape, anchors, the intervening commits, and
  the findings listed further down.

The scripts behind it are in `explore_tools/`. They are pure Python and never import torch.

### 2. One trainer, one switch per mechanism

[`train_flagged.py`](train_flagged.py) (≈1,340 lines) is record #11's script with every later mechanism added
behind a switch. Presets select the records:

| preset | meaning |
|---|---|
| [`configs/r11.yaml`](configs/r11.yaml) | every flag off = record #11 |
| `configs/r12.yaml` … `r17.yaml` | the intermediate records |
| [`configs/r18.yaml`](configs/r18.yaml) | every flag on = record #18 (except `code_opts_17`, which #18 itself reverts) |

Design rules:

* **Every flag and every hyperparameter lives in the config. There are no defaults in code.** A config that omits
  a field is rejected.
* **Prerequisites are enforced.** `REQUIRES` at the top of the trainer lists them, for example `ve_unet` needs
  `value_embeds`. A combination that is missing a prerequisite stops at start-up and names the missing flag.
* **"Code optimisations" that change the maths are separate flags.** Examples: #15's hand-built attention mask and
  #15's EMA-form Muon momentum.
* **Record behaviour is never removed to simplify the code.** The 65,536-token micro-batch is asserted at start-up.

Three pieces of infrastructure sit on top of the record code. Each has its own commit and a write-up in
[`CHANGES/infra.md`](CHANGES/infra.md):

1. **Seeding.** The records seed nothing. `--seed` now seeds every randomness source enumerated in `infra.md`.
2. **A unified batch convention.** One optimizer step is always 8 micro-batches of 65,536 tokens (524,288 tokens).
   On W GPUs each rank accumulates 8/W of them. Micro-batch *v = rank·accum + a* gets exactly the data that GPU *v*
   would get in an 8-GPU run, so the token stream is the 8-GPU stream at any GPU count. The records' own loaders do
   not have this property: their 1-GPU and 8-GPU streams diverge at step 189. The data equivalence is verified by
   simulation. Gradient equivalence, including loss scaling, is argued on paper only and is **unverified**.
3. **Run logging.** Each run writes `metrics.jsonl`, `timing.jsonl` (CUDA-event step times), `meta.json` (config
   hash, host, peak memory) and, last of all, a `DONE` marker.

### 3. Replaying the history, one commit per record

The git history of this repository *is* the audit trail. It replays the records in order, one commit and one tag
per record (`rec-11` … `rec-18`). For each record:

1. Export the snapshot.
2. Diff it against the previous snapshot: [`diffs/rNN-vs-r(N-1).diff`](diffs).
3. Classify **every** hunk in the ledger [`ledger/rNN.yaml`](ledger): which flag or config field it maps to, or
   why it needs none. Between 17 and 50 hunks per record.
4. Implement the flags.
5. Pass the static gate.
6. Commit as `rec-NN: <title> [flags: …]` and write up the record in [`CHANGES/rNN.md`](CHANGES).

The **static gate** (`tools/gate.py NN` → [`gates/rNN.txt`](gates)) checks, without torch:

| check | what it verifies |
|---|---|
| `syntax` | the trainer parses |
| `constants` | every hyperparameter and every mechanism detector extracted from the record's script matches preset `rNN` |
| `ledger` | every hunk of the record's diff is classified |
| `params` | an AST interpreter builds the model on paper and counts parameters and buffers per tensor: #11 = 162,201,642, #18 = 275,742,772 |
| `loader` | the trainer's data stream equals the record's 8-GPU stream at W = 1, 2, 4, 8 (simulated on synthetic shards) |

All eight records pass. `tools/matrix_inventory.py` builds every generated experiment configuration the same way,
including flag combinations that never existed historically.

### 4. Arms and the experiment matrix

An **arm** is a group of switches measured together ([`experiments/arms.yaml`](experiments/arms.yaml)):

| arm | mechanism | record | switches |
|---|---|---|---|
| A1a | FlexAttention kernel | 12 | `flex_kernel` |
| A1b | 64K-context sliding window + document masking | 12 | `long_context`, `doc_masking` |
| A2 | attention-window warmup | 13 | `window_warmup` |
| A3 | value embeddings (VE rung 1) | 14 | `value_embeds` |
| A4 | U-net value embeddings (VE rung 2) | 15 | `ve_unet`, `ve_lambdas2` |
| A4b | #15 assorted code optimisations | 15 | `code_opts_15`, `muon_ema_momentum`, `blockmask_manual` |
| A5 | split VE + block sliding window (VE rung 3) | 16 | `ve_split`, `block_window`, `blockmask_separate` |
| A5b | #16 MFU tweaks | 16 | `code_opts_16` |
| A6 | sparsified VE + half-truncated rotary (VE rung 4) | 17 | `ve_sparsify`, `rope_half_truncate` |
| A6b | dropped attention layer | 17 | `drop_attn_layer` |
| A7 | logit softcap 30 → 15 | 18 | config `logit_softcap` |
| A8 | Muon → AdamW (positive control) | — | `optimizer` |

Each arm is compared against a fixed point:

* **Fixed points:** `fp_r11` and `fp_r18`, the two unmodified presets.
* **Leave-one-out (`loo_<arm>`):** r18 with the arm switched off. It keeps r18's hyperparameters.
* **Add-one-in (`aoi_<arm>`):** r11 with the arm switched on. It keeps r11's hyperparameters.
* **A8 control:** AdamW replaces Muon in both endpoints, plus an AdamW learning-rate sweep.

[`experiments/matrix.py`](experiments/matrix.py) **generates** the matrix from `arms.yaml` and
[`experiments/study.yaml`](experiments/study.yaml); configurations are never listed by hand. It writes one
`experiments/runs/<exp_id>/config.json` per configuration and a summary in `matrix.json`. Seed counts are config
fields: 3 by default, 5 for RQ2 arms.

The matrix has **43 configurations**: the 21 of the sweep (D2, 63 planned runs) and the 22 budget-scaled RQ3b ones (`rq3b_s0.5_*`, `rq3b_s0.75_*`; tag `a100-v1`), of which 28 + 24 runs were executed (`DEVIATIONS.md`). Leave-one-out is withheld for A1a, A1b, A3, A4 and A5,
because switches that stay on in r18 depend on them. Choosing what "remove" means for those arms is decision D1.

### 5. Running: `launch/`

* **`run_one.py`** runs one configuration with one seed.
  * Runs are atomic: there is no resume path, by design. A run without `DONE` is moved aside and restarted from
    step 0.
  * It refuses to run from a dirty git tree, and records the commit and the Docker tag.
* **`run_sweep.sh` + `plan.py`** enforce the host rules: an arm's comparison pair and all its seeds stay on one host,
  results are never pooled across hosts, and overflow to a second host happens per arm, with the baselines
  replicated there.
* **`throughput.py`** measures tokens/s, kept separate from the learning-curve runs. It refuses to start unless
  you confirm the node is otherwise idle. The study measures throughput on one host only (`study.yaml:
  throughput.host`, the rented H100; the lab A100 was used for the complement of `DEVIATIONS.md` 7).
* **Environment:** `Dockerfile` is upstream #18's (CUDA 12.6.2, Python 3.12.7) with one deviation: **stable
  `torch==2.6.0+cu126`** instead of the nightly `2.6.0.dev20241231+cu126`, which PyTorch has pruned from its index
  (decided 2026-10-18; what changes is in `CHANGES/infra.md`, "compat: torch pin"). `env.lock` separates pinned
  lines from unverified ones.

### 6. Analysis: `analysis/` (CPU only)

* **`persistence.py`** computes steps to a loss threshold in two ways: the first logged step at or below it, and a
  linear interpolation between logged points. If a curve never crosses, it falls back to the loss at a fixed budget.
  It also runs the RQ3a thresholds.
* **`decompose.py`** splits the wall-clock change into sample-efficiency and throughput shares:
  ΔTime = ΔS·T₀ + S₀·ΔT + ΔS·ΔT. The cross term is split symmetrically between the two shares, and its size is
  reported.
* **`mde.py`** computes the detectable effects. It reproduces the proposal's table exactly at σ = 0.0013: RQ1
  ±46 / ±36 steps and RQ2 ±65 / ±51 steps, for 3 and 5 seeds. Those are z intervals; t intervals are wider and also
  reported.
* **`tables.py`** builds the primary deliverable, one row per arm: registered prediction, Δ steps (LOO and AOI,
  with 95% CI), Δ tokens/s, wall-clock split, and the LOO−AOI gap as an upper bound on interaction; with `--results-a100` it
  also reads the A100 dataset (never merged with the H100 one) and adds the transfer check (`transfer.py`), the RQ3b table, the A8 row
  and the whole-run step times.
* **`load.py`** reads only completed runs and **refuses to pool runs from different hosts**.

All of it is tested on synthetic curves.

### 7. Pre-registration

[`prereg/predictions.yaml`](prereg/predictions.yaml) copies proposal §4 verbatim (10 rows, five marked
"unsupported"). It was committed on its own and tagged `prereg-v1` before any run. At commit time it was checked
mechanically against the proposal: all 10 rows match. The softcap row is copied as written; the group has not yet signed off on it
(D5).

---

## Reproducing the results

**Requirements.** Python 3.12 (the version used), no GPU, no network after the clone and `pip install`. Setup and exact commands:

```bash
git clone git@github.com:jorgesilva2407/nanogpt-speedrun-decomposition.git && cd nanogpt-speedrun-decomposition
git checkout submission                                    # the state of the code at submission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-tools.txt                      # numpy, pyyaml, pytest: pinned; no torch needed

python -I analysis/tables.py --results saved_results/h100 --results-a100 saved_results/a100 --out reports/table.md
git diff --stat reports/table.md                           # empty: the committed table is exactly this output
python -m pytest tools/tests experiments/tests launch/tests analysis/tests     # tests, incl. saved_results -> table
```

`saved_results/` holds the 73 completed runs (H100: 28 learning + 21 throughput; A100: 24 learning) in the layout `analysis/load.py` reads:
`<configuration>/seed<k>/{meta.json, metrics.jsonl, timing.jsonl, wall_seconds, DONE}`. It is a lossless trim of the full datasets (training-loss rows
are dropped; validation losses and per-step GPU times are copied unchanged; see [`saved_results/README.md`](saved_results/README.md)). The full
datasets are on Hugging Face (`speedrun-decomp/speedrun-results`, `speedrun-decomp/speedrun-results-a100`; private, access on request) and
`tools/export_saved_results.py` rebuilds `saved_results/` from them.

**Re-running the training** needs a GPU (H100 SXM 80GB for the main sweep; any 80 GB card for the A100 protocol), Docker and the data:

```bash
docker pull ghcr.io/jorgesilva2407/modded-nanogpt-flagged:9ec04dbd69e72372fba7fbd797030e46363b8662   # the image behind every result
# or build it: docker build -f launch/Dockerfile -t modded-nanogpt-flagged:$(git rev-parse --short HEAD) .   (CI: .github/workflows/image.yml)
python data/cached_fineweb10B.py 18                         # FineWeb 10B tokens, 18 shards (~4 GB), into data/fineweb10B/
python launch/run_one.py --exp experiments/runs/fp_r11 --seed 0 --host my-gpu --nproc 1     # one run, one seed (~1 h on an H100)
```

The sweeps themselves are the two drivers described in **How the study was run** below. Credentials are never stored in the repository: they are
read from the environment (`GH_TOKEN`, `HF_TOKEN`, `RESULTS_REPO`; see the header of `launch/cloud_worker.sh`).

## Static checks (no GPU)

```bash
sh explore_tools/fetch_sources.sh      # upstream clone, exported record trees, gists, torch sources (idempotent; ~0.5 GB, not committed)
python -I tools/gate.py 11             # static gate of record 11 ... 18 -> gates/rNN.txt
python -I experiments/matrix.py        # regenerate the experiment matrix
python -I tools/changes_report.py      # regenerate reports/changes.*
```

## Inspecting what each record changed

| question | command |
|---|---|
| what did record N add to the trainer? | `git diff rec-12 rec-13 -- train_flagged.py` (any consecutive pair of tags) |
| the whole replay at once | `git log --oneline --decorate rec-11..rec-18` |
| every upstream hunk of record N, classified | `ledger/rNN.yaml` (with `diffs/rNN-vs-r(N-1).diff`) |
| every hunk that maps to one switch, across records | `python -I tools/show_flag.py value_embeds` (or `config:logit_softcap`) |
| flag × record matrix, coupled hyperparameters, unbundled bundles | [`reports/changes.md`](reports/changes.md) |
| every hunk side by side, labelled by ledger id and category | open `reports/changes.html` (self-contained, filterable) |
| arm → flag → dependency table | `python -I tools/arm_table.py --markdown` |
| static gates for record N | `python -I tools/gate.py NN` → `gates/rNN.txt` |
| does every torch API the trainer calls exist in the pinned torch? | `python -I tools/torch_api_check.py ext/torch_src/v2.6.0` → `gates/torch_api.txt` |

## How the study was run

Everything below was executed; the order, the host rules and every deviation from the plan are in [`DEVIATIONS.md`](DEVIATIONS.md).

**Image and data.** Image `ghcr.io/jorgesilva2407/modded-nanogpt-flagged:9ec04dbd69e72372fba7fbd797030e46363b8662` (pinned dependencies: `requirements.txt`,
full freeze `launch/requirements-frozen.txt`); data `python data/cached_fineweb10B.py 18`. Validation: every 125 steps and every 25 in the final third, on
the full validation set (`measurement:` in each config; D3).

**1. The experiment matrix** (generated, never written by hand):
```bash
python -I experiments/matrix.py            # experiments/runs/<exp_id>/config.json and experiments/matrix.json
```

**2. One run** writes `results/<exp_id>/seed<k>/{metrics.jsonl, timing.jsonl, meta.json, DONE}` (atomic: no resume; a run without `DONE` restarts at step 0):
```bash
python launch/run_one.py --exp experiments/runs/loo_A7 --seed 0 --host vast-h100 --nproc 1
python launch/run_one.py --exp experiments/runs/loo_A7 --seed 0 --host vast-h100 --nproc 1 --dry-run   # print only
```

**3. The H100 sweep** (tag `study-freeze`): one rented on-demand H100 SXM 80GB (Vast.ai) running `launch/cloud_worker.sh` per tier
(`fixed_points`, then `p1`, then `throughput`), with the environment variables listed in its header, inside the image above; every finished run is
uploaded to the results dataset. The job lists and the tier script actually used are in `saved_results/h100/_logs/`.
```bash
TIER=fixed_points bash launch/cloud_worker.sh     # needs GH_TOKEN HF_TOKEN RESULTS_REPO STUDY_TAG DOCKER_TAG HOST_LABEL; DRY_RUN=1 to check first
python -I launch/make_jobs.py --tier p1 --status --repo <hf-user>/speedrun-results      # which jobs are finished
```

**4. Throughput** (300 steps per configuration, otherwise-idle node, one run each, seed 0): tier `throughput` of the same worker
(`launch/throughput.py`), host `vast-h100`. The jobs are `experiments/matrix.json → throughput_jobs`.

**5. The lab-A100 complement** (tag `a100-v1`): one `docker run` that executes all five phases of [`lab_a100/PROTOCOL.md`](lab_a100/PROTOCOL.md) with
`lab_a100/run.sh all` (anchor, gaps, rq2, seeds2, rq3b), into the separate dataset `speedrun-results-a100`; the exact command is in the protocol, section 5.

**6. Tables:** the commands of **Reproducing the results**. `python -I analysis/mde.py` prints the detectable effects.

## Findings so far (static; details in `EXPLORATION.md` and `CHANGES/`)

* **The commit that adds #11's log is not #11's code.** `551ec61` still holds #10's script. The maintainer's README
  at #12 says to use the code embedded in the log.
* **#15 and #17 exist only in logs and gists.** 279 script versions were checked, including all PR heads.
  YouJiacheng's gists corroborate both.
* **#18's change is not "softcap only" at snapshot level.** #18 was built on a rewrite that also changes the
  precision of small parameters, the LR cooldown, the data order (1,201 of 1,390 steps get different batches) and
  the embedding row count. Each of these is its own switch here, so arm A7 really is just softcap 30 → 15.
* **#15's hand-built block mask drops attention pairs that the window allows:** 24.6% at window 64, 49.6% at 128,
  3.5% at 1792. #16 re-adds them.
* **#13 retuned hyperparameters that its README does not mention.** They are captured in config and marked as
  coupled.
* **The records are not GPU-count invariant.** The 1-GPU and 8-GPU streams diverge at step 189, despite the README
  claim, which #16 itself softens. Section 2 above describes the fix.

## Decisions ([`OPEN_DECISIONS.md`](OPEN_DECISIONS.md))

All taken on 2026-11-04 except D8 (2026-10-18). Reasoning in `OPEN_DECISIONS.md`.

| | decision |
|---|---|
| D1 | O4: no leave-one-out for A1a, A1b, A3, A4, A5 (add-one-in only); RQ2 covers six arms. |
| D2 | The generated matrix: 21 configurations (2 fp, 11 aoi, 6 loo, 2 ctrl), 63 runs. Supersedes the proposal's 14. |
| D3 | v8 validation cadence (every 125, every 25 in the final third, full set). |
| D4 | Window warmup kept as is; A2's add-one-in is a truncated mechanism and its LOO/AOI gap is reported as structurally inflated. |
| D5 | Softcap prediction signed off as registered. |
| D6 | All-cloud schedule: freeze 2026-11-04 evening, sweep 2026-11-05 to 2026-11-07, results freeze 2026-11-08. |
| D7 | 3 seeds everywhere; no AdamW LR sweep. |
| D8 | Stable `torch==2.6.0+cu126` replaces the pruned nightly (2026-10-18). |
| D9 | One GPU model for every run: H100 SXM 80GB. |
| D10 | Many rented machines of that model with the pinned image are one host class (`vast-h100`). |
| D11 | Throughput: all 21 jobs back to back on one rented instance; only deltas are reported. |
| D12 | Fixed-point pass (revised 2026-11-05): fp_r18 vs the 80-run mean 3.2791 at σ = max(σ̂, 0.0019); fp_r11 vs 3.2753 within max(4.97σ̂, 0.0044), weakly verified; seeding and curve checks trigger investigation. |
| D13 | Cut line fixed in advance from r = measured / estimated fixed-point wall time. |

## Repository map

| path | contents |
|---|---|
| `train_flagged.py` | the trainer; `FLAGS`, `FLAG_CHOICES` and `REQUIRES` at the top |
| `configs/rNN.yaml` | record presets: every flag, every hyperparameter, the measurement section |
| `ledger/`, `diffs/`, `CHANGES/` | per-record hunk classification, the diffs it covers, per-record write-ups (+ `infra.md`, `arm-muon.md`) |
| `gates/` | static gate outputs (`rNN.txt`, `matrix_inventory.txt`) |
| `tools/` | gate, ledger, diff, AST parameter inventory, loader simulation, reports (+ tests) |
| `experiments/` | `arms.yaml`, `study.yaml`, `matrix.py`, generated `runs/` and `matrix.json` (+ tests) |
| `launch/` | `run_one.py`, `run_sweep.sh`, `plan.py`, `throughput.py`, `Dockerfile`, `env.lock` (+ tests) |
| `analysis/` | `load`, `persistence`, `decompose`, `mde`, `tables` (+ tests) |
| `prereg/predictions.yaml` | proposal §4, verbatim (tag `prereg-v1`) |
| `reports/` | `changes.md`, `changes.html` (generated) |
| `data/cached_fineweb10B.py` | upstream's data download script, unchanged |
| `EXPLORATION.md`, `explore_tools/` | the history investigation and its scripts and outputs |
| `records/` | `MANIFEST.md` and per-record snapshots (`log_source/`, `ANCHOR.txt`) |
| `SPIKE.md`, `DEFERRED_RUNTIME_CHECKS.md`, `OPEN_DECISIONS.md` | what must run first, what is unverified, what is undecided |
| `requirements.txt` / `requirements-tools.txt` | GPU image dependencies / CPU-only tooling (pinned) |

These paths are **not versioned** and are re-created by `explore_tools/fetch_sources.sh` from pinned SHAs:

* `upstream/`: the upstream clone, with full history and PR heads;
* `records/rNN/tree/`: the exported upstream trees, 442 MB;
* `ext/`: the gists and the pinned torch sources;
* `.venv/`.

`results/` and `experiments/hosts.json` are run outputs and are also ignored.

## Provenance: what was reused and what was developed

**Reused (upstream, [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), MIT License, © 2024 Keller Jordan and contributors):**
the eight record scripts (`records/rNN/log_source/`, byte-exact from the record logs), `data/cached_fineweb10B.py` (unchanged), the CUDA/Python base of
`launch/Dockerfile`, and the training recipe itself, which `train_flagged.py` starts from (record #11). PyTorch 2.6.0 is used with two upstream
backports (`launch/patch_torch.py`: FlexAttention `d2ec7f0756`, cuDNN SDPA `ec93aa7f84`). Data: FineWeb 10B tokens as tokenised by upstream.

**Developed here:** the single flagged trainer (one switch per mechanism, prerequisites enforced, seeding and a world-size-independent batch convention),
the ledger that classifies every hunk of the eight records, the static gates, the experiment matrix generator and its 43 configurations, the launch and
run-logging tooling, the lab-A100 protocol and driver, the analysis (steps-to-threshold with a persistence fallback, the additive wall-clock split, the
transfer check, the tables), the tests, and the pre-registered predictions (`prereg/predictions.yaml`, tag `prereg-v1`). All experiments, results and
conclusions are the group's.

## AI assistance

> Claude Code (Anthropic) was used as an assistant during the project: to help write and
> revise parts of the code in this repository (including pieces of the trainer, `tools/`, `experiments/`, `launch/`, `lab_a100/`, `analysis/` and the
> tests), and to help draft the documentation (README, `DEVIATIONS.md`, `CHANGES/`), which the group reviewed and edited. The
> study design (research questions, arms, pre-registered predictions, the decisions D1–D13, the budget and spending choices), the running of the
> experiments and the interpretation of the results are the group's own.
