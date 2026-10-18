# EXPLORATION — upstream history for records #11–#18

Repository: `github.com/KellerJordan/modded-nanogpt`, cloned with full history into `upstream/`
(not shallow; 2,269 commits on `master`, HEAD `4ea6b93` 2026-09-28). PR heads were fetched read-only
(`refs/pull/*/head`, 288 refs). Two public gists linked from the README were cloned (read-only, never
executed) into `ext/`. Ancestor nanoGPT was not needed for any finding below.

Everything in this file was established by reading git objects and by pure-Python text tooling
(`explore_tools/*.py`, run with `python -I`; no torch import). Nothing was executed on a GPU.
Raw outputs: `explore_tools/match_logs*.json`, `readme_rows.json`, `logmeta.json`, `intervening_table.md`.

**Verified** = read directly from git objects / log files. **Inferred** = reasoning over verified facts,
labelled where used.

---

## 0. Headline findings (all verified unless marked)

1. **Every committed log for #11–#18 begins with the full training-script source.** All logs inside each
   record directory embed the *same* source (1 distinct source per record). The embedded source is therefore
   the primary anchor for every snapshot, and it is unambiguous.
2. **The brief's #11 commit `551ec61` is the log-adding commit, not the code.** Its `train_gpt2.py` is still
   record #10's script (no U-net, single LR, 3242 iterations). The #11 code reaches master's first-parent line
   only on 2024-11-20 (`607f2a3`), in a refactored form that differs substantively from the log. The only
   commit whose script matches the #11 log is `ff93d20b` (bluecoconut, PR #25, side branch merged via
   `2e7b3df7`), differing by one cosmetic docstring prefix (`r"""` vs `"""`); the very next commit on that PR
   reverted it.
3. **Two records' code exists in no git commit at all: #15 and #17.** Their code is known only from the
   embedded log source, corroborated by YouJiacheng's public gists (linked from the README before the logs
   were committed). #15: gist `train_gpt2.py` is whitespace-identical to the log source. #17: the gist log
   (run 2024-12-23) and the committed log (run 2024-12-25) embed the same model code, differing only by a
   3-line diagnostic print.
4. **#18 commit `45544f5` changes only the softcap (30→15) and `num_iterations` (1490→1390)** — the brief's
   "sole modelling change" is true *at the commit level*. **But the #17→#18 snapshot diff is not
   softcap-only**: #18 was built on Keller's 2024-12-13…2025-01-01 re-implementation of #17 (`32fce07`), not
   on the code that produced the #17 log. That rewrite changes numerics and semantics relative to snapshot
   #17 (details in §5 and `ledger/r18.yaml` once built): non-embedding scalar parameters and rotary cos/sin
   tables go bf16→fp32 (`model.bfloat16()` replaced by `bf16_embeds`), LR cooldown becomes a fraction
   (600 iters → `cooldown_frac=0.4`), embedding/value-embedding tables grow 50257→50304 rows, the data loader
   is rewritten (different shard-advance rule), gradient accumulation stops dividing by the number of
   micro-steps, `coordinate_descent_tuning` and `set_stance` are dropped.
5. **Master's history was rewritten after 2024-12-09.** Every commit after 2024-06-12 exists twice: once in
   the current master graph and once (same author/date/message, different hash, identical trees where
   checked) in `refs/pull/*/head` and `origin/soap2`. All SHAs in this study are current-master SHAs;
   pre-rewrite twins are noted where they were the only place a match was found (`917332c` ≡ `ff93d20b`).
6. **Record data loaders are not world-size invariant at shard boundaries for #11–#17** (verified by reading
   the loader code; consequence inferred by arithmetic): each rank decides independently when to advance to
   the next 100M-token shard, using a rank-dependent offset, and the advance threshold depends on
   `world_size`. #18's rewritten loader decides globally and is world-size invariant. This matters for the
   unified batch convention (Step 1) and for spike item 3.
7. Six different torch builds produced the eight logs (2.5.1+cu124 → 2.6.0.dev20241231+cu126); all logs ran
   on 8×H100 80GB HBM3.

---

## 1. History shape

Window 2024-11-10 … 2025-01-05 (record #11 through #18's log commit):

| quantity | value |
|---|---|
| first-parent commits on `master` | 209 |
| all commits in `master` ancestry | 305 |
| first-parent merges | 16 (13 `Merge branch 'master' of …` — i.e. local commits merged with web-UI commits; 2 with subject `.`; 1 PR merge #31) |
| PR merges anywhere in ancestry | 9: #25, #27, #31, #42, #43, #44, #51, #56, #58 |
| first-parent authors | `KellerJordan` (CLI, 106) and `Keller Jordan` (GitHub web UI, 103) — same person |
| first-parent commits with subject exactly `.` | 156 of 209 |
| squash vs merge | No squash merges. PRs are merged with merge commits (`Merge pull request #N`), usually *into the web-UI line*, which Keller's local line then merges back (`Merge branch 'master'`). |
| linearity | Non-linear: two interleaved lines (local CLI commits vs web-UI commits) joined by back-merges; record code is frequently integrated by hand rather than by merging the contributor's branch. |
| rewritten history | yes — see finding 5; `git merge-base origin/pr/43 master` = `b9d141e` (2024-06-12) although PR #43 *was* merged (`d395a2c0`, twin `8e5bbd60` of PR head `11aba6c1`). |

**Rename.** `train_gpt2.py` → `train_gpt.py` at `5db4da3` (2024-12-31, "rename from gpt2 to gpt", R100,
content-identical). `git log --follow train_gpt.py` crosses it; every probe in this file searched both
filenames at every commit.

Implication (inferred): commit messages carry almost no information; record membership must be established
from content (embedded log source, README rows), never from messages or PR boundaries.

---

## 2. Leaderboard provenance (README.md, `git log -p` over master ancestry)

Tool: `explore_tools/readme_rows.py` tracks each row (keyed by record time) through every commit touching
`README.md`. Before 2024-11-24 the README used a prose list; the table rows for #11–#13 were first written
at `62755357` (2024-11-24, web UI) and numbered at `2d091d8c` minutes later.

| # | first row commit (date) | time / date in row | description history | log link history | other links |
|---|---|---|---|---|---|
| 11 | `62755357` (2024-11-24) | 7.2 min, 11/10/24 | "U-net & 2x lr" → "U-net skip connections & 2x lr" (`0280f14a`, 2025-01-17) → "…& double lr" (`1467a120`) → "U-net pattern skip connections & double lr" (`3398582c`) | `records/111024_UNetDoubleLr/c87bb826-….txt` throughout | tweet `x.com/kellerjordan0/status/1856053121103093922`; record README: `x.com/brendanh0gan/status/1855273758681866352`, fork `github.com/brendanhogan/modded-nanogpt`; contributor @brendanh0gan |
| 12 | `62755357` (2024-11-24) | 5.03 min, 11/19/24 | "FlexAttention" → "Switch from 1024-ctx causal attention to 64K-ctx FlexAttention" (`1467a120`, 2025-01-17) → … → "1024-ctx dense causal attention → 64K-ctx FlexAttention" (`8482b0eb`) | `records/111924_FlexAttention/8384493d-….txt` | tweet `x.com/kellerjordan0/status/1859331370268623321`; record README (`a6ecbb84`, 2024-11-21) cites `x.com/YouJiacheng/status/1859876224639828068` (mean ≈3.279, sd ≈0.005); @KoszarskyB |
| 13 | `62755357` (2024-11-24) | 4.66 min, 11/24/24 | "Attention window warmup" (unchanged); tweet link added `d1b62120` | `records/112424_WindowWarmup/cf9e4571-….txt` | tweet `x.com/hi_tysam/status/1860851011797053450`; @fernbear.bsky.social; no record README |
| 14 | `45a8dc24` (2024-12-05) | 4.41 min, 12/04/24 | "Value Embeddings" (link empty until `4442c394`, 2024-12-09) | `./records/120424_ValueEmbed` (directory) | tweet `x.com/KoszarskyB/status/1864746625572257852`; record README with 40 val losses; @KoszarskyB |
| 15 | `4442c394` (2024-12-09) | 3.95 min, 12/08/24 | "U-net structure for value embeddings, assorted code improvements" → "U-net pattern for value embeds, …" (`92b60a20`) → "…value embeddings, assorted code optimizations" (`0280f14a`) → "U-net pattern value embeddings, assorted code optimizations" (`3398582c`) | **gist** `gist.github.com/YouJiacheng/1fed95270177aa04f389b46e8c7d7733` → `records/120824_UNetValueEmbedsTweaks` at `2035d87c` (2024-12-11) | tweet `x.com/YouJiacheng/status/1865761473886347747`; @leloykun, @YouJiacheng; PR #43 (YouJiacheng, merged 2024-12-09); PR #45 (leloykun "add 12/7/24 record", **not merged**) |
| 16 | `b6e0ae8e` (2024-12-12) | 3.80 min, 12/10/24 | "MFU tweaks" → "Code optimizations" / "Various code optimizations" (2025-01-06) → "Split value embeddings, block sliding window, separate block mask" (`0280f14a`, 2025-01-17) | `records/121024_MFUTweaks` (directory) | tweet `x.com/YouJiacheng/status/1866734331559071981`; @YouJiacheng; no record README |
| 17 | `a146238e` (2024-12-31) | 3.57 min, 12/17/24 | "Sparsify value embeds & improve rotary & drop attn layer" → "Sparsify value embeddings, improve rotary, drop an attn layer" (`96f1d856`) → "…improve rotary embeddings, drop an attn layer" (`0280f14a`) | **"tbd"** → **gist** `gist.github.com/YouJiacheng/dff723d9a362303f4dca6d71a6469555` (`a61236fa`, 2024-12-31) → `records/121724_SparsifyEmbeds` (`e7e00c24`, 2025-01-05) | tweet `x.com/YouJiacheng/status/1868938024731787640`; record README (1261 runs, mean 3.2794, sd 0.0018); @YouJiacheng |
| 18 | `45544f53` (2025-01-04) | 3.4 min, 01/04/25 | "Softer softcapping" → "Increased"/"Intensified softcapping" → "Softcapping 30 -> 15" → "Lower logit softcapping from 30 to 15" → "Lower logit softcap from 30 to 15" (`fac09a46`, 2025-01-05) | `records/010425_SofterSoftCap/31d6c427-….txt` → dir renamed `010425_SoftCap` at `ccc46863` | tweet `x.com/kellerjordan0/status/1876048851158880624` (added `66822eba`); record README (80 runs, mean 3.2791, sd 0.0019; reduces steps 1490→1390); @KoszarskyB |

Observation (verified): the descriptions the proposal uses for #15–#17 ("U-net pattern value embeddings,
assorted code optimizations", "Split value embeddings, block sliding window, separate block mask", "Sparsify
value embeddings, improve rotary embeddings, drop an attn layer") were written on **2025-01-17**, one month
after the records; #16 was originally labelled "MFU tweaks".

---

## 3. Committed logs

| # | log path (at `5deb47d`, pre-2025-07-27 folder rename) | logs in dir | first commit adding a log | embeds source? | run timestamp (nvidia-smi) | torch | steps / final val / train_time |
|---|---|---|---|---|---|---|---|
| 11 | `records/111024_UNetDoubleLr/c87bb826-….txt` | 1 | `551ec61` 2024-11-11 | yes (`'='*100` + code + `'='*100`) | Nov 10 19:09 2024 | 2.5.1+cu124 | 3000 / 3.2753 / 433.6 s |
| 12 | `records/111924_FlexAttention/8384493d-….txt` | 1 | `ebce774` 2024-11-20 (same commit as code) | yes | Nov 20 01:46 | 2.6.0.dev20241119+cu124 | 1875 / 3.2783 / 301.8 s |
| 13 | `records/112424_WindowWarmup/cf9e4571-….txt` (+6) | 7 | `759b8fc` 2024-11-25 (same as code); more at `f76ef0b`, `9fed095` | yes | Nov 24 23:36 – Nov 25 00:24 | 2.6.0.dev20241124+cu124 | 1750 / mean 3.2770 / 279.4 s |
| 14 | `records/120424_ValueEmbed/*` | 38 at `5deb47d` (40 originally; 1 removed by `5fb62cc` "rm a log to save space test", 2025-01-04) | `2305e1e` 2024-12-04 (before the code commit) | yes | Dec 5 00:51 – 04:56 | 2.6.0.dev20241203+cu124 | 1530 / mean 3.2776 |
| 15 | `records/120824_UNetValueEmbedsTweaks/*` | 75 | `2035d87` 2024-12-11 | yes | Dec 8 07:23 – 14:01 | 2.6.0.dev20241203+cu124 | 1480 / mean 3.2785 |
| 16 | `records/121024_MFUTweaks/*` | 40 | `b6e0ae8` 2024-12-12 | yes | Dec 11 07:03 – 10:53 | 2.6.0.dev20241203+cu124 | 1480 / mean 3.2785 |
| 17 | `records/121724_SparsifyEmbeds/165384d5-….txt` | 1 | `f7b4b6e` **2025-01-05** (after #18) | yes (`print0(code)` style) | **Dec 25 04:28** (record date 12/17) | 2.6.0.dev20241223+cu126 | 1490 / 3.2794 / 213.1 s |
| 18 | `records/010425_SoftCap/31d6c427-….txt` | 1 | `45544f5` 2025-01-04 (same as code; dir renamed `ccc4686`) | yes | Jan 4 08:29 | 2.6.0.dev20241231+cu126 | 1390 / 3.2785 / 204.3 s |

Two embedding layouts exist (verified from each script's logging code): #11–#13 write
`'='*100 + '\n' + code + '='*100`; #14–#18 call `print0(code); print0('='*100)`. `explore_tools/logsrc.py`
handles both; extraction is byte-exact (the #16 and #18 extractions are byte-identical to their commits).
All logs in a directory embed one identical source (`match_logs.py`: one source hash per directory).

All logs: `val_loss_every = 125`, 8 GPUs, NVIDIA H100 80GB HBM3.

---

## 4. Anchor reconciliation

`explore_tools/match_logs.py` compared each embedded source against **every** version of the script on any
ref (279 distinct blobs, both filenames, 2024-11-01…2025-01-31, including PR heads and pre-rewrite twins).

| # | code commit (best match to log) | relation | log commit | README row commit | PR | disagreements |
|---|---|---|---|---|---|---|
| 11 | `ff93d20b` (bluecoconut, 2024-11-12; side branch, PR #25, merged `2e7b3df7`) | cosmetic: 1 line (`r"""` docstring prefix; runtime string value identical) | `551ec61` | `62755357` | #25 (carried it transiently; reverted next commit `df8d0a0d`) | **`551ec61` (brief) holds #10 code** — 32 changed lines vs log (no U-net, LRs 0.3/0.002/0.02/0.02 vs 0.6/0.008/0.04/0.04, 3242/926 vs 3000/900 iters). First-parent `607f2a3` (2024-11-20) differs substantively (lazy on-device rotary `inv_freq`, logging restructure, `return loss` vs `loss.float()`, renamed U-net attributes). |
| 12 | `ebce774` (2024-11-20) | cosmetic: one comment line (`# Use of FlexAttention contributed by @bkoszarsky`) absent from the log | `ebce774` | `62755357` | — | none |
| 13 | `759b8fc` (2024-11-25) | cosmetic: the `attn_blocksize = …` statement sits 6 lines earlier in the log (before the step-10 timer reset instead of after) and its comment lacks "By @fernbear.bsky.social" | `759b8fc` | `62755357` | — | none of substance; the moved statement uses only `step`/`args`, and the lines it moves across touch only timing variables |
| 14 | `49c3afa` (2024-12-05) | whitespace only (final newline) | `2305e1e` (logs committed *before* the script) | `45a8dc24` | — | `77732eb` ("new KoszarskyB record") has `num_iterations=1550`; the logs ran 1530 (fixed in `49c3afa`) |
| 15 | **none in any ref**; nearest master `8011fe7` (49 lines, itself internally inconsistent: loop passes `attn_blocksize=` to a forward taking `sliding_window_size`) | — | `2035d87` | `4442c394` | #43 merged (loader `+1` shard check, `perf_counter`); #45 not merged | **gist 1fed9527 @866bad1 `train_gpt2.py` = log source up to final newline**; gist log `z` embeds byte-identical source |
| 16 | `0690f2c` (2024-12-11) | identical | `b6e0ae8` | `b6e0ae8` | — | none |
| 17 | **none in any ref**; first master code with #17 mechanisms is `5db4da3`/`32fce07` (2024-12-31/2025-01-01), ≥320 lines away (Keller rewrite) | — | `f7b4b6e` (2025-01-05, after #18) | `a146238e` (2024-12-31, log "tbd") | — | **gist dff723d9 log (run Dec 23) embeds the same code except 3 lines printing attention lambdas at each validation**. Both logs post-date the 12/17 record date by 6–8 days. |
| 18 | `45544f5` (2025-01-04) | identical | `45544f5` | `45544f5` | — | brief's 558-line claim verified |

---

## 5. Snapshot rule outcome

`snapshot_N` := the code that produced record N's committed log = the embedded source, byte-exact. It is
determined for all eight records. Where a commit holds the same code (up to whitespace/cosmetics) that commit
is the **snapshot SHA**; where none does (#15, #17) the snapshot SHA is "none (log-embedded only)".

For the repo context of each snapshot (requirements, Dockerfile, data scripts) the pristine tree exported to
`records/rNN/` is:

* the snapshot SHA when there is one; otherwise
* **the master first-parent HEAD at the time the committed logs were produced** (#15: `155afd5`, the HEAD
  from 2024-12-05 07:07 until 2024-12-09 15:45, bracketing the Dec 8 runs; #17: `e76ab99`, HEAD from
  2024-12-17 02:43 until 2024-12-31, bracketing both the 12/17 record date and the Dec 23/25 runs). This is a
  **weaker anchor** than a code commit: its own script is *not* the snapshot. The log-adding commits
  (`2035d87`, `f7b4b6e`) were rejected as tree anchors because `f7b4b6e` post-dates #18's code and would put
  #18's script and README into #17's tree.

Embedded vs candidate, whitespace-only vs substantive: see §4 "relation" column. Summary: identical (#16,
#18), whitespace-only (#14), cosmetic only (#11 docstring prefix, #12 comment, #13 statement order + comment),
no commit (#15, #17; corroborated by external gists as stated).

What the snapshots are built on (established by diffing embedded sources against master; used by §7):
#11 = `7c1a1af` (#10 script `41bffe9`) + U-net + hparams from @brendanh0gan's fork; #15 = master `155afd5`
+ PR #43 + U-net VE + type-annotation/rename edits; #17 = `0690f2c` (#16) + sparsified VE, half-truncated
rotary, dropped attention layer, 50257-row embeddings, Muon `param_lr`/uneven-split support, logging changes.

---

## 6. Shared commits and hunk-level splitting

* No snapshot needs to be split out of a shared commit: every record has its own embedded source, so the
  r(N-1)→rN diff is computed between embedded sources, not between commits.
* Commits that carry more than one record (verified): `0344d5f` (adds #17 README edits and #18 images);
  `45544f5` (#18 code + #18 log + README rows for #11–#18 re-pathed); the 2024-12-30/31 series (#17
  mechanisms re-implemented on master **and** the rewrite that #18 inherits); `8011fe7`…`0690f2c` (Keller
  integrating #15's code and then #16's changes on master).
* Hunk-level *attribution within* a record's diff (e.g. which #15 hunks are "U-net VE" vs "assorted code
  optimizations"; which #18 hunks come from the rewrite vs the softcap commit) uses README descriptions,
  PR #43/#45 contents, the per-commit table in §7 and neighbouring snapshots. Such attributions are marked
  `provenance: inferred` in the ledgers; hunks whose identity is evident from the diff alone are `direct`.

---

## 7. Intervening commits touching the script

Full per-commit table (138 commits, 2024-11-09…2025-01-05, `--full-history`, merges excluded):
[`explore_tools/intervening_table.md`](explore_tools/intervening_table.md). Assignment rule (deterministic):
a commit belongs to the first record whose code *base* has it as ancestor-or-self (bases in the table
header). A visibility column reports how much of the commit's text survives in that snapshot; low
visibility means superseded or reverted before the snapshot. Counts: r11 2, r12 12, r13 2, r14 23, r15 2,
r16 26, r17 0, r18 75, after-r18 2.

Grouped summary (verified from net diffs; "why" = lineage + content):

| interval | commits | belongs to | why / what |
|---|---|---|---|
| ≤ #11 log | `3b3d650`, `7c1a1af` | r11 | #10-era edits in the #11 base script `41bffe9` |
| #11 → #12 | `c3e1443`…`95c47b8` (8, 2024-11-11), `299faa6` (11-14), PR #25 `ff93d20b`/`df8d0a0d`, PR #27, `607f2a3` (11-20), `ebce774` | r12 | Keller's 11-11 edits to the #10 script (lazy on-device rotary `inv_freq`, docstring, `print0` logging helper moved earlier, `return loss`) and his 11-20 re-implementation of the U-net (`num_encoder_layers` naming, skip added before the block call) are the base `ebce774` builds on. `ff93d20b` carries #11's code but is reverted by `df8d0a0d` within PR #25, so its net effect on master is nil. **Refactors/logging only, plus the #12 change itself.** |
| #12 → #13 | `77001e6`, `759b8fc` | r13 | `759b8fc` is #13; `77001e6` is a one-line edit |
| #13 → #14 | PR #31 (`4232efe6`, `1e6fa0b1`: no_sync restructuring of the accumulation loop), `d517ae7`…`dc96f1a` (11-25/26), side web edits `5dd88367`, `c94d5d42`, `af912632`, `732b6b2`, `8dd2fec`, `96e82b1`, `30370b5` ("rotary refactor and leave cos/sin in fp32. fixes #39"), `caaa59f`, `d012644`, `4c7bb55`, `2305e1e`, `77732eb`, `49c3afa` | r14 | infra/refactor + rotary precision change + #14's value embeddings |
| #14 → #15 | Fern `33feba78` (peak-memory logging; **not** visible in #15), PR #43 `8e5bbd60` | r15 | PR #43 is in #15's code (loader shard check `+ 1`, `time.perf_counter`) |
| #15 → #16 | PR #44 (`c643e6d6`, `03632607`: logging), `45a8dc2`, `8011fe7`…`594bdc9` (2024-12-09), `2c4bb7f`, `0690f2c` | r16 | Keller integrating #15's code on master, then #16's split VE / block-level window / separate block masks |
| #16 → #17 | *(none on master)* | — | #17 was built off-repo on `0690f2c`. Master's 2024-12-13…16 web edits (`e6f2eb5`…`e76ab99`) are **absent from #17** (it keeps #16's Muon normalisation order, `arange(512)`, window formula) |
| #17 → #18 | `e6f2eb5`…`e76ab99` (12-13…16), `74b982a`…`f4829b6` (12-30/31, incl. rename `5db4da3`), `32fce07`, `45544f5` | r18 | 12-13…16: Muon normalisation after transpose, `total_num_blocks`, window formula rewrite, checkpoint code commented. 12-30/31: Keller's rewrite re-implementing #17 (`use_attn=(i != 7)`, `ValueEmbedding(vocab, dim)`, `.bfloat16()` VE casts) with new loader, single Adam with param groups, `cooldown_frac`, `bf16_embeds`, `model.cuda()` without global bf16 cast, removal of `coordinate_descent_tuning`/`set_stance`/`no_sync`/grad division. `45544f5`: softcap 15, 1390 steps. |

---

## 8. Facts that change the build (consequences, inferred from the verified facts above)

* **Bootstrap file** = #11 embedded source (= `ff93d20b` up to the docstring prefix), not `551ec61`.
* **#18 flags.** Besides `softcap` (30→15, with coupled `num_iterations` 1490→1390), the r17→r18 snapshot diff
  needs its own switches for: small-parameter/rotary precision (bf16 vs fp32), LR schedule form (absolute
  cooldown iterations vs fraction), embedding row count (50257 vs 50304), loader shard-advance rule,
  gradient-accumulation scaling, and throughput-only compile settings. None of these is in the proposal's
  arm list; they will be reported as non-arm flags.
* **Loader invariance.** #11–#17's loaders advance shards per rank with a rank-dependent threshold that also
  depends on `world_size` (verified in code; e.g. at 8 GPUs with 100M-token shards, ranks 6–7 switch shard one
  step before ranks 0–5 — arithmetic in `CHANGES/infra.md` once written). A literal single-GPU run of the #11
  code therefore does not see the same token stream as the 8-GPU record past the first shard boundary
  (≈ step 189). Step 1's unified batch convention must preserve each record's 8-GPU stream to remain a no-op.
* **#14→#15 loader change.** PR #43 changes the shard-advance test from `pos + B >= len` to `pos + B + 1 >= len`;
  this is a data-order change at shard boundaries, to be classified in `ledger/r15.yaml`.
* **Torch drift.** The records span torch 2.5.1 → 2.6.0 nightlies; FlexAttention API (`create_block_mask`,
  `BlockMask.from_kv_blocks`, `enable_gqa`), `nn.Buffer`, `torch.compiler.set_stance`,
  `dist.init_process_group(device_id=…)` all appear inside the window. Only drift that can be established
  statically becomes a `compat:` commit; the rest goes to `DEFERRED_RUNTIME_CHECKS.md`.

---

## 9. Addenda found during the replay (verified)

* The README in the #12 tree (`ebce774`) says: *"The `train_gpt2.py` in this repo is the 11/08/24 record. To run the
  latest 11/10/24 record, use the code in its reproducible log."* — the maintainer's own statement that #11's code is
  the log-embedded source (finding 2).
* Upstream's `data/cached_fineweb10B.py` states "Each chunk is 100M tokens", the shard size assumed by the loader
  simulations (`explore_tools/shard_sim.py`). With it: the #11–#17 loaders' 8-GPU and 1-GPU streams diverge at step
  189; the #17 and #18 streams (both at 8 GPUs) differ on 1,201 of 1,390 steps; under the per-rank rule ranks drift
  apart at every shard boundary (in a scaled-down harness run 268 of 600 steps mixed two shards).
* #15's hand-built block mask drops in-window (query, key) pairs: 24.6% at window 64, 49.6% at 128, 3.5% at 1792
  (`tools/blockmask_check.py`). #16's block window re-adds them.
* Window schedules: the r13 and r15 token expressions agree at every step for every N tested; the r16 and r18 block
  expressions agree at the records' N (1390, 1480, 1490) but differ at isolated boundary steps for N sharing factors
  with 27 (e.g. N = 1530 step 1020; RQ3b-scaled N = 1110, 765) (`tools/window_schedule_check.py`).
* The record README's fewer-GPU claim ("you should get the exact same loss curve") is replaced at #16 by "this does not
  change the expected behavior of the training".
