# records/MANIFEST.md — snapshot manifest, records #11–#18

`snapshot_N` = the code that produced record N's committed log = the source embedded at the top of that log,
extracted byte-exact by `explore_tools/logsrc.py` (see `EXPLORATION.md` §3–§5). Each `records/rNN/` holds:

* `tree/` — `git archive <tree SHA> | tar -x` (pristine repository state; its training script is *not*
  necessarily the snapshot — see "relation");
* `log_source/<script name>` — the embedded source, byte-exact (THE snapshot);
* `ANCHOR.txt` — the SHAs, log path and hashes below.

Directories are exported one at a time during the replay (Step 1 for r11, Step 2 for r12…r18).

| record | snapshot SHA (code commit) | tree SHA exported | date (log run) | filename | lines | sha1(log source)[:12] | relation of snapshot SHA's script to log source | evidence | confidence |
|---|---|---|---|---|---|---|---|---|---|
| #11 | `ff93d20b` (side branch, PR #25, merged via `2e7b3df7`) | `ff93d20b` | 2024-11-10 | `train_gpt2.py` | 568 | `36138bfa4e2f` | cosmetic (1 line: `r"""` docstring prefix; same runtime string) | 1 log, `551ec61`; brief's `551ec61` has #10 code (32-line substantive diff) | **high** |
| #12 | `ebce774` | `ebce774` | 2024-11-20 | `train_gpt2.py` | 575 | `5f26f6fed98c` | cosmetic (1 comment line) | 1 log in same commit | **high** |
| #13 | `759b8fc` | `759b8fc` | 2024-11-24/25 | `train_gpt2.py` | 578 | `9084e4d297aa` | cosmetic (statement moved across timing-only lines; comment text) | 7 logs, one source; same commit | **high** |
| #14 | `49c3afa` | `49c3afa` | 2024-12-05 | `train_gpt2.py` | 565 | `7e2e3af9668f` | whitespace only (final newline) | 38 logs (40 originally), one source; logs committed at `2305e1e` | **high** |
| #15 | **none** (code in no ref) | `155afd5` (master HEAD during the Dec 8 runs) — weaker anchor | 2024-12-08 | `train_gpt2.py` | 584 | `b818ea978d83` | n/a; tree script differs substantively | 75 logs, one source; README first linked gist `1fed9527` whose `train_gpt2.py` (@`866bad1`, 2024-12-08) equals the log source up to final newline and whose log `z` embeds it byte-identically | **high** for the code (embedded + external corroboration); tree anchor weaker |
| #16 | `0690f2c` | `0690f2c` | 2024-12-11 | `train_gpt2.py` | 611 | `e65a1ad22757` | identical | 40 logs, one source; logs at `b6e0ae8` | **high** |
| #17 | **none** (code in no ref) | `e76ab99` (master HEAD 12-17…12-31) — weaker anchor | 2024-12-25 | `train_gpt2.py` | 628 | `f0fff92d05ae` | n/a; tree script (#16 + Keller's 12-13…16 edits) differs substantively | 1 log, committed `f7b4b6e` (2025-01-05). README linked gist `dff723d9` (log run 2024-12-23) whose embedded code differs only by a 3-line lambda print at validation. Caveat: both runs post-date the 12/17 record date; the README links them as the record's logs | **high** for "code that produced the committed log" (the rule); the record-date gap is reported, not resolvable from the repo |
| #18 | `45544f5` | `45544f5` | 2025-01-04 | `train_gpt.py` | 558 | `8c47c97f5ee7` | identical | 1 log in same commit | **high** |

No snapshot is ambiguous at medium or lower under the snapshot rule. Discrepancies with the brief (reported,
not ambiguities): #11's code is not in `551ec61`; #15 and #17 have no code commit; #18's snapshot diff
from #17 is not softcap-only (the softcap *commit* is).
