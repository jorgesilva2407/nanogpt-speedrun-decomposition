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
