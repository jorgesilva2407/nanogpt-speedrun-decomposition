# saved_results/

The compact copy of the study's results that the tables are built from. Layout (what `analysis/load.py` reads):

```
h100/<configuration>/seed<k>/{meta.json, metrics.jsonl, timing.jsonl, wall_seconds, resolved.json, DONE}       28 learning runs, host vast-h100
h100/throughput/<configuration>/vast-h100/<timestamp>/run/seed0/{...}                                         21 throughput runs (300 steps each)
h100/_logs/59638827/{jobs.txt, jobs_full_p1.txt, run_tiers.sh}                                                the job lists and tier script of the H100 sweep
a100/<configuration>/seed<k>/{...}                                                                            24 learning runs, host lab-a100
```

Never pool `h100/` and `a100/`: both contain `fp_r11` and `fp_r18` and `analysis/load.py` refuses mixed hosts.

**What was trimmed, compared with the Hugging Face datasets** (`speedrun-decomp/speedrun-results`, `speedrun-decomp/speedrun-results-a100`):
`metrics.jsonl` keeps the `kind=val` rows only (the training-loss rows are dropped; the analysis does not read them), and `timing.jsonl` keeps
`step`, `gpu_ms` and `compile_warmup` (the wall-clock field is dropped). Values are copied unchanged. Not included: raw worker logs, nvidia-smi captures,
self-tests and the A100 throughput check (not a result of the study; `DEVIATIONS.md` section 7).

Rebuild from the full datasets: `python -I tools/export_saved_results.py <h100 dir> <a100 dir> saved_results`.
Rebuild the tables: `python -I analysis/tables.py --results saved_results/h100 --results-a100 saved_results/a100 --out reports/table.md`.
