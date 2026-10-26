# Job-level compute for `tab:compute` (App. A), built for runner 061

Built 2026-10-12T17:15+08:00 on the NCHC login node, repo at `55ee277`. Read-only; no job submitted.

## Unit and method
- H200-h = `ElapsedRaw` (s) × `gres/gpu` from `AllocTRES` / 3600, summed over **every** job in the file, whatever its
  state. "Not completed" = state ≠ `COMPLETED` (FAILED, CANCELLED, TIMEOUT …). Same rule as
  `scripts/report_008.py` and `runs/016/sacct_jobs.psv`.
- Format of every `.psv`: `sacct -X -P -n -o JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList` (no header), the
  same 6 fields as `runs/016/sacct_jobs.psv`.
- One pull: `sacct -X -P -n --starttime 2026-08-01 -u harrychang08 -o JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList`
  → `raw/all_jobs_raw.psv` (881 jobs). An earlier window (`--starttime 2026-01-01 --endtime 2026-08-01`) returns 0.
- Castor jobs = name starts `NNN-`, **or** the job id appears in a `logs/nchc/<name>-<jobid>.out` filename
  (`raw/logs_map.txt`, 792 files). Every log id is in sacct with the same name; the only castor sacct job with no log is
  463835 (`009-eval-rel1B-b`, CANCELLED, 0 s). 784 castor jobs, 10,164.67 H200-h in all (1,509.89 not completed).
- Bucketing: `uv run --frozen python scripts/compute_buckets_061.py <this dir>` (058 step 0b: the path read `../compute_tools/bucket.py`, the bucketing session's scratch copy) (rules below; every castor job lands in exactly one file; files are
  disjoint, so the files sum to the castor total). Totals per file: `summary.json`.
- Verification: for every file, `sacct -X -P -n -j <the file's own ids, comma-joined> -o <same 6 fields>` was re-run and is
  byte-identical (sorted) to the file. The `-j` list of each file is its first column (`cut -d'|' -f1 FILE | paste -sd,`).

## Files (rows of `tab:compute`)

| file | selection | jobs | H200-h | of which not completed |
|---|---|---|---|---|
| `pretrain_reruns.psv` | names `011-i6198-d6198-seg[123]`, `003-s2-i6198-d4-seg[123]`, `003-s2-i4-d6198-seg[123]` (ids 449825–7, 444667/8/70, 444672/3/4; the ids notes/011 L77 and notes/009 L19 cite) | 9 | **2,230.39** | 1,152.03 (the three `seg1` TIMEOUTs, see A1) |
| `first_setup_twelve.psv` | `006-train-*` | 14 | **791.73** | 0.016 (2 FAILED starts, 449843 / 449847) |
| `genjudge_first_setup_twelve.psv` | `006-genjudge-*` | 12 | **21.92** | 0 |
| `partial_reruns_posttrain.psv` | `009-train-*` | 12 | **525.55** | 0.027 (4 FAILED starts 476670/2/4/6, dirty tree) |
| `genjudge_partial_reruns.psv` | `009-gj-*` | 13 | **20.54** | 6.094 (476697 CANCELLED 6.09 slow node; 4 FAILED starts 0.007) |
| `recipe_grid_train.psv` | ids under `…|sft` keys of `/work/harrychang08/castor/runs/008/driver/submissions.json` | 159 | **4,666.42** | 4.849 (1 CANCELLED, 2 FAILED) |
| `genjudge_recipe_grid.psv` | ids under `…|gj` and `…|gj.pre040` keys of the same ledger | 190 | **303.49** | 11.699 (3 CANCELLED, 31 FAILED) |
| `recipe_grid_refill.psv` | `008-refill` with id ≥ 517485 (= started after `report_008.py --start 2026-10-08T11:00`) | 4 | 0.0125 | 0 |
| `same_seed_rerun_posttrain.psv` | `011-train-dclm-baseline-1B-replicate-*` (489014/16/18/20) | 4 | **119.39** | 0 |
| `genjudge_same_seed_rerun.psv` | `011-gj-dclm-baseline-1B-replicate-*` | 4 | **7.52** | 0 |
| `ladder_train.psv` | ids of `runs/016/sacct_jobs.psv` with `-train-` | 26 | **394.50** | 0.071 |
| `genjudge_ladder.psv` | ids of `runs/016/sacct_jobs.psv` with `-genjudge-` | 32 | **51.05** | 0.046 |
| `checks_appC.psv` | 004-* (17), 005-* (2: judge smoke, judging 004's generations), 010-* (7), 012-* (11), 024-train-A/B + 024-gj-A/B (4) | 41 | **571.34** | 224.29 (see A3) |
| `tests_appA.psv` | 024-profile, 024-smoke-* (8), 025-* (6: GPU sharing `coloc-*`, `probe1-*`) | 14 | **23.89** | 0 |
| `smokes_other.psv` | pipeline smokes/mocks/probes outside 024/025: unprefixed `smoke-*`, `smoke_generate`, `judge-smoke` (002/004/005 era, 19), 006-smoke-train (3.45), 008-smoke/mock/probe and the 11 pre-window `008-refill` (33), 016-smoke-* (6), 007/021/027/040/046 smokes, 035-smoke-r2 (2.28) | 68 | 21.59 | 0.743 |
| `pretrain_checks_003.psv` | every `003-*` except the six `003-s2-*` (150M reproduction runs 443698/443699, 1B smokes, evals, diagnostics, dl-dclm) | 30 | 101.59 | 0.539 |
| `pretrain_rerun_evals.psv` | 009-eval-*, 009-smoke-convert, 009-stage2-after, 011-after | 11 | 130.66 | 14.760 (476650 CANCELLED 14.75) |
| `other_analysis.psv` | everything else castor: 007 (41.55), 022 (25.84), 020 billing probe (12.23), 021 (10.14), unprefixed 002–004-era jobs (convert/validate/panel/thru/step0/seedcheck/dl-dclm/valcmp/prep004/seedmps/uv, 8.47), 011 analyses (step0, wdist*, divergence, floor; 2.29), and < 1 h each for 006/008/016/027/035/044–057 analysis jobs | 140 | 103.78 | 15.41 |
| `held_035.psv` | 035-r2-seg1 520993 (CANCELLED, 8 GPUs, 79.32) + 035-r2-seg2 521000 (CANCELLED, never ran, 0; added by `sacct -j 521000`, absent from the `-u --starttime` pull) | 2 | 79.32 | 79.32 |

Sums for the card's rows (from the files above):
- **1B pretraining reruns (3):** 2,230.39 = 744.89 (011, i6198-d6198) + 744.19 (003-s2 i6198-d4) + 741.31 (003-s2 i4-d6198).
- **Twelve, first set-up:** train 791.73 + gen/judge 21.92 = **813.64**.
- **Partial reruns' eight:** train 525.55 + gen/judge 20.54 = **546.09** (not completed 6.12).
- **Recipe grid 156:** 4,666.42 + 303.49 + 0.0125 = **4,969.92** = `cells.json` `total_h200_hours` 4969.9197 exactly.
- **Same-seed rerun's four:** 119.39 + 7.52 = **126.91**.
- **150M/300M 24:** 394.50 + 51.05 = **445.55** (not completed 0.117).
- **App. C checks + App. A test runs:** 571.34 + 23.89 = **595.23** (+ 21.59 if the generic smokes count as test runs).
- All castor jobs (every file, `held_035` included): **10,164.67**, of which not completed 1,509.89 (1,152.03 of it the planned `seg1` TIMEOUTs, A1).

## Cross-checks against the notes
- notes/011 L77 744.9 → 744.887 ✓; notes/009 § 1 744.2 / 741.3 → 744.191 / 741.309 ✓; notes/009 L58 1,485.5 ✓ (1,485.50).
- notes/006 L129–130: train 65.7–66.8 per model → completed jobs 65.67–66.84 ✓, total 791.7 ✓ (791.73 includes 0.016 of failed
  starts; notes say "+0.02 two failed starts" separately); gen+judge 1.83 (1.75–1.89), 21.9 → 1.826 (1.752–1.889), 21.92 ✓.
  Dry run 444525 = 3.45 ✓ (notes "3.5"; here in `smokes_other`).
- notes/009 L60–62: post-training 525.5 ✓ (completed 525.52), gen+judge 14.45 ✓ (completed 14.45), losses 6.09 + 0.03 ✓; evaluation
  129.5 ✓ (the six 009-eval/stage2-after jobs = 129.47; my file adds 009-smoke-convert 0.03 and 011-after 1.10).
- 008: train 4,666.42 / train:COMPLETED 4,661.57 / genjudge 303.49 / genjudge:COMPLETED 291.79 all equal `cells.json` ✓;
  completed gen+judge per model 291.786 / 156 = 1.870 (the "1.87 owed").
- 016: 58 jobs = `runs/016/sacct_jobs.psv` and = every `016-train-*`/`016-genjudge-*` in sacct ✓; 445.55 all, 445.44 completed.
- Same-seed rerun's four (second set-up): 29.64–29.97 train per model; gen+judge 1.85–1.89.

## Answers to the two questions
- **Grid 4,969.9 (`runs/008/cells.json`)**: includes generation + judging (303.49, kind `genjudge`) and failed/cancelled jobs
  (16.55 = train 4.85 + gen/judge 11.70) and the 4 refill-driver jobs (0.0125). Job set = the driver ledger (156 `sft` keys,
  156 `gj` keys + 1 `gj.pre040` key; 349 ids) — `scripts/report_008.py` L104–115. Excludes 008 smokes/mocks/probes,
  the 11 earlier `008-refill` jobs, `008-analyze` and `008-audit-draw` (listed in `cells.json` `unledgered_008_jobs` only
  for the last two).
- **016's 445.4**: the 58 jobs include generation + judging (51.05). 445.4 is **completed jobs only**: all 58 jobs sum to
  445.55, completed 445.44, not completed 0.117 (10 FAILED). notes/016 L78 says 446.4 + 0.17 (card todo already says → 445.4
  and 0.12; artifact gives 0.117). Excludes 016 smokes (5.15), overlength, anchor-a0, signal.

## Ambiguities
- A1. **Pretraining `seg1` TIMEOUTs** (384.0 H200-h each, 1,152 total) are planned 2-day segments that the next segment resumed;
  by the "state ≠ COMPLETED" rule they count as not completed. The table probably should not show them as lost; decide.
  Same for the `seg3` 1-s no-ops (COMPLETED).
- A2. Partial reruns' pretraining was submitted by prompt 003 (names `003-s2-*`), verdict in 009. 003's other 30 jobs
  (incl. two 150M reproduction runs 443698 / 443699, 28.35 + 28.46 H200-h, and 003-eval-released150M 26.33) are in `pretrain_checks_003.psv`; no card row names them
  (App. B / C.4 order check?). Likewise the reruns' evaluation/conversion (`pretrain_rerun_evals.psv`, 130.66).
- A3. `checks_appC.psv` not-completed 224.29 is mostly **productive**: 004-sftdpo-s1/s2 (80.4 + 79.8) are FAILED only on a final
  `assert config.json` in `posttrain_run.py::finish` after training (notes/004 L119–120 use these runs); 012-dpo-B 453364
  TIMEOUT 64.0 is a real loss (slow node, rerun 457231).
- A4. 024-train-A/B + gj-A/B (99.46) are both App. C.6.2's check and App. A.2's speed-up measurement; put in checks. 024's
  smokes/profile and all of 025 in `tests_appA`.
- A5. Which smokes are "A's test runs": I put only 024/025 there; the other pipeline smokes (21.59) are separate.
- A6. 027 has no training jobs (config-check models of C.6.3 are the grid's twelve main-recipe models, inside the 156);
  its 4 analysis jobs (0.50) are in `other_analysis`. The "twelve trained again under the second set-up" are therefore not a separate row.
- A7. 007/021/022 mechanism jobs (77.5 H200-h) and 020's billing probe (12.23) have no row in the card list; they are in
  `other_analysis` and in the 10,164.67 total.
- A8. Excluded as non-castor: 97 sacct jobs without `NNN-` prefix and without a castor log (ids 415456–427924 astra
  162/166/168 work, 1,418.6 H200-h with them); 16 tiny jobs 476277–476300 (`base1`, `bind1`, `benchA`, …; 1 GPU, ≤ 10 s each,
  ≈ 0.01 H200-h) are of unknown owner, excluded.
- A9. `held_035`: 035-r2 seg1 ran 9.9 h on 8 GPUs then was cancelled (79.32); if 035 is killed these hours still count toward
  `\computeTotal` (they are inside 10,164.67) but have no row.
- A10. No running/pending jobs at build time (`squeue` empty); 061's own jobs, if any later, are not included.

## Runner 061's choices for `tab:compute` (2026-10-12; declared in notes/061)
- Copied here byte for byte from the bucketing session's scratch directory (`raw/` is not committed: it holds other
  projects' jobs of the shared account). The bucketing script is committed as `scripts/compute_buckets_061.py`
  (it reads the full sacct pull and 008's driver ledger on /work).
- **Not completed** in the table = Slurm state other than `COMPLETED`, **except** a planned two-day pretraining segment
  (`*-seg1`, state TIMEOUT) that the next segment resumed (A1): those three 384-H200-h segments count as completed.
  004's two FAILED SFT → DPO chains (A3) keep their Slurm state: they did not complete as jobs.
- Each model set's row includes its generation and judging (shown in a column of its own), so the grid row equals
  `cells.json` `total_h200_hours` (A, card § Figures / tables). App. C's checks and App. A's test runs are one row
  (`checks_appC` + `tests_appA`); all other castor jobs (`smokes_other`, `pretrain_checks_003`, `pretrain_rerun_evals`,
  `other_analysis`) are one "other jobs" row, so the total covers every castor job.
- `held_035` (79.32) is left out of the table and the total until 035's fate is decided (HELD-035).

## 033 sitting i (2026-10-17): 035 of record
- `035.psv`: every 035 job, `sacct -X -P -n -j 517666,517723,520993,521000,534947,534974,560417,560418,560419,561064,561065,561066,561067,561069,561070,561071,561072,561663 -o <same 6 fields>` (18 jobs, **873.21** H200-h = notes/035 § 9's 873.2; 521000 never ran, 0). It is the `tab:compute` row "second same-seed rerun: pretraining and post-trained models"; its gen + judge column = the four `035-gj-*` jobs.
- To keep the files disjoint, the two 035 gate jobs bucketed by 061 moved here: 517666 (`035-smoke-r2`, 2.28) out of `smokes_other.psv`, 517723 (`035-floor-dryrun`, 0.016) out of `other_analysis.psv`; `summary.json` updated for those two files and `035`. `held_035.psv` (its two jobs, 520993 and 521000, are in `035.psv`) is removed, and its `summary.json` entry with it.
- Not completed: the planned segment 534947 (`035-r2-seg2`, TIMEOUT, resumed by seg3) counts as completed, as 011's; seg1 520993 (cancelled by hand at step 7,505, D088a; seg2 resumed from its checkpoint) keeps its Slurm state, 79.3.

## 093 (2026-10-19, D129): 080 and 089 of record
- `080.psv`: the 25 jobs of 080's grid driver (13 `080-train-*`, one FAILED on hgpn071 and retried, and 12 `080-gj-*`), the ids of `runs/080/compute_grid.psv` re-pulled as `sacct -X -P -n -j <those ids> -o JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList`: **380.62** H200-h (not completed 0.12). 080's swap-mixture build, WildGuard labelling, smokes and analysis jobs are not in it (as 008's row leaves out the grid's smokes and analyses).
- `089.psv`: every `089-*` job (`sacct -X -P -n -u harrychang08 --starttime 2026-10-17 --name <the 089 names>`; none before Oct 17): 33 jobs, **36.42** H200-h (not completed 3.52, the first launch's seven dose arms, notes/089 deviation 3).
- `tab:compute` rows "main data recipe, other safety data" (080, gen + judge = its `080-gj-*` jobs) and "projection transplant" (089); the total becomes 10,956.3 + 417.04 = 11,373.3.

## 091 (2026-10-19, D129): transplant replication of record
- `091.psv`: every `091-*` job (`sacct -X -P -n -u harrychang08 --starttime 2026-10-19 -o JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList`, rows named `091-*`): the `--limit 16` smoke and the 15 arms, 16 jobs, **23.28** H200-h (not completed 0).
- `tab:compute` row "projection transplant, replication" (091), or folded into 089's "projection transplant" row (36.42 + 23.28 = 59.70) if /direct prefers one row; the total becomes 11,373.3 + 23.28 = 11,396.6.
