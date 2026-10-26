# DeepSpeed launcher configurations

Selected per run by `train.sh --ds-config PATH` (relative to the repo root); the default is
open-instruct's own stage config (`castor/posttrain.py` `OI_DS_CONFIG`). Every file here differs
from that default only in scheduling settings; `diff` against
`/work/harrychang08/castor/third_party/open-instruct/configs/ds_configs/stage3_no_offloading_accelerate.conf`.

- `sft_zero3_profile.json`: the current SFT config + `wall_clock_breakdown`, `steps_per_print 1`
  (profiling only; never a training configuration).
- `sft_zero3_tuned.json`: candidate (c): ZeRO-3 with 5e8 reduce/prefetch buckets and a 1e6
  persistence threshold.
- `sft_zero1.json`: candidate (b): ZeRO-1, otherwise as open-instruct's `stage2_accelerate.conf`.
- Candidate (a) is open-instruct's shipped `stage2_accelerate.conf`, passed by absolute path.
