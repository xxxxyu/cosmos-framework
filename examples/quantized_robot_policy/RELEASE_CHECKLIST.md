# Quantized Robot Policy Release Checklist

A release tag is valid only after all required rows pass from a fresh checkout.
The RoboLab BF16 baseline is intentionally not part of this release gate.

| Gate | RoboCasa365 | RoboLab |
|---|---:|---:|
| Locked runtime installs without external site-packages | required | required |
| Runtime import/Marlin self-check | required | required |
| Strong bundle hash/tensor validation | required | required |
| Direct-load server reaches ready state under 24GB | required | required |
| Deterministic replay smoke | required | required |
| Closed-loop simulator smoke | required | required |
| Full benchmark already documented | required | required |

Release procedure:

1. Start from a clean worktree at the proposed commit.
2. Run `examples/quantized_robot_policy/setup.sh` with no external
   `PYTHONPATH` or `LD_LIBRARY_PATH` overrides.
3. Run each pipeline's `validate`, `replay`, and one-episode `rollout` command.
4. Confirm logs contain no source-checkpoint access, network fallback, OOM, or
   simulator/server crash.
5. Run targeted unit tests and `git diff --check`.
6. Update both benchmark documents with release-environment smoke results.
7. Create one annotated tag at the tested commit and never move it.
