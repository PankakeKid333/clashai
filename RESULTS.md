# Results so far

Both runs trained on a 5-core, CPU-only dev sandbox (no GPU). 40 games per
matchup, 20 in each seat, seeded; intervals are 95%.

## Run 1 — Hog 2.6 vs the random bot (274k decisions, 35 min)

Win rate of the trained bot:

| Opponent | Result | Win rate |
|---|---|---|
| noop | 40-0-0 | **100%** |
| first-affordable | 35-4-1 | **89.7%** |
| push | 33-7-0 | **82.5%** |
| greedy | 23-16-1 | 59.0% (too close to call) |
| random-legal | 17-23-0 | 42.5% (too close to call) |
| defend | 9-31-0 | 22.5% — defend wins |
| patient | 4-36-0 | 10.0% — patient wins |

## Run 2 — + self-play ladder (warm-started from run 1, +275k decisions, 40 min)

The ladder plays half its battles against the current version of itself, a
third against older saved versions, and the rest against scripted bots. Its
ladder Elo rose 1232 → ~1330 during the run. Against the fixed baselines:

| Opponent | Result | Win rate |
|---|---|---|
| noop | 40-0-0 | **100%** |
| first-affordable | 36-4-0 | **90.0%** |
| push | 25-15-0 | 62.5% (too close to call) |
| greedy | 21-18-1 | 53.8% (too close to call) |
| random-legal | 15-25-0 | 37.5% (too close to call) |
| defend | 9-31-0 | 22.5% — defend wins |
| patient | 4-36-0 | 10.0% — patient wins |

## What this says

- 550k decisions is early. The bot has solid fundamentals (it beats three of
  the gym's scripted playstyles and never loses to noop) but still overcommits
  to Hog pushes, which is what *defend* and *patient* punish. The ladder is
  the fix: leave stage 2 running overnight (`python train.py --opponent self
  --minutes 600 --save-dir runs/hog26-self`) and re-measure with
  `python evaluate.py --run runs/hog26-self/bot --opponent all`.
- Sampled play currently beats argmax play: the policy's logits are still
  flat, so its single most-likely move is often a weak hold. More training
  sharpens this.
- Machine speed (see `benchmark.py`): this sandbox did 1,425 decisions/s raw
  engine and ~130–225/s end to end during training. Run `python benchmark.py`
  on your machine and compare the two numbers.
