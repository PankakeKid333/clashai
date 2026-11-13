# Royale AI

A Clash Royale bot trained with reinforcement learning on
[RoyaleGym](https://royalegym.github.io/RoyaleGym/) — a deterministic, integer-only
reimplementation of the game (Rust engine, Python bindings). No GPU needed: the
battles run on the CPU at ~1,300 sim steps/s per core, and the PPO network is
small enough to train on a laptop. If you do have a CUDA card, pass
`device="auto"` in `train.py` and torch will use it.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install "royalegym[all]" royalelearn
```

Python 3.12. `royalelearn` pulls in torch and safetensors; on Windows with an
NVIDIA card install torch first (see RoyaleGym's Install page).

## Train

```bash
python train.py --minutes 60                    # Hog 2.6, auto curriculum
python train.py --deck random --minutes 60 --warm-start runs/hog26/stage2-self
```

The curriculum (`--opponent auto`, the default):

1. **Stage 1** — plays a bot that moves randomly until `--promote-at` decisions
   (default 150,000). This teaches the basics fast: cycle discipline, elixir
   trades, when a Hog push connects.
2. **Stage 2** — the ladder: about half its battles against the current version
   of itself, a third against older saved versions of itself, and the rest
   against the gym's scripted bots (defend, push, patient). Two bots that keep
   getting better, competing against each other.

Checkpoints save on Ctrl+C and at the `--minutes` deadline; run the same
command again to carry on. `runs/<deck>/stage2-self/metrics.jsonl` has one line
per update; `env/crown_diff` (crowns taken minus lost) and `ladder/elo_readout`
are the numbers to watch.

## Measure it

```bash
python evaluate.py --run runs/hog26/stage2-self --opponent all --games 60
```

Win rate against the whole baseline ladder (noop, random, greedy, defend, push,
patient, itself) with 95% confidence intervals. `too close to call` means not
enough games, not a tie.

## Watch it

The easiest way — a self-contained HTML animation, no installs, just open it
in a browser. Blue (bottom) is the bot; you get play/pause, speed, a timeline
scrub, elixir bars, crowns, hp bars, and a flash where each card was played:

```bash
python visualize.py --run runs/hog26-self/bot --opponent defend --out battle.html
python visualize.py --run runs/hog26-self/bot --opponent patient --seed 7
```

Or the gym's own live viewer:

```bash
python watch.py --run runs/hog26/stage2-self --save my_bot_battle.msgpack
royaleviser my_bot_battle.msgpack
```

`visualize.py --trace my_bot_battle.msgpack --out replay.html` renders a saved
battle too.

## How fast is this machine?

```bash
python benchmark.py
```

Prints raw engine speed and end-to-end training throughput. Compare your two
numbers against another machine's. Reference: a 5-core CPU-only dev sandbox did
~1,350 decisions/s raw and ~130/s end to end; the RoyaleGym docs' RTX 4070 Ti
reference did 16,384 steps per 36–71 s with 32 battles at once.

## Files

| File | What it is |
|---|---|
| `envs.py` | Battle environments and the reward (win + crowns + tower damage + elixir trades) |
| `decks.py` | Deck lists |
| `train.py` | The trainer: auto curriculum, warm-starting, time budget |
| `warmstart.py` | Carries weights across a deck or opponent change |
| `evaluate.py` | Win rates with error bars |
| `benchmark.py` | Machine speed, to compare computers |
| `visualize.py` | Renders a battle to a self-contained HTML animation |
| `watch.py` | One battle, saved for the `royaleviser` viewer |

## Honest expectations

From-scratch PPO beats random and scripted bots within a few hundred thousand
steps, and the self-play ladder keeps improving it from there — but a bot that
plays like a strong human needs far more steps, ideally started from a clone of
real players (RoyaleGym's [Cloning a Bot](https://royalegym.github.io/RoyaleGym/)
guide). The pipeline here is the same either way; a clone only replaces the
starting weights.
