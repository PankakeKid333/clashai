"""Train a Clash Royale bot, Hog 2.6 first, then any deck.

    python train.py --minutes 60                    # Hog 2.6, auto curriculum
    python train.py --deck random --minutes 60      # every deck, from scratch
    python train.py --deck random --warm-start runs/hog26/stage2-self

The curriculum ("auto"): first it plays a bot that moves randomly, which
teaches the basics fast. It is promoted to the ladder once it actually beats
that bot -- an 80% win rate over its last 200 battles -- with --promote-at
decisions as the fallback cap. The ladder: about a third of its battles
against the current version of itself, a quarter against older saved
versions, and the rest against the gym's scripted bots (defend, push,
patient, random-legal). That is two bots that get better competing against
each other, which is where skill comes from. Checkpoints save on Ctrl+C and
on the --minutes deadline; run the same command again to carry on.
"""

import argparse
import contextlib
import json
import pathlib
import signal
import threading
import time

from royalelearn import Learner

from envs import build_env
from warmstart import _allow_env_drift, copy_latest_checkpoint, hard_foes_ladder

CUSTOM_DECK = "hog26"
DEFENSE = 0.5
DECISION_MS = 500
ADAPTIVE = False
WINCOND_CARD = "HogRider"
WINCOND_WEIGHT = 0.15
WINCOND_DEPLOY = 0.05
LEAK = 0.01
ELIXIR_ADV = 0.0


def build_env_hog26():
    return build_env("hog26", decision_ms=DECISION_MS, defense=DEFENSE,
                     adaptive=ADAPTIVE, wincond_card=WINCOND_CARD,
                     wincond_weight=WINCOND_WEIGHT,
                     wincond_deploy=WINCOND_DEPLOY, leak=LEAK,
                     elixir_adv=ELIXIR_ADV)


def build_env_random():
    return build_env("random", decision_ms=DECISION_MS, defense=DEFENSE,
                     adaptive=ADAPTIVE, wincond_card=WINCOND_CARD,
                     wincond_weight=WINCOND_WEIGHT,
                     wincond_deploy=WINCOND_DEPLOY, leak=LEAK,
                     elixir_adv=ELIXIR_ADV)


def build_env_custom():
    return build_env(CUSTOM_DECK.split(","), decision_ms=DECISION_MS,
                     defense=DEFENSE, adaptive=ADAPTIVE,
                     wincond_card=WINCOND_CARD, wincond_weight=WINCOND_WEIGHT,
                     wincond_deploy=WINCOND_DEPLOY, leak=LEAK,
                     elixir_adv=ELIXIR_ADV)


def build_learner(args, save_dir, opponent, timestep_limit):
    fn = {"hog26": build_env_hog26, "random": build_env_random}.get(args.deck, build_env_custom)
    return Learner(
        fn,
        n_envs=args.n_envs,
        device=args.device,
        opponent=opponent,
        trunk_channels=args.trunk_channels,
        trunk_blocks=args.trunk_blocks,
        steps_per_update=4096,
        ppo_batch_size=4096,
        ppo_minibatch_size=1024,
        ppo_epochs=args.ppo_epochs,
        policy_lr=args.lr,
        critic_lr=args.lr,
        ppo_ent_coef=args.ent_coef,
        checkpoint_every=args.checkpoint_every,
        timestep_limit=timestep_limit,
        save_dir=str(save_dir),
        resume=True,
        seed=args.seed,
    )


_nullcontext = contextlib.nullcontext


def recent_scores(run_dir, foe, window=200):
    """The learner's scores in its last ladder battles against one foe.

    Every training battle is appended to the run's ladder/games.jsonl with a
    score of 1 win, 0.5 draw, 0 loss -- including stage 1's battles against
    the random bot, which is what makes promotion by skill observable."""
    games = pathlib.Path(run_dir) / "ladder" / "games.jsonl"
    if not games.exists():
        return []
    scores = []
    try:
        with open(games, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("a") == "learner" and row.get("b") == foe:
                    scores.append(float(row.get("score_a", 0.0)))
    except OSError:
        return []
    return scores[-window:]


def make_promotion_watch(run_dir, threshold, min_games, window=200, poll=20.0):
    """A watcher that stops stage 1 early, once the bot has actually beaten
    the random bot. Promotion by decision count alone can move a bot that
    still loses to randomness into self-play, where two weak players happily
    reinforce each other's mistakes; this promotes on the win rate over the
    last ``window`` battles instead, once at least ``min_games`` of them
    exist. The --promote-at decision cap still applies as the fallback."""

    def watch(running):
        while running.is_set():
            time.sleep(poll)
            scores = recent_scores(run_dir, "scripted:random_legal", window)
            if len(scores) < min_games:
                continue
            rate = sum(scores) / len(scores)
            if rate >= threshold:
                print(f"promoting: {rate:.0%} over the last {len(scores)} "
                      f"battles vs the random bot (target {threshold:.0%})")
                signal.raise_signal(signal.SIGINT)
                return

    return watch


def run_until(learner, seconds, watch=None):
    if seconds <= 0:
        return
    running = threading.Event()
    running.set()

    def stop_when_time_is_up():
        time.sleep(seconds)
        if running.is_set():
            signal.raise_signal(signal.SIGINT)

    threads = [threading.Thread(target=stop_when_time_is_up, daemon=True)]
    if watch is not None:
        threads.append(threading.Thread(target=watch, args=(running,), daemon=True))
    for t in threads:
        t.start()
    try:
        learner.learn()
    finally:
        running.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", default="hog26", help="'hog26', 'random', or 8 comma-separated names")
    p.add_argument("--opponent", choices=["auto", "random", "self", "noop"], default="auto")
    p.add_argument("--warm-start", default=None, help="run dir whose weights to begin from")
    p.add_argument("--save-dir", default=None, help="default: runs/<deck>")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--promote-at", type=int, default=150_000,
                   help="decision cap for stage 1; promotion now happens on skill, this is only the fallback")
    p.add_argument("--promote-win-rate", type=float, default=0.8,
                   help="stage 1 promotes early once the bot wins this share of its last 200 battles vs the random bot")
    p.add_argument("--promote-games", type=int, default=100,
                   help="battles vs the random bot required before the win-rate promotion can fire")
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--device", default="cpu",
                   help="'auto' uses your GPU if torch can see it, else the CPU")
    p.add_argument("--defense", type=float, default=0.5,
                   help="starting weight of the tower-damage-taken penalty; 0 disables, 1-2 defends harder")
    p.add_argument("--adaptive-defense", action="store_true",
                   help="let the defense weight tune itself: up while the bot takes more tower damage than it deals, down while it defends fine")
    p.add_argument("--wincond-card", default="HogRider",
                   help="card that earns a deploy bonus; the fix for a bot that cycles cheap cards and never plays its win condition")
    p.add_argument("--wincond-bonus", type=float, default=0.15,
                   help="reward when the win-condition card connects (alive while an enemy tower loses hp); 0 disables", )
    p.add_argument("--wincond-deploy-bonus", type=float, default=0.05,
                   help="smaller reward for merely deploying the win-condition card, the nudge that gets it out of hand at all")
    p.add_argument("--leak-weight", type=float, default=0.01,
                   help="per-step penalty for sitting at 10 elixir while the enemy has units on the board; holding at 10 against an empty board is free")
    p.add_argument("--elixir-advantage", type=float, default=0.0,
                   help="per-decision reward for holding more elixir than the enemy; pays for winning trades and banking counter-pushes, keep it tiny (0.0005)")
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="PPO passes over each batch of experience; 2 learns more per decision, 1 updates faster on a CPU-only machine")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="entropy bonus; higher explores more, lower commits harder")
    p.add_argument("--decision-ms", type=int, default=500,
                   help="game-time per decision; 250 doubles decisions per battle (finer control, slower wall-clock), 1000 halves them")
    p.add_argument("--ladder", choices=["defense", "default"], default="defense",
                   help="'defense' aims the scripted share at the playstyles that beat the bot")
    p.add_argument("--trunk-channels", type=int, default=32)
    p.add_argument("--trunk-blocks", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    global CUSTOM_DECK, DEFENSE, ADAPTIVE, WINCOND_CARD, WINCOND_WEIGHT, LEAK
    global DECISION_MS, ELIXIR_ADV, WINCOND_DEPLOY
    CUSTOM_DECK = args.deck
    DEFENSE = args.defense
    ADAPTIVE = args.adaptive_defense
    WINCOND_CARD = args.wincond_card
    WINCOND_WEIGHT = args.wincond_bonus
    WINCOND_DEPLOY = args.wincond_deploy_bonus
    LEAK = args.leak_weight
    DECISION_MS = args.decision_ms
    ELIXIR_ADV = args.elixir_advantage
    ladder_ctx = hard_foes_ladder if args.ladder == "defense" else _nullcontext

    deck_tag = args.deck if args.deck in ("hog26", "random") else "custom"
    base = pathlib.Path(args.save_dir or f"runs/{deck_tag}")
    stage1_dir = base / "stage1-random" if args.opponent == "auto" else base
    stage2_dir = base / "stage2-self" if args.opponent == "auto" else base
    deadline = time.time() + args.minutes * 60

    if args.warm_start:
        copy_latest_checkpoint(args.warm_start, stage1_dir)

    if args.opponent == "auto":
        print(f"stage 1: vs the random bot, until it wins {args.promote_win_rate:.0%} "
              f"of 200 battles (cap {args.promote_at} decisions, "
              f"{args.minutes / 2:.0f} minutes) -> {stage1_dir}")
        watch = make_promotion_watch(stage1_dir, args.promote_win_rate,
                                     args.promote_games)
        with _allow_env_drift(), ladder_ctx():
            learner = build_learner(args, stage1_dir, "random", args.promote_at)
            run_until(learner, deadline - time.time(), watch)

        scores = recent_scores(stage1_dir, "scripted:random_legal")
        if scores:
            rate = sum(scores) / len(scores)
            if rate < args.promote_win_rate:
                print(f"note: stage 1 ended at {rate:.0%} vs the random bot "
                      f"(target {args.promote_win_rate:.0%}); the ladder may be "
                      f"weak until the bot catches up")

        if time.time() >= deadline:
            print("the --minutes deadline passed during stage 1; nothing to save")
            return
        print(f"stage 2: the ladder -- vs itself, its older versions and scripted bots "
              f"-> {stage2_dir}")
        with _allow_env_drift(), ladder_ctx():
            learner = build_learner(args, stage2_dir, "self", 1_000_000_000)
            copy_latest_checkpoint(stage1_dir, stage2_dir)
            run_until(learner, deadline - time.time())
            learner.save(str(stage2_dir / "bot"))
    else:
        print(f"training vs {args.opponent} -> {stage1_dir}")
        with _allow_env_drift(), ladder_ctx():
            learner = build_learner(args, stage1_dir, args.opponent, 1_000_000_000)
            run_until(learner, deadline - time.time())
            if time.time() < deadline:
                learner.save(str(stage1_dir / "bot"))


if __name__ == "__main__":
    main()
