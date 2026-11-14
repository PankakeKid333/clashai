"""Train a Clash Royale bot, Hog 2.6 first, then any deck.

    python train.py --minutes 60                    # Hog 2.6, auto curriculum
    python train.py --deck random --minutes 60      # every deck, from scratch
    python train.py --deck random --warm-start runs/hog26/stage2-self

The curriculum ("auto"): first it plays a bot that moves randomly, which
teaches the basics fast. Once it has beaten that --promote-at decisions'
worth, it is promoted to the ladder: about half its battles against the
current version of itself, a third against older saved versions, and the
rest against the gym's scripted bots (defend, push, patient). That is two
bots that get better competing against each other, which is where skill
comes from. Checkpoints save on Ctrl+C and on the --minutes deadline; run
the same command again to carry on.
"""

import argparse
import contextlib
import pathlib
import signal
import threading
import time

from royalelearn import Learner

from envs import build_env
from warmstart import _allow_env_drift, copy_latest_checkpoint, hard_foes_ladder

CUSTOM_DECK = "hog26"
DEFENSE = 0.5
ADAPTIVE = False
WINCOND_CARD = "HogRider"
WINCOND_WEIGHT = 0.15


def build_env_hog26():
    return build_env("hog26", defense=DEFENSE, adaptive=ADAPTIVE,
                     wincond_card=WINCOND_CARD, wincond_weight=WINCOND_WEIGHT)


def build_env_random():
    return build_env("random", defense=DEFENSE, adaptive=ADAPTIVE,
                     wincond_card=WINCOND_CARD, wincond_weight=WINCOND_WEIGHT)


def build_env_custom():
    return build_env(CUSTOM_DECK.split(","), defense=DEFENSE, adaptive=ADAPTIVE,
                     wincond_card=WINCOND_CARD, wincond_weight=WINCOND_WEIGHT)


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
        ppo_epochs=1,
        policy_lr=3e-4,
        critic_lr=3e-4,
        ppo_ent_coef=0.01,
        checkpoint_every=args.checkpoint_every,
        timestep_limit=timestep_limit,
        save_dir=str(save_dir),
        resume=True,
        seed=args.seed,
    )


_nullcontext = contextlib.nullcontext


def run_until(learner, seconds):
    if seconds <= 0:
        return

    def stop_when_time_is_up():
        time.sleep(seconds)
        signal.raise_signal(signal.SIGINT)

    threading.Thread(target=stop_when_time_is_up, daemon=True).start()
    learner.learn()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", default="hog26", help="'hog26', 'random', or 8 comma-separated names")
    p.add_argument("--opponent", choices=["auto", "random", "self", "noop"], default="auto")
    p.add_argument("--warm-start", default=None, help="run dir whose weights to begin from")
    p.add_argument("--save-dir", default=None, help="default: runs/<deck>")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--promote-at", type=int, default=150_000,
                   help="decisions against the random bot before the ladder takes over")
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
                   help="reward per deploy of --wincond-card; 0 disables", )
    p.add_argument("--ladder", choices=["defense", "default"], default="defense",
                   help="'defense' aims the scripted share at the playstyles that beat the bot")
    p.add_argument("--trunk-channels", type=int, default=32)
    p.add_argument("--trunk-blocks", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    global CUSTOM_DECK, DEFENSE, ADAPTIVE, WINCOND_CARD, WINCOND_WEIGHT
    CUSTOM_DECK = args.deck
    DEFENSE = args.defense
    ADAPTIVE = args.adaptive_defense
    WINCOND_CARD = args.wincond_card
    WINCOND_WEIGHT = args.wincond_bonus
    ladder_ctx = hard_foes_ladder if args.ladder == "defense" else _nullcontext

    deck_tag = args.deck if args.deck in ("hog26", "random") else "custom"
    base = pathlib.Path(args.save_dir or f"runs/{deck_tag}")
    stage1_dir = base / "stage1-random" if args.opponent == "auto" else base
    stage2_dir = base / "stage2-self" if args.opponent == "auto" else base
    deadline = time.time() + args.minutes * 60

    if args.warm_start:
        copy_latest_checkpoint(args.warm_start, stage1_dir)

    if args.opponent == "auto":
        print(f"stage 1: vs the random bot, until {args.promote_at} decisions or "
              f"{args.minutes / 2:.0f} minutes -> {stage1_dir}")
        with _allow_env_drift(), ladder_ctx():
            learner = build_learner(args, stage1_dir, "random", args.promote_at)
            run_until(learner, deadline - time.time())

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
            learner.save(str(stage1_dir / "bot"))


if __name__ == "__main__":
    main()
