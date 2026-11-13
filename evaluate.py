"""Measure a trained bot: win rate against a ladder of baselines, with error bars.

    python evaluate.py --run runs/hog26/stage2-self --deck hog26
    python evaluate.py --run runs/hog26/stage2-self --opponent push
    python evaluate.py --run runs/anydeck --deck random --games 100

Opponents, easiest to hardest: noop, random, first-affordable, defend, push,
patient, self (the same bot: 50% is all two equals can do). --opponent all
plays the whole ladder in one go.
"""

import argparse
import pathlib

import numpy as np
from royalegym import (
    DefendOpponent,
    FirstAffordableOpponent,
    NoopOpponent,
    PatientOpponent,
    PushOpponent,
    RandomLegalOpponent,
    evaluate,
)
from royalelearn import Learner

from envs import build_env


class PolicyOpponent:
    """Adapt a saved bot (``bot(obs) -> action``) to the ``.act(obs, mask, rng)``
    interface ``evaluate`` plays against."""

    def __init__(self, bot):
        self.bot = bot

    def act(self, obs, mask, rng):
        return int(self.bot(obs)) if mask.any() else 0


class PlayEverything:
    """Dump whatever it can afford, wherever it is legal, as fast as possible."""

    def act(self, obs, mask, rng):
        legal = np.flatnonzero(mask)
        legal = legal[legal != 0]
        return int(rng.choice(legal)) if legal.size else 0


def make_foes():
    return {
        "noop": ("noop", NoopOpponent()),
        "random": ("random-legal", RandomLegalOpponent(0.5)),
        "greedy": ("greedy", PlayEverything()),
        "first-affordable": ("first-affordable", FirstAffordableOpponent()),
        "defend": ("defend", DefendOpponent()),
        "push": ("push", PushOpponent()),
        "patient": ("patient", PatientOpponent()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/hog26/stage2-self")
    p.add_argument("--deck", default="hog26", help="'hog26', 'random', or 8 comma-separated names")
    p.add_argument("--opponent", default="random",
                   choices=[*make_foes(), "self", "all"])
    p.add_argument("--games", type=int, default=60)
    args = p.parse_args()

    deck = args.deck
    if deck not in ("hog26", "random"):
        deck = deck.split(",")
        if len(deck) != 8:
            raise SystemExit("give 8 comma-separated card names, or 'hog26'/'random'")
    build = lambda: build_env(deck)

    foes = []
    if args.opponent in ("all", "self"):
        foes.append(("itself", PolicyOpponent(Learner.load_policy(args.run))))
    if args.opponent == "all":
        foes.extend(make_foes().values())
    elif args.opponent != "self":
        foes.append(make_foes()[args.opponent])

    for foe_name, foe in foes:
        bot = PolicyOpponent(Learner.load_policy(args.run))
        result = evaluate(bot, foe, build, games=args.games, seed=0,
                          names=("bot", foe_name))
        print(f"== vs {foe_name} ==")
        print(result.summary())
        print()


if __name__ == "__main__":
    main()
