"""Play one battle with a trained bot and save it to watch.

    python watch.py --run runs/hog26 --save my_bot_battle.msgpack
    royaleviser my_bot_battle.msgpack
"""

import argparse
import random

from royalegym import play_battle
from royalelearn import Learner

from envs import build_hog26_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/hog26")
    p.add_argument("--save", default="my_bot_battle.msgpack")
    p.add_argument("--greedy", action="store_true", help="always its most likely move")
    args = p.parse_args()

    bot = Learner.load_policy(args.run, greedy=args.greedy)
    battle = play_battle(
        build_hog26_env(),
        blue=bot,
        red="random",
        seed=random.randrange(1_000_000),
        save_to=args.save,
    )
    print(f"Crowns {battle.crowns[0]}-{battle.crowns[1]}, saved to {args.save}")


if __name__ == "__main__":
    main()
