"""Measure how fast this machine trains Clash bots, to compare machines.

    python benchmark.py

Prints raw engine speed and end-to-end training throughput (env + neural net).
Compare the two `steps/s` numbers against another machine's.
"""

import shutil
import time

import numpy as np
from royalegym import ClashParallelEnv, DefaultStateMutator, GameOverCondition, RustEngine

from envs import build_env


def build_env_hog26():
    return build_env("hog26")


def raw_engine_speed():
    env = ClashParallelEnv(
        RustEngine(),
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=500,
    )
    obs, infos = env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps, battles, t0 = 0, 0, time.time()
    while battles < 5 and time.time() - t0 < 20:
        actions = {}
        for a in env.agents:
            m = env.action_masks(a)
            legal = np.flatnonzero(m)
            actions[a] = int(rng.choice(legal)) if legal.size else 0
        obs, _, term, trunc, infos = env.step(actions)
        steps += 1
        if term["blue"] or trunc["blue"]:
            battles += 1
            obs, infos = env.reset()
    dt = time.time() - t0
    print(f"raw engine (random play): {steps/dt:8.0f} decisions/s, {dt/battles:.2f} s/battle")
    env.close()


def training_speed():
    from royalelearn import Learner

    shutil.rmtree("/tmp/bench", ignore_errors=True)
    learner = Learner(
        build_env_hog26,
        n_envs=5,
        device="cpu",
        opponent="random",
        trunk_channels=32,
        trunk_blocks=2,
        steps_per_update=2048,
        ppo_batch_size=2048,
        ppo_minibatch_size=1024,
        ppo_epochs=1,
        timestep_limit=8192,
        save_dir="/tmp/bench",
        resume=False,
        seed=0,
    )
    t0 = time.time()
    learner.learn()
    dt = time.time() - t0
    print(f"training loop (net included): {8192/dt:8.0f} decisions/s over one update")


if __name__ == "__main__":
    raw_engine_speed()
    training_speed()
    print("reference: this project's dev sandbox did ~130 decisions/s end to end")
