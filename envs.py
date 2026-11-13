"""Environment factories.

A battle is a two-seat parallel env: seat "blue" is the bot being trained,
seat "red" is whoever the trainer puts there. Both seats see the same shapes,
so one policy can play either side.
"""

from royalegym import (
    ClashParallelEnv,
    CombinedReward,
    CrownReward,
    DefaultStateMutator,
    ElixirLeakPenalty,
    ElixirTradeReward,
    GameOverCondition,
    RustEngine,
    SpatialObsBuilder,
    TileActionParser,
    TowerHPReward,
    WinLossReward,
)

from decks import HOG_26


def make_reward():
    """Win is what counts. Crowns, tower damage and elixir trades are hints
    that speed learning; the elixir terms teach discipline early."""
    return CombinedReward([
        (WinLossReward(), 1.0),
        (CrownReward(), 0.2),
        (TowerHPReward(), 0.1),
        (ElixirTradeReward(), 0.2),
        (ElixirLeakPenalty(), 0.01),
    ])


def _env(state_mutator, decision_ms):
    return ClashParallelEnv(
        RustEngine(),
        state_mutator=state_mutator,
        obs_builder=SpatialObsBuilder(),
        action_parser=TileActionParser(),
        reward_fn=make_reward(),
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=decision_ms,
    )


def build_env(deck="hog26", decision_ms=500):
    """deck: 'hog26' (mirror), 'random' (eight random cards a battle), or a
    list of eight card names for a mirror of that deck."""
    if deck == "hog26":
        return _env(DefaultStateMutator(decks=[HOG_26, HOG_26]), decision_ms)
    if deck == "random":
        return _env(DefaultStateMutator(), decision_ms)
    return _env(DefaultStateMutator(decks=[deck, deck]), decision_ms)
