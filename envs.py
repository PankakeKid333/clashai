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
    RewardFunction,
    RustEngine,
    SpatialObsBuilder,
    TileActionParser,
    TowerHPReward,
    WinLossReward,
)

from decks import HOG_26


class TowerDamageTakenPenalty(RewardFunction):
    """Charges the seat for its own towers' lost hitpoints, scaled so losing
    every tower to the last hitpoint costs about -1. The symmetric
    TowerHPReward pays for damage dealt AND taken; this one makes the taken
    side hurt on its own, which is what teaches defending."""

    def get_reward(self, team, prev, state, results):
        before = prev.players[team].tower_hp
        after = state.players[team].tower_hp
        total = float(sum(before)) or 1.0
        lost = sum(b - a for b, a in zip(before, after) if a < b)
        return -lost / total

    def config(self):
        return {"name": "TowerDamageTakenPenalty"}


def make_reward(defense=0.5):
    """Win is what counts. Crowns, tower damage, elixir trades and (with
    weight ``defense``) tower damage taken are hints that speed learning."""
    terms = [
        (WinLossReward(), 1.0),
        (CrownReward(), 0.2),
        (TowerHPReward(), 0.1),
        (ElixirTradeReward(), 0.2),
        (ElixirLeakPenalty(), 0.01),
    ]
    if defense:
        terms.append((TowerDamageTakenPenalty(), float(defense)))
    return CombinedReward(terms)


def _env(state_mutator, decision_ms, defense):
    return ClashParallelEnv(
        RustEngine(),
        state_mutator=state_mutator,
        obs_builder=SpatialObsBuilder(),
        action_parser=TileActionParser(),
        reward_fn=make_reward(defense),
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=decision_ms,
    )


def build_env(deck="hog26", decision_ms=500, defense=0.5):
    """deck: 'hog26' (mirror), 'random' (eight random cards a battle), or a
    list of eight card names for a mirror of that deck.

    defense: weight of the tower-damage-taken penalty. 0 turns it off; raise
    it (1.0, 2.0) for a bot that defends sooner in its training."""
    if deck == "hog26":
        return _env(DefaultStateMutator(decks=[HOG_26, HOG_26]), decision_ms, defense)
    if deck == "random":
        return _env(DefaultStateMutator(), decision_ms, defense)
    return _env(DefaultStateMutator(decks=[deck, deck]), decision_ms, defense)
