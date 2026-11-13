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


class AdaptiveDefensePenalty(RewardFunction):
    """TowerDamageTakenPenalty whose weight tunes itself while training.

    It watches a rolling ratio of tower damage this seat TOOK versus DEALT.
    Taking more than it deals means its defense is the weak side, so the
    penalty weight grows (capped); defending fine lets it decay toward the
    floor. The bot is thus paid more for defending exactly while defending
    badly, with no human watching. Bounded and slow on purpose: a reward that
    swings makes PPO's value estimates chase a moving target."""

    def __init__(self, weight=0.5, min_weight=0.1, max_weight=3.0,
                 target_ratio=1.0, adapt_rate=0.0005, ema=0.999):
        self.weight = float(weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.target_ratio = float(target_ratio)
        self.adapt_rate = float(adapt_rate)
        self.ema = float(ema)
        self._took = {}   # per seat: EMA of tower hp lost this step
        self._dealt = {}

    def _towers_lost(self, before, after):
        return sum(b - a for b, a in zip(before, after) if a < b)

    def get_reward(self, team, prev, state, results):
        took = self._towers_lost(prev.players[team].tower_hp,
                                 state.players[team].tower_hp)
        enemy = 1 - team
        dealt = self._towers_lost(prev.players[enemy].tower_hp,
                                  state.players[enemy].tower_hp)
        a = self.ema
        self._took[team] = took + a * self._took.get(team, took)
        self._dealt[team] = dealt + a * self._dealt.get(team, dealt)
        ratio = self._took[team] / max(self._dealt[team], 1.0)
        if ratio > self.target_ratio:
            self.weight = min(self.max_weight, self.weight * (1.0 + self.adapt_rate))
        else:
            self.weight = max(self.min_weight, self.weight * (1.0 - self.adapt_rate / 2))
        total = float(sum(prev.players[team].tower_hp)) or 1.0
        return -self.weight * took / total

    def config(self):
        return {"name": "AdaptiveDefensePenalty", "weight": round(self.weight, 3)}


def make_reward(defense=0.5, adaptive=False):
    """Win is what counts. Crowns, tower damage, elixir trades and (with
    weight ``defense``) tower damage taken are hints that speed learning.
    With ``adaptive`` the defense weight tunes itself while training."""
    terms = [
        (WinLossReward(), 1.0),
        (CrownReward(), 0.2),
        (TowerHPReward(), 0.1),
        (ElixirTradeReward(), 0.2),
        (ElixirLeakPenalty(), 0.01),
    ]
    if defense:
        if adaptive:
            terms.append((AdaptiveDefensePenalty(weight=defense), 1.0))
        else:
            terms.append((TowerDamageTakenPenalty(), float(defense)))
    return CombinedReward(terms)


def _env(state_mutator, decision_ms, defense, adaptive=False):
    return ClashParallelEnv(
        RustEngine(),
        state_mutator=state_mutator,
        obs_builder=SpatialObsBuilder(),
        action_parser=TileActionParser(),
        reward_fn=make_reward(defense, adaptive),
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=decision_ms,
    )


def build_env(deck="hog26", decision_ms=500, defense=0.5, adaptive=False):
    """deck: 'hog26' (mirror), 'random' (eight random cards a battle), or a
    list of eight card names for a mirror of that deck.

    defense: weight of the tower-damage-taken penalty. 0 turns it off; raise
    it (1.0, 2.0) for a bot that defends sooner in its training. With
    adaptive=True that weight moves by itself: up while the bot takes more
    tower damage than it deals, down while it defends fine."""
    if deck == "hog26":
        return _env(DefaultStateMutator(decks=[HOG_26, HOG_26]), decision_ms, defense, adaptive)
    if deck == "random":
        return _env(DefaultStateMutator(), decision_ms, defense, adaptive)
    return _env(DefaultStateMutator(decks=[deck, deck]), decision_ms, defense, adaptive)
