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
    ElixirTradeReward,
    EntityKind,
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


class PatientLeakPenalty(RewardFunction):
    """Sitting at full elixir is only penalized while the enemy has units on
    the board.

    The stock ElixirLeakPenalty fires unconditionally, which taxes three
    tactics good players use on purpose: waiting at 10 to force the first
    move, banking elixir after a won defense to build a counter-push, and
    holding so a 4-cost win condition becomes castable. It priced holding at
    ten as worse than losing a tower, and the policy responded by spending
    every drop the moment it arrived. This version fires only when the enemy
    has troops or buildings alive and this seat is floating -- that is when
    leaked regeneration actually concedes board presence. A mutual standoff
    at 10 costs nothing, so forcing the other player to move first is free.
    """

    UNIT_KINDS = (EntityKind.TROOP, EntityKind.BUILDING)

    def __init__(self, weight=0.01, max_milli=10_000):
        self.weight = float(weight)
        self.max_milli = int(max_milli)

    def get_reward(self, team, prev, state, results):
        if self.weight <= 0 or prev is None:
            return 0.0
        full = self.max_milli
        mine = (prev.players[team].elixir_milli >= full
                and state.players[team].elixir_milli >= full)
        if not mine:
            return 0.0
        foe = 1 - team
        pressure = any(e.kind in self.UNIT_KINDS
                       for e in state.entities if e.team == foe)
        return -self.weight if pressure else 0.0

    def config(self):
        return {"name": "PatientLeakPenalty", "weight": self.weight,
                "max_milli": self.max_milli}


class DeployCardBonus(RewardFunction):
    """A small plus every time this seat deploys a named card.

    PPO found a cheap-cycle local optimum: spend elixir the moment it reaches
    1-3 and a 4-cost win condition never becomes castable, so the Hog never
    gets played and the bot cannot win. This pays it for deploying the card
    anyway, which gives the gradient a way out: play the Hog, see what it
    earns, and keep doing it when it works.
    """

    def __init__(self, card="HogRider", weight=0.15):
        self.card = card
        self.weight = float(weight)
        self.card_id = None

    def resolve(self, engine):
        cards = engine.cards() if callable(engine.cards) else engine.cards
        for c in cards:
            if c.name == self.card:
                self.card_id = c.card_id
                return True
        return False

    def get_reward(self, team, prev, state, results):
        if self.card_id is None or prev is None:
            return 0.0
        prev_uids = {e.uid for e in prev.entities}
        n = sum(1 for e in state.entities
                if e.team == team and e.card_id == self.card_id
                and e.uid not in prev_uids)
        return self.weight * n

    def config(self):
        return {"name": "DeployCardBonus", "card": self.card,
                "weight": self.weight}


class ElixirAdvantageReward(RewardFunction):
    """A small plus per decision for holding more elixir than the enemy.

    Efficient play wins trades: stopping a 5-cost push with a 1-cost card
    widens the elixir gap, and the gap is what funds a counter-push. This
    pays for the standing gap itself, so a positive trade pays twice -- once
    in ElixirTradeReward for the units killed, once here for the advantage
    it left behind. Kept tiny on purpose: summed over a whole battle it can
    rival a crown, and it must never pay for hoarding elixir instead of
    using it -- PatientLeakPenalty guards the other side of that line."""

    def __init__(self, weight=0.0005, max_milli=10_000):
        self.weight = float(weight)
        self.max_milli = int(max_milli)

    def get_reward(self, team, prev, state, results):
        if self.weight <= 0 or prev is None:
            return 0.0
        mine = state.players[team].elixir_milli
        foe = state.players[1 - team].elixir_milli
        return self.weight * (mine - foe) / self.max_milli

    def config(self):
        return {"name": "ElixirAdvantageReward", "weight": self.weight}


def make_reward(defense=0.5, adaptive=False, wincond=None, leak=0.01,
                elixir_adv=0.0):
    """Win is what counts. Crowns, tower damage, elixir trades and (with
    weight ``defense``) tower damage taken are hints that speed learning.
    With ``adaptive`` the defense weight tunes itself while training. With
    ``elixir_adv`` > 0 the bot is also paid a little for holding more
    elixir than the enemy."""
    terms = [
        (WinLossReward(), 1.0),
        (CrownReward(), 0.2),
        (TowerHPReward(), 0.1),
        (ElixirTradeReward(), 0.2),
        (PatientLeakPenalty(weight=leak), 1.0),
    ]
    if defense:
        if adaptive:
            terms.append((AdaptiveDefensePenalty(weight=defense), 1.0))
        else:
            terms.append((TowerDamageTakenPenalty(), float(defense)))
    if wincond:
        terms.append((wincond, 1.0))
    if elixir_adv:
        terms.append((ElixirAdvantageReward(weight=elixir_adv), 1.0))
    return CombinedReward(terms)


def _env(state_mutator, decision_ms, defense, adaptive=False, wincond=None,
         leak=0.01, elixir_adv=0.0):
    engine = RustEngine()
    if wincond is not None and not wincond.resolve(engine):
        wincond = None
    return ClashParallelEnv(
        engine,
        state_mutator=state_mutator,
        obs_builder=SpatialObsBuilder(),
        action_parser=TileActionParser(),
        reward_fn=make_reward(defense, adaptive, wincond, leak, elixir_adv),
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=decision_ms,
    )


def build_env(deck="hog26", decision_ms=500, defense=0.5, adaptive=False,
              wincond_card="HogRider", wincond_weight=0.15, leak=0.01,
              elixir_adv=0.0):
    """deck: 'hog26' (mirror), 'random' (eight random cards a battle), or a
    list of eight card names for a mirror of that deck.

    defense: weight of the tower-damage-taken penalty. 0 turns it off; raise
    it (1.0, 2.0) for a bot that defends sooner in its training. With
    adaptive=True that weight moves by itself: up while the bot takes more
    tower damage than it deals, down while it defends fine.

    wincond: a small bonus (wincond_weight) each time the bot deploys
    wincond_card. Without it PPO can settle into spamming cheap cards and
    never saving the elixir to play its win condition at all. A card that is
    not in the deck disables the term, so random decks are unaffected."""
    wincond = (DeployCardBonus(wincond_card, wincond_weight)
               if wincond_card and wincond_weight else None)
    if deck == "hog26":
        return _env(DefaultStateMutator(decks=[HOG_26, HOG_26]), decision_ms,
                    defense, adaptive, wincond, leak, elixir_adv)
    if deck == "random":
        return _env(DefaultStateMutator(), decision_ms, defense, adaptive,
                    wincond, leak, elixir_adv)
    return _env(DefaultStateMutator(decks=[deck, deck]), decision_ms,
                defense, adaptive, wincond, leak, elixir_adv)
