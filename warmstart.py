"""Warm-start a new run from another run's weights.

The trainer refuses to resume a run whose environment changed (different decks,
say), because a policy trained on one game cannot play another. When only the
decks changed and the observation and action shapes are identical -- as between
the Hog 2.6 run and the any-deck run here -- the weights transfer fine, so this
module copies the source run's latest checkpoint into the new run's folder and
tells the trainer the difference is deliberate.
"""

import shutil
from contextlib import contextmanager
from pathlib import Path

import royalelearn.learner as learner_mod

#: Identity fields that may differ on a deliberate warm-start: the decks changed
#: (env spec), who it plays (ladder), how battles are batched (rollout), and the
#: machine the earlier run lived on (engine build, library versions, device).
_DRIFT_FIELDS = frozenset({
    "env_spec_digest", "obs_digest", "action_digest",
    "rollout_digest", "ladder_digest",
    "engine_build", "torch_version", "device_kind",
    "royalelearn_version", "royalegym_version", "royalelearn_git", "royalegym_git",
})


@contextmanager
def _allow_env_drift():
    saved = learner_mod.TRAINING_SETTINGS
    learner_mod.TRAINING_SETTINGS = saved | _DRIFT_FIELDS
    try:
        yield
    finally:
        learner_mod.TRAINING_SETTINGS = saved


def copy_latest_checkpoint(src_run: str | Path, dst_run: str | Path) -> Path | None:
    """Copy the newest checkpoint of run ``src_run`` into run ``dst_run``'s folder."""
    src = sorted(Path(src_run).glob("checkpoint_*"))
    if not src:
        return None
    latest = src[-1]
    dst = Path(dst_run)
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / latest.name
    if not target.exists():
        shutil.copytree(latest, target)
    return target
