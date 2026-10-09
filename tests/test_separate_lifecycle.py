from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from long_horizon_rl.adapters.separate_lifecycle import DedicatedRolloutLifecycleMixin


class Native:
    def on_init_end(self):
        self.events.append("native_init")

    def on_validate_begin(self):
        self.events.append("native_validate")


class Trainer(DedicatedRolloutLifecycleMixin, Native):
    global_steps = 3

    def __init__(self, hybrid=False):
        self.events = []
        self.hybrid_rollout_config = SimpleNamespace(enable_switch=hybrid)
        self.checkpoint_manager = Mock()
        self.standalone_checkpoint_manager = SimpleNamespace(
            update_weights=lambda step: self.events.append(("standalone_sync", step))
        )

    def switch_to_trainer(self):
        self.events.append("remove_abort_sleep")


def test_dedicated_resume_never_wakes_colocated_weights_or_kv_cache():
    trainer = Trainer()
    trainer.on_init_end()
    trainer.on_validate_begin()
    assert trainer.events == ["remove_abort_sleep", ("standalone_sync", 3)]
    assert trainer.checkpoint_manager.mock_calls == []


def test_hybrid_keeps_native_lifecycle():
    trainer = Trainer(hybrid=True)
    trainer.on_init_end()
    trainer.on_validate_begin()
    assert trainer.events == ["native_init", "native_validate"]


def test_failed_sleep_prevents_sync_and_dispatch_readiness():
    trainer = Trainer()
    trainer.switch_to_trainer = Mock(side_effect=RuntimeError("sleep failed"))
    with pytest.raises(RuntimeError, match="sleep failed"):
        trainer.on_init_end()
    assert trainer.events == []
