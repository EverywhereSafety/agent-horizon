import pytest
from long_horizon_rl.single_pass import SinglePassGuard


def test_exactly_25_groups_of_four_without_dataset_wrap():
    guard = SinglePassGuard(100)
    for step in range(25):
        guard.admit([f"query-{4*step+i}" for i in range(4)])
    assert len(guard.seen) == 100
    with pytest.raises(RuntimeError, match="exhausted"):
        guard.before_fetch(4)


def test_duplicate_is_rejected_without_mutating_admission_state():
    guard = SinglePassGuard(100)
    guard.admit(["a", "b", "c", "d"])
    with pytest.raises(RuntimeError, match="Duplicate"):
        guard.admit(["e", "f", "a", "g"])
    assert guard.seen == {"a", "b", "c", "d"}
