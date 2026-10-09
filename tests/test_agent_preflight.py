from types import SimpleNamespace
import pytest
from long_horizon_rl.diagnostics.agent_preflight import validate_dataset_agents


class Frame:
    column_names = ["agent_name"]

    def __init__(self, names):
        self.names = names

    def unique(self, name):
        assert name == "agent_name"
        return self.names


def dataset(names):
    return SimpleNamespace(dataframe=Frame(names))


def test_obsolete_fixture_fails_before_gpu_setup():
    with pytest.raises(ValueError, match="vhd_long_horizon"):
        validate_dataset_agents(
            dataset(["vhd_long_horizon"]), {"long_horizon"}, "long_horizon"
        )


def test_declared_custom_agent_and_native_names_are_accepted():
    assert validate_dataset_agents(
        dataset(["single_turn_agent", "custom"]),
        {"single_turn_agent", "custom"},
        "custom",
    )


def test_missing_agent_column_uses_validated_default():
    frame = SimpleNamespace(column_names=["prompt"])
    assert validate_dataset_agents(
        SimpleNamespace(dataframe=frame), {"custom"}, "custom"
    )
    with pytest.raises(ValueError, match="default agent"):
        validate_dataset_agents(SimpleNamespace(dataframe=frame), {"custom"}, "missing")


def test_nonstring_agent_is_configuration_error():
    with pytest.raises(ValueError, match="unregistered"):
        validate_dataset_agents(dataset([None]), {"custom"}, "custom")


def test_streaming_dataset_is_not_materialized_by_preflight():
    assert validate_dataset_agents(object(), {"custom"}, "custom") is False


def test_custom_frame_is_not_assumed_to_be_huggingface_dataset():
    assert (
        validate_dataset_agents(SimpleNamespace(dataframe=[]), {"custom"}, "custom")
        is False
    )
