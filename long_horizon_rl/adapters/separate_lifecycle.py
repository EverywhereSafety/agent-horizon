"""Keep disabled hybrid replicas asleep during dedicated-rollout initialization."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DedicatedRolloutLifecycleMixin:
    def _init_dataloader(self):
        from omegaconf import OmegaConf
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        from verl.utils.import_utils import resolve_config_path
        from long_horizon_rl.diagnostics.agent_preflight import validate_dataset_agents

        super()._init_dataloader()
        agent = self.config.actor_rollout_ref.rollout.agent
        allowed = set(_agent_loop_registry)
        if agent.agent_loop_config_path:
            loops = OmegaConf.load(resolve_config_path(agent.agent_loop_config_path))
            allowed.update(loop.name for loop in loops)
        for dataset in (self.train_dataset, self.val_dataset):
            if not validate_dataset_agents(dataset, allowed, agent.default_agent_loop):
                logger.warning(
                    "Custom dataset has no static dataframe; agent-name preflight unavailable"
                )

    def on_init_end(self):
        if self.hybrid_rollout_config.enable_switch:
            return super().on_init_end()
        # Setup admits hybrid replicas even when switching is disabled. Remove
        # them before dispatch and never wake their KV cache after optimizer
        # restore. Reuse native routing/abort/sleep and standalone sync.
        self.switch_to_trainer()
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        logger.info(
            "Dedicated rollout initialized; hybrid replicas remain asleep step=%s",
            self.global_steps,
        )

    def on_validate_begin(self):
        if self.hybrid_rollout_config.enable_switch:
            return super().on_validate_begin()
        # The native separate-async client always targets standalone servers.
        # Validation therefore needs no colocated engine wake-up.
        return None
