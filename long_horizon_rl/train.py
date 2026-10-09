"""Entrypoint that registers local extensions inside the upstream Ray runner."""

import ray
from verl.trainer import main_ppo

_Runner = main_ppo.TaskRunnerV1.__ray_metadata__.modified_class


@ray.remote
class TaskRunner(_Runner):
    def run(self, config):
        # Upstream isolates its own loggers; Ray/root filters can hide adapter
        # INFO messages. Configure only our namespace, not process-wide logging.
        import logging
        import os

        project_logger = logging.getLogger("long_horizon_rl")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))
        project_logger.handlers = [handler]
        project_logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
        project_logger.propagate = False
        from pathlib import Path
        from long_horizon_rl.run_manifest import write_manifest

        manifest_path = write_manifest(config, Path(__file__).resolve().parent.parent)
        project_logger.info("RESOLVED_RUN_MANIFEST %s", manifest_path)
        from long_horizon_rl.async_profile import validate_async_profile
        from long_horizon_rl.trainer_plugins import load_trainer_plugin

        plugin_async = load_trainer_plugin(config.trainer.v1.trainer_mode)
        validate_async_profile(config, async_mode=plugin_async)
        if config.actor_rollout_ref.rollout.name == "vllm":
            from long_horizon_rl.adapters.vllm_cancel_v1 import install

            install()
        from long_horizon_rl.adapters import trainer_v1

        if config.trainer.v1.trainer_mode == "long_horizon_opd":
            from long_horizon_rl.adapters import opd_v1
        from verl.trainer.ppo.v1.trainer_base import _tq_supports_checkpoint

        if (
            plugin_async
            or config.trainer.v1.trainer_mode
            in ("long_horizon_separate_async", "long_horizon_opd", "separate_async")
        ) and not _tq_supports_checkpoint():
            raise RuntimeError(
                "Async long-horizon training requires TransferQueue checkpoint support; run scripts/bootstrap.sh for the pinned compatible version"
            )
        return super().run(config)


def main():
    main_ppo.TaskRunnerV1 = TaskRunner
    main_ppo.main()


if __name__ == "__main__":
    main()
