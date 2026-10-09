#!/bin/bash -l
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${LONG_HORIZON_TRAIN_FILE:?Set the training parquet path}"
: "${LONG_HORIZON_VAL_FILE:?Set the validation parquet path}"
export LONG_HORIZON_TRAINER_MODULE=long_horizon_rl.train
exec bash "$project_root/scripts/launch_train.sh" \
 trainer.v1.trainer_mode=long_horizon_separate_async \
 trainer.v1.separate_async.num_warmup_batches=1 \
 trainer.v1.sampler.max_off_policy_threshold=4 \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.use_fused_kernels=True \
 actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
 data.train_files="$LONG_HORIZON_TRAIN_FILE" \
 data.val_files="$LONG_HORIZON_VAL_FILE" \
 data.train_batch_size=64 data.max_prompt_length=8192 data.max_response_length=131072 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_epochs=1 \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.optim.lr=2e-6 \
 actor_rollout_ref.actor.clip_ratio=0.004 \
 actor_rollout_ref.actor.clip_ratio_low=0.004 \
 actor_rollout_ref.actor.clip_ratio_high=0.004 \
 actor_rollout_ref.actor.use_kl_loss=False algorithm.use_kl_in_reward=False \
 actor_rollout_ref.actor.loss_agg_mode=token-sum \
 actor_rollout_ref.rollout.n=16 \
 actor_rollout_ref.rollout.max_model_len=131072 \
 actor_rollout_ref.rollout.agent.agent_loop_config_path="$project_root/configs/agent_long_horizon.yaml" \
 algorithm.rollout_correction.rollout_is=token \
 algorithm.rollout_correction.rollout_is_threshold=2.0 \
 trainer.total_training_steps=100 trainer.total_epochs=100 trainer.save_freq=100 \
 trainer.max_actor_ckpt_to_keep=2 \
 trainer.checkpoint_callback_class=long_horizon_rl.adapters.checkpoint_v1.PersistentRetentionCallback \
 trainer.default_local_dir="$project_root/checkpoints/long-horizon" "$@"
