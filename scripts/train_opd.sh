#!/bin/bash -l
# Launch within a four-GPU allocation: one trainer, two students, one teacher.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${LONG_HORIZON_MODEL_PATH:?Set the student model/checkpoint path}"
: "${LONG_HORIZON_TEACHER_PATH:?Set the downloaded teacher model path}"
: "${LONG_HORIZON_TRAIN_FILE:?Set the training parquet path}"
: "${LONG_HORIZON_VAL_FILE:?Set the held-out parquet path}"
export LONG_HORIZON_TRAINER_MODULE=long_horizon_rl.train
exec bash "$project_root/scripts/launch_train.sh" \
 trainer.v1.trainer_mode=long_horizon_opd \
 data.train_files="$LONG_HORIZON_TRAIN_FILE" data.val_files="$LONG_HORIZON_VAL_FILE" \
 data.train_batch_size=1 data.gen_batch_size=1 data.max_prompt_length=8192 data.max_response_length=65536 \
 data.apply_chat_template_kwargs.enable_thinking=True \
 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=True \
 actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
 actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.ppo_epochs=1 \
 actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.n_gpus_per_node=2 actor_rollout_ref.rollout.n=2 \
 actor_rollout_ref.rollout.max_model_len=65536 actor_rollout_ref.rollout.max_num_seqs=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.80 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True \
 actor_rollout_ref.rollout.agent.agent_loop_config_path="$project_root/configs/agent_opd_24k.yaml" \
 trainer.n_gpus_per_node=1 trainer.v1.separate_async.num_warmup_batches=0 \
 trainer.v1.sampler.max_off_policy_threshold=1 trainer.v1.sampler.max_off_policy_strategy=drop \
 distillation.enabled=True distillation.nnodes=1 distillation.n_gpus_per_node=1 \
 distillation.teacher_models.teacher_model.model_path="$LONG_HORIZON_TEACHER_PATH" \
 distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
 distillation.teacher_models.teacher_model.inference.max_model_len=73729 \
 distillation.teacher_models.teacher_model.inference.max_num_seqs=2 \
 distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.80 \
 distillation.teacher_models.teacher_model.inference.enforce_eager=True \
 +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.gdn_prefill_backend=triton \
 distillation.distillation_loss.loss_mode=k1 \
 distillation.distillation_loss.use_policy_gradient=True distillation.distillation_loss.use_task_rewards=False \
 distillation.distillation_loss.loss_max_clamp=10 distillation.distillation_loss.log_prob_min_clamp=-20 \
 trainer.total_training_steps=5 trainer.total_epochs=1 trainer.resume_mode=disable \
 actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
 trainer.max_actor_ckpt_to_keep=1 \
 trainer.checkpoint_callback_class=long_horizon_rl.adapters.checkpoint_v1.PersistentRetentionCallback \
 +trainer.checkpoint_min_free_bytes=68719476736 \
 trainer.default_local_dir="$project_root/checkpoints/opd" "$@"
