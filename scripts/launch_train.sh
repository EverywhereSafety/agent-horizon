#!/bin/bash -l
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
# Shared four-GPU launch defaults; profiles and caller overrides follow them.
# CUDA modules are optional and deployment-specific.
cuda_module="${LONG_HORIZON_CUDA_MODULE-}"
if test -n "$cuda_module" && type module >/dev/null 2>&1; then
    module load "$cuda_module"
fi
python_executable="${LONG_HORIZON_PYTHON:-$project_root/.venv/bin/python}"
test -x "$python_executable"
export PATH="$(dirname "$python_executable"):$PATH"
export HF_HOME="${HF_HOME:-$project_root/../huggingface_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/../uv_cache}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
export LONG_HORIZON_AUDIT_DIR="${LONG_HORIZON_AUDIT_DIR:-$project_root/outputs/episodes}"
export RAY_TMPDIR="${TMPDIR:-/tmp}/long-horizon-ray-${SLURM_JOB_ID:-local}"
model_path="${LONG_HORIZON_MODEL_PATH:?Set a model/checkpoint path}"
model_overrides=()
if test -n "${LONG_HORIZON_ATTN_IMPLEMENTATION:-}"; then
    model_overrides+=("+actor_rollout_ref.model.override_config.attn_implementation=$LONG_HORIZON_ATTN_IMPLEMENTATION")
fi
if test -n "${LONG_HORIZON_VLLM_PREFILL_BACKEND:-}"; then
    model_overrides+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=$LONG_HORIZON_VLLM_PREFILL_BACKEND")
fi
cd "$project_root"
exec "$python_executable" -m "${LONG_HORIZON_TRAINER_MODULE:-long_horizon_rl.train}" \
 algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False \
 data.train_files="$project_root/outputs/data/train.parquet" \
 data.val_files="$project_root/outputs/data/val.parquet" \
 data.train_batch_size=2 data.max_prompt_length=6144 data.max_response_length=8192 \
 data.filter_overlong_prompts=True data.truncation=error data.shuffle=False \
 +data.apply_chat_template_kwargs.enable_thinking=False \
 actor_rollout_ref.model.path="$model_path" \
 actor_rollout_ref.model.use_remove_padding=False \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=2 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=False \
 actor_rollout_ref.actor.use_torch_compile=False \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
 actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.hybrid_engine=False \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.max_model_len=8192 \
 actor_rollout_ref.rollout.max_num_seqs=16 actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
 actor_rollout_ref.rollout.enforce_eager=True \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt='{image:0,video:0}' \
 actor_rollout_ref.rollout.calculate_log_probs=True \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 actor_rollout_ref.rollout.agent.default_agent_loop=long_horizon \
 actor_rollout_ref.rollout.agent.agent_loop_config_path="$project_root/configs/agent_smoke.yaml" \
 +actor_rollout_ref.rollout.agent.agent_loop_manager_class=long_horizon_rl.adapters.verl_v1.AgentLoopManager \
 trainer.v1.trainer_mode=separate_async \
 trainer.v1.separate_async.parameter_sync_step=1 \
 trainer.v1.sampler.max_off_policy_threshold=4 \
 trainer.v1.sampler.max_off_policy_strategy=drop \
 trainer.n_gpus_per_node=2 trainer.nnodes=1 \
 trainer.logger='[console]' trainer.val_before_train=False trainer.test_freq=-1 \
 trainer.save_freq=1 trainer.total_training_steps=2 trainer.total_epochs=2 \
 trainer.default_local_dir="$project_root/checkpoints/gpu-smoke" \
 trainer.balance_batch=False \
 ray_kwargs.ray_init.runtime_env.py_executable=null \
 ray_kwargs.ray_init.num_cpus=${LONG_HORIZON_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-24}} \
 ${model_overrides[@]+"${model_overrides[@]}"} "$@"
