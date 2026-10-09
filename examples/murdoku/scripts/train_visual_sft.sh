#!/usr/bin/env bash
set -euo pipefail
: "${VISUAL_SFT_MODEL:?Set full multimodal model checkpoint}"
: "${VISUAL_SFT_DATA:?Set prepared train.pt}"
: "${VISUAL_SFT_OUTPUT:?Set checkpoint output directory}"
demo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$demo_dir${PYTHONPATH:+:$PYTHONPATH}"
gpus="${VISUAL_SFT_GPUS:-4}"
batch="${VISUAL_SFT_BATCH:-$gpus}"
if (( gpus < 1 || batch < gpus || batch % gpus != 0 )); then
  echo 'SFT batch must be a positive multiple of GPU count' >&2
  exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
torchrun --standalone --nnodes=1 --nproc_per_node="$gpus" \
  -m verl.trainer.sft_trainer \
  model.path="$VISUAL_SFT_MODEL" \
  model.enable_gradient_checkpointing=True model.enable_activation_offload=False \
  model.use_remove_padding=True model.use_fused_kernels=True \
  model.fused_kernel_options.impl_backend="${VISUAL_SFT_FUSED_BACKEND:-liger}" \
  +model.override_config.attn_implementation=sdpa \
  engine.strategy=fsdp2 engine.fsdp_size="$gpus" \
  engine.model_dtype=bf16 engine.use_torch_compile=False \
  engine.ulysses_sequence_parallel_size=1 \
  data.train_files="$VISUAL_SFT_DATA" data.val_files=null \
  data.custom_cls.path="$demo_dir/murdoku_demo/visual_sft.py" \
  data.custom_cls.name=VisualTeacherDataset data.pad_mode=no_padding \
  data.max_length=131072 data.truncation=error \
  data.train_batch_size="$batch" \
  data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=False data.num_workers=0 \
  optim.lr=1e-5 optim.lr_warmup_steps_ratio=0.05 optim.weight_decay=0.01 \
  trainer.total_epochs=1 trainer.save_freq="${VISUAL_SFT_SAVE_FREQ:--1}" trainer.test_freq=-1 \
  trainer.resume_mode=auto trainer.max_ckpt_to_keep=1 \
  trainer.project_name=murdoku-visual-sft trainer.experiment_name=screenshot-pilot \
  trainer.logger='[console]' trainer.n_gpus_per_node="$gpus" \
  trainer.default_local_dir="$VISUAL_SFT_OUTPUT" "$@"
