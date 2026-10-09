"""Pinned verl NCCL compatibility: serialize strided weights in logical order."""


def install():
    from verl.checkpoint_engine import base, nccl_checkpoint_engine

    if getattr(base.split_weight_chunks, "_long_horizon_contiguous", False):
        return
    original = base.split_weight_chunks

    async def split_weight_chunks(weights, bucket_size, meta_only=False):
        async def contiguous_weights():
            async for name, weight in base.ensure_async_iterator(weights):
                # Preserve shape, dtype and values; allocate only for strided tensors.
                yield name, weight.contiguous()

        async for chunk in original(contiguous_weights(), bucket_size, meta_only):
            yield chunk

    split_weight_chunks._long_horizon_contiguous = True
    base.split_weight_chunks = split_weight_chunks
    nccl_checkpoint_engine.split_weight_chunks = split_weight_chunks
