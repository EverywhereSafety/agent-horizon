import asyncio
import pytest
import torch

pytest.importorskip(
    "verl", reason="native weight transfer requires the pinned veRL runtime"
)
from long_horizon_rl.adapters.weight_chunks_v1 import install


@pytest.mark.parametrize("bucket_size", [8, 1024])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_weight_chunk_roundtrip(layout, bucket_size):
    from verl.checkpoint_engine import base, nccl_checkpoint_engine

    install()
    splitter = base.split_weight_chunks
    install()
    assert base.split_weight_chunks is splitter
    assert nccl_checkpoint_engine.split_weight_chunks is splitter
    source = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8)
    weight = {"contiguous": source, "transpose": source.T, "slice": source[:, ::2]}[
        layout
    ]
    before = weight.clone()

    async def run():
        chunks = [
            item async for item in splitter(iter([("expert", weight)]), bucket_size)
        ]

        async def stream():
            for item in chunks:
                yield item

        restored = [
            item async for item in base.merge_weight_chunks(stream(), bucket_size)
        ]
        assert len(restored) == 1
        name, value = restored[0]
        assert name == "expert"
        assert value.dtype == weight.dtype and value.shape == weight.shape
        assert torch.equal(value, before)
        metadata = [
            item
            async for item in splitter(
                iter([("expert", weight)]), bucket_size, meta_only=True
            )
        ]
        assert all(chunk is None for _, chunk in metadata)
        assert [m for m, _ in metadata] == [m for m, _ in chunks]

    asyncio.run(run())
    assert torch.equal(weight, before)
