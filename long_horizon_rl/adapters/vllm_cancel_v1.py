"""Adapt native request cancellation to vLLM's external/internal ID contract."""


def request_status(server, request_id):
    processor = server.engine.output_processor
    ids = list(processor.external_req_ids.get(request_id, ()))
    return {
        "active": any(key in processor.request_states for key in ids),
        "internal_ids": ids,
        "count": len(processor.request_states),
    }


async def abort_external_request(server, request_id, reset_prefix_cache=True):
    if server.node_rank != 0:
        return {"aborted": False, "request_id": request_id}
    before = request_status(server, request_id)
    if not before["active"]:
        return {"aborted": False, "error": f"Request {request_id} not found"}
    # Public engine API creates the abort output, removes external-ID mappings,
    # and forwards the resolved internal IDs to the engine core atomically.
    await server.engine.abort(request_id, internal=False)
    if reset_prefix_cache:
        await server.clear_kv_cache()
    return {
        "aborted": True,
        "request_id": request_id,
        "internal_request_ids": before["internal_ids"],
    }


def install():
    from verl.workers.rollout.vllm_rollout import vllm_async_server as native

    if getattr(native.vLLMHttpServer, "_long_horizon_external_abort", False):
        return

    class ExternalAbortServer(native.vLLMHttpServer):
        _long_horizon_external_abort = True

        async def abort_request(self, request_id, reset_prefix_cache=True):
            return await abort_external_request(self, request_id, reset_prefix_cache)

    native.vLLMHttpServer = ExternalAbortServer
