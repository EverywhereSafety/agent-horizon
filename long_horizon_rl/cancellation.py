"""Track a pinned token client's actual server request for targeted cancellation."""

import asyncio
from contextvars import ContextVar


def cancel_aware_client(client, timeout=10, on_abort=None):
    if getattr(client, "_long_horizon_cancel_aware", False):
        return client
    request_state = ContextVar("long_horizon_request_state", default=None)
    base = type(client)

    class CancelAwareClient(base):
        _long_horizon_cancel_aware = True

        async def _acquire_server(self, request_id, **kwargs):
            result = await super()._acquire_server(request_id, **kwargs)
            state = request_state.get()
            if state is not None:
                state["server"] = result[1]
            return result

        def _vllm_request_id(self, request_id):
            actual = super()._vllm_request_id(request_id)
            state = request_state.get()
            if state is not None:
                state["request_id"] = actual
            return actual

        async def generate(self, *args, **kwargs):
            state = {}
            token = request_state.set(state)
            try:
                return await super().generate(*args, **kwargs)
            except asyncio.CancelledError:
                if "server" in state and "request_id" in state:
                    result = await asyncio.wait_for(
                        asyncio.shield(
                            state["server"].abort_request.remote(
                                state["request_id"], reset_prefix_cache=False
                            )
                        ),
                        timeout,
                    )
                    # Already-finished requests make cancellation idempotent.
                    if not result.get("aborted") and "not found" not in result.get(
                        "error", ""
                    ):
                        raise RuntimeError(
                            f"Generation abort was not acknowledged: {result}"
                        )
                    if on_abort is not None:
                        receipt = on_abort(
                            {
                                "request_id": state["request_id"],
                                "acknowledgement": result,
                            }
                        )
                        if asyncio.iscoroutine(receipt):
                            await receipt
                raise
            finally:
                request_state.reset(token)

    wrapped = object.__new__(CancelAwareClient)
    wrapped.__dict__.update(client.__dict__)
    return wrapped
