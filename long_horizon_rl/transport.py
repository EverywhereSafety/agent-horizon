"""Token-native transport boundary, independent of a trainer framework."""

import asyncio
import json
import inspect
import urllib.request
from .contracts import Generation


class TokenHTTPBackend:
    """Connect to a normalized generation gateway; never retokenize model output.

    Gateway must return output_token_ids, output_logprobs, text and
    served_weight_version. This is not the OpenAI chat-completions protocol.
    Cancellation of the coroutine does not cancel an underlying HTTP request;
    a production adapter must provide explicit server-side cancellation.
    """

    def __init__(self, endpoint, timeout=120, request=None):
        self.endpoint = endpoint
        self.timeout = timeout
        self.request = request or self._request

    def _request(self, payload):
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    async def generate(self, ids, cap, dispatch_version, uid, turn):
        payload = {
            "request_id": f"{uid}/{turn}",
            "session_id": uid,
            "input_ids": list(ids),
            "sampling_params": {"max_new_tokens": cap},
            "return_logprob": True,
            "dispatch_version": dispatch_version,
        }
        if inspect.iscoroutinefunction(self.request):
            response = await self.request(payload)
        else:
            response = await asyncio.to_thread(self.request, payload)
        version = response["served_weight_version"]
        if type(version) is not int:
            raise ValueError("serving version must be an explicit integer")
        token_ids = response["output_token_ids"]
        if not isinstance(token_ids, list) or any(
            type(x) is not int or x < 0 for x in token_ids
        ):
            raise ValueError("output must contain exact nonnegative integer token IDs")
        if not isinstance(response["text"], str):
            raise ValueError("decoded sidecar must be a string")
        generation = Generation(
            token_ids, response["output_logprobs"], response["text"], version
        )
        generation.validate(cap)
        if version < dispatch_version:
            raise ValueError("server is older than dispatch policy")
        return generation
