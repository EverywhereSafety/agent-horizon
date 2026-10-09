"""Bounded prompt admission for experiments that must never wrap a dataset."""


class SinglePassGuard:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive prompt limit required")
        self.limit = limit
        self.seen = set()

    def before_fetch(self, count):
        if len(self.seen) + count > self.limit:
            raise RuntimeError(
                "Single-pass training data exhausted; refusing to repeat queries"
            )

    def admit(self, uids):
        self.before_fetch(len(uids))
        if len(set(uids)) != len(uids) or self.seen.intersection(uids):
            raise RuntimeError(
                "Duplicate logical query rejected before rollout dispatch"
            )
        self.seen.update(uids)
