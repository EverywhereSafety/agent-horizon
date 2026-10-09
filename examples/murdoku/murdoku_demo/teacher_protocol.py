"""Configure teacher collection using the same task prompt as other callers."""

from murdoku_lab.environment.protocol import simplify_record


def prepare_teacher_record(record, max_turns=20, python_timeout=120):
    result = simplify_record(
        record, workspace=True, python_timeout=python_timeout, max_turns=max_turns
    )
    result["murdoku_reward"] = {"mode": "strict"}
    return result
