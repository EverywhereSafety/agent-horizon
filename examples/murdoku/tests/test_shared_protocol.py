from copy import deepcopy
from murdoku_lab.environment.protocol import simplify_record
from murdoku_lab.environment.replay import replay_candidate
from murdoku_demo.teacher_protocol import prepare_teacher_record
from murdoku_demo import murdoku_protocol, teacher_replay


def test_demo_uses_canonical_protocol_and_replay():
    assert murdoku_protocol.simplify_record is simplify_record
    assert teacher_replay.replay_candidate is replay_candidate
    query = {
        "messages": [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "puzzle"},
        ],
        "murdoku_observation": "vision",
    }
    before = deepcopy(query)
    teacher = prepare_teacher_record(query, max_turns=20)
    expected = simplify_record(query, workspace=True, max_turns=20)
    assert teacher["messages"] == expected["messages"]
    assert teacher["tool_schemas"] == expected["tool_schemas"]
    assert teacher["agent_info"]["max_turn"] == 20
    assert teacher["murdoku_reward"] == {"mode": "strict"}
    assert query == before
