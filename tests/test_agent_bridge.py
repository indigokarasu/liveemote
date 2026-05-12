from hermes_avatar.protocol.agent_bridge import AgentBridge, normalize_agent_response


def test_agent_bridge_offline_capabilities():
    bridge = AgentBridge("none", harness="none")
    caps = bridge.capability_status()
    assert caps["available"] is False
    assert caps["mode"] == "none"


def test_normalize_agent_response_accepts_common_harness_shapes():
    assert normalize_agent_response({"text": "hi", "tags": {"affect": "calm"}}).text == "hi"
    deerflow = normalize_agent_response({"message": {"content": "hello"}, "emotion": "happy"})
    assert deerflow.text == "hello"
    assert deerflow.tags == {"value": "happy"}
    openclaw = normalize_agent_response({"output": {"text": "ok", "tags": {"voice": {"pace": 0.5}}}})
    assert openclaw.text == "ok"
    assert openclaw.tags["voice"]["pace"] == 0.5
