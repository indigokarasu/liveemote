from __future__ import annotations

from hermes_avatar.protocol.agent_bridge import AgentBridge, AgentResponse, affect_summary

HermesResponse = AgentResponse
HermesBridge = AgentBridge

__all__ = ["AgentBridge", "AgentResponse", "HermesBridge", "HermesResponse", "affect_summary"]
