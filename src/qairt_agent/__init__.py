"""Agent-native orchestration for QAIRT 2.49.

Two public layers:

* The asynchronous native workflow (``QairtAgentClient`` / ``JobHandle``)
  backed by a persistent file journal, family presets, and ``WorkflowSpec``.
* The synchronous stage engine (``QairtAgent``) and the ``BuildSpec`` contract
  it consumes; a build spec is converted to a workflow spec with
  :func:`qairt_agent.contracts.to_workflow_spec`.
"""

from qairt_agent.agent import JobHandle, QairtAgentClient
from qairt_agent.contracts import BuildSpec, RunManifest, ToolResult, WorkflowSpec
from qairt_agent.pipeline import QairtAgent

__all__ = [
    "BuildSpec",
    "JobHandle",
    "QairtAgent",
    "QairtAgentClient",
    "RunManifest",
    "ToolResult",
    "WorkflowSpec",
]
__version__ = "0.1.0"
