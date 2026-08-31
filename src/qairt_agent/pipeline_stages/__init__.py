"""Stage bodies extracted from the ``QairtAgent`` facade.

Each module holds one stage's logic as plain functions taking the agent as
their first argument. The facade keeps its public methods and delegates, so the
CLI, job worker and MCP layers see no change at all; what changes is that a
stage is now a file you can read end to end.

Stage modules must never import ``qairt_agent.pipeline``: shared helpers live in
``qairt_agent.pipeline_support``, which both sides import.
"""
