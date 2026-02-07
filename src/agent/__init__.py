"""Agent components for DocOps.

This module provides the multi-step reasoning agent with:
- Tool definitions and execution
- System prompts for reasoning
- Core agent loop with context management
- Pre-defined workflows for common tasks
"""

from .tools import AgentTools, ToolResult
from .prompts import (
    get_system_prompt,
    get_tool_selection_prompt,
    get_analysis_prompt,
    get_report_synthesis_prompt,
)
from .agent_core import (
    DocOpsAgent,
    AgentResponse,
    AgentStep,
    StepType,
)
from .workflows import (
    WorkflowEngine,
    WorkflowResult,
)

__all__ = [
    # Tools
    "AgentTools",
    "ToolResult",
    # Prompts
    "get_system_prompt",
    "get_tool_selection_prompt",
    "get_analysis_prompt",
    "get_report_synthesis_prompt",
    # Agent
    "DocOpsAgent",
    "AgentResponse",
    "AgentStep",
    "StepType",
    # Workflows
    "WorkflowEngine",
    "WorkflowResult",
]
