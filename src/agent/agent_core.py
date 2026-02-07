"""Core agent implementation for DocOps.

Implements multi-step reasoning with tool orchestration,
following Elasticsearch Agent Builder patterns.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from .tools import AgentTools, ToolResult
from .prompts import get_system_prompt


class StepType(str, Enum):
    """Types of agent steps."""
    USER_MESSAGE = "user_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    ERROR = "error"


@dataclass
class AgentStep:
    """Represents a single step in agent execution."""
    step_number: int
    step_type: StepType
    content: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentResponse:
    """Complete response from the agent."""
    answer: str
    steps: list[AgentStep]
    tools_used: list[str]
    total_steps: int
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "tools_used": self.tools_used,
            "total_steps": self.total_steps,
            "success": self.success,
            "error": self.error
        }


class DocOpsAgent:
    """Multi-step reasoning agent for document operations.

    This agent follows the Elasticsearch Agent Builder pattern:
    1. Receive user query
    2. Determine which tools to use
    3. Execute tools in sequence, maintaining context
    4. Synthesize results into a final answer
    """

    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
        max_steps: int = 10,
        es_host: str = "localhost",
        es_port: int = 9200,
        es_scheme: str = "http",
    ):
        """Initialize the DocOps agent.

        Args:
            tools: Pre-configured AgentTools instance.
            llm_provider: LLM provider (openai, anthropic, or local).
            model: Model name to use.
            max_steps: Maximum number of tool calls per query.
            es_host: Elasticsearch host.
            es_port: Elasticsearch port.
            es_scheme: http or https.
        """
        self.tools = tools or AgentTools(
            host=es_host,
            port=es_port,
            scheme=es_scheme
        )
        self.llm_provider = llm_provider
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = get_system_prompt()

        # Track conversation history
        self._conversation_history: list[dict] = []
        self._step_counter = 0

    @property
    def tool_definitions(self) -> list[dict]:
        """Get tool definitions for the LLM."""
        return self.tools.get_tool_definitions()

    def chat(self, message: str) -> AgentResponse:
        """Process a user message and return a response.

        This is the main entry point for the agent. It:
        1. Analyzes the user's request
        2. Determines which tools to use
        3. Executes tools and maintains context
        4. Synthesizes a final response

        Args:
            message: The user's message.

        Returns:
            AgentResponse with answer, steps, and tools used.
        """
        steps: list[AgentStep] = []
        tools_used: list[str] = []
        self._step_counter = 0

        # Record user message
        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.USER_MESSAGE,
            content=message
        ))

        try:
            # Determine intent and execute appropriate workflow
            response = self._process_message(message, steps, tools_used)

            return AgentResponse(
                answer=response,
                steps=steps,
                tools_used=list(set(tools_used)),
                total_steps=len(steps),
                success=True
            )

        except Exception as e:
            self._step_counter += 1
            steps.append(AgentStep(
                step_number=self._step_counter,
                step_type=StepType.ERROR,
                content=str(e)
            ))

            return AgentResponse(
                answer=f"I encountered an error while processing your request: {str(e)}",
                steps=steps,
                tools_used=list(set(tools_used)),
                total_steps=len(steps),
                success=False,
                error=str(e)
            )

    def _process_message(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Process a message through the agent workflow.

        Args:
            message: User message.
            steps: List to append steps to.
            tools_used: List to append tool names to.

        Returns:
            Final response string.
        """
        message_lower = message.lower()

        # Determine the intent and route to appropriate handler
        # Check for conflict/consistency queries first (highest priority)
        conflict_keywords = ["conflict", "contradict", "inconsisten", "differ", "mismatch", "discrepancy", "disagree", "clash"]
        if any(word in message_lower for word in conflict_keywords):
            return self._handle_consistency_query(message, steps, tools_used)

        # Check for staleness queries
        staleness_keywords = ["stale", "outdated", "expired", "old", "update", "refresh", "renew", "obsolete", "dated"]
        if any(word in message_lower for word in staleness_keywords):
            return self._handle_staleness_query(message, steps, tools_used)

        # Check for gap/coverage queries
        gap_keywords = ["gap", "missing", "coverage", "cover", "lack", "absent", "incomplete"]
        if any(word in message_lower for word in gap_keywords):
            return self._handle_gap_query(message, steps, tools_used)

        # Check for health/status queries
        health_keywords = ["health", "status", "overview", "summary", "dashboard", "metrics", "statistics"]
        if any(word in message_lower for word in health_keywords):
            return self._handle_health_query(message, steps, tools_used)

        # Check for report generation
        report_keywords = ["report", "generate", "create report", "audit", "analyze all", "full analysis"]
        if any(word in message_lower for word in report_keywords):
            return self._handle_report_query(message, steps, tools_used)

        # Check for "check" or "scan" which often means consistency check
        if any(word in message_lower for word in ["check", "scan", "verify", "validate", "compare"]):
            return self._handle_consistency_query(message, steps, tools_used)

        # Default: search and summarize
        return self._handle_search_query(message, steps, tools_used)

    def _handle_consistency_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle queries about document consistency."""
        # Extract topic if mentioned
        topic = self._extract_topic(message)

        # Step 1: Search for relevant documents
        self._step_counter += 1
        search_result = self.tools.search_documents(
            query=topic or message,
            top_k=10
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "search_documents", "query": topic or message}
        ))
        tools_used.append("search_documents")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content={"hits": search_result.data.get("total_hits", 0) if search_result.success else 0}
        ))

        # Step 2: Run consistency check
        self._step_counter += 1
        consistency_result = self.tools.run_consistency_check(focus_topic=topic)
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "run_consistency_check", "topic": topic}
        ))
        tools_used.append("run_consistency_check")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content=consistency_result.data if consistency_result.success else {"error": consistency_result.error}
        ))

        # Step 3: Create alerts for critical issues
        if consistency_result.success:
            inconsistencies = consistency_result.data.get("inconsistencies", [])
            critical = [i for i in inconsistencies if i.get("severity") == "critical"]

            for issue in critical[:3]:  # Limit alerts
                self._step_counter += 1
                alert_result = self.tools.create_alert(
                    severity="critical",
                    title=f"Conflict: {issue.get('description', 'Unknown')}",
                    description=f"Conflict between {issue.get('document_a')} and {issue.get('document_b')}: {issue.get('value_a')} vs {issue.get('value_b')}",
                    affected_doc_ids=[]
                )
                steps.append(AgentStep(
                    step_number=self._step_counter,
                    step_type=StepType.TOOL_CALL,
                    content={"tool": "create_alert", "title": issue.get('description')}
                ))
                tools_used.append("create_alert")

        # Synthesize response
        return self._synthesize_consistency_response(
            search_result.data if search_result.success else {},
            consistency_result.data if consistency_result.success else {}
        )

    def _handle_staleness_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle queries about document staleness."""
        # Step 1: Get document health (includes staleness)
        self._step_counter += 1
        health_result = self.tools.get_document_health(
            include_staleness=True,
            include_gaps=False
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "get_document_health", "include_staleness": True}
        ))
        tools_used.append("get_document_health")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content=health_result.data if health_result.success else {"error": health_result.error}
        ))

        # Step 2: Generate staleness report
        self._step_counter += 1
        report_result = self.tools.generate_report(
            report_type="staleness",
            include_recommendations=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "generate_report", "type": "staleness"}
        ))
        tools_used.append("generate_report")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content={"sections": len(report_result.data.get("sections", [])) if report_result.success else 0}
        ))

        return self._synthesize_staleness_response(
            health_result.data if health_result.success else {},
            report_result.data if report_result.success else {}
        )

    def _handle_gap_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle queries about coverage gaps."""
        # Step 1: Get document health (includes gaps)
        self._step_counter += 1
        health_result = self.tools.get_document_health(
            include_staleness=False,
            include_gaps=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "get_document_health", "include_gaps": True}
        ))
        tools_used.append("get_document_health")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content=health_result.data if health_result.success else {"error": health_result.error}
        ))

        # Step 2: Generate gap report
        self._step_counter += 1
        report_result = self.tools.generate_report(
            report_type="gap",
            include_recommendations=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "generate_report", "type": "gap"}
        ))
        tools_used.append("generate_report")

        return self._synthesize_gap_response(
            health_result.data if health_result.success else {},
            report_result.data if report_result.success else {}
        )

    def _handle_health_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle queries about overall document health."""
        # Step 1: Get comprehensive health
        self._step_counter += 1
        health_result = self.tools.get_document_health(
            include_staleness=True,
            include_gaps=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "get_document_health"}
        ))
        tools_used.append("get_document_health")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content=health_result.data if health_result.success else {"error": health_result.error}
        ))

        # Step 2: Generate summary report
        self._step_counter += 1
        report_result = self.tools.generate_report(
            report_type="summary",
            include_recommendations=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "generate_report", "type": "summary"}
        ))
        tools_used.append("generate_report")

        return self._synthesize_health_response(
            health_result.data if health_result.success else {},
            report_result.data if report_result.success else {}
        )

    def _handle_report_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle requests to generate reports."""
        # Determine report type from message
        message_lower = message.lower()
        if "conflict" in message_lower:
            report_type = "conflict"
        elif "stale" in message_lower or "outdated" in message_lower:
            report_type = "staleness"
        elif "gap" in message_lower or "coverage" in message_lower:
            report_type = "gap"
        elif "compliance" in message_lower:
            report_type = "compliance"
        else:
            report_type = "summary"

        # Generate the report
        self._step_counter += 1
        report_result = self.tools.generate_report(
            report_type=report_type,
            include_recommendations=True
        )
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "generate_report", "type": report_type}
        ))
        tools_used.append("generate_report")

        if report_result.success:
            report = report_result.data
            sections = report.get("sections", [])
            recommendations = report.get("recommendations", [])

            response = f"## {report_type.title()} Report\n\n"
            for section in sections:
                response += f"### {section.get('title', 'Section')}\n"
                response += f"{section.get('content', '')}\n\n"

            if recommendations:
                response += "### Recommendations\n"
                for rec in recommendations:
                    response += f"- {rec}\n"

            return response
        else:
            return f"I couldn't generate the report: {report_result.error}"

    def _handle_search_query(
        self,
        message: str,
        steps: list[AgentStep],
        tools_used: list[str]
    ) -> str:
        """Handle general search queries."""
        # Search for relevant content
        self._step_counter += 1
        search_result = self.tools.search_documents(query=message, top_k=5)
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_CALL,
            content={"tool": "search_documents", "query": message}
        ))
        tools_used.append("search_documents")

        self._step_counter += 1
        steps.append(AgentStep(
            step_number=self._step_counter,
            step_type=StepType.TOOL_RESULT,
            content={"hits": search_result.data.get("total_hits", 0) if search_result.success else 0}
        ))

        if search_result.success and search_result.data.get("results"):
            results = search_result.data["results"]

            response = f"I found {len(results)} relevant sections:\n\n"
            for i, result in enumerate(results, 1):
                response += f"**{i}. {result.get('document_title', 'Unknown')}** - {result.get('section_title', 'Section')}\n"
                content = result.get('content', '')[:200]
                response += f"> {content}...\n\n"

            return response
        else:
            # If no search results, provide a helpful overview
            self._step_counter += 1
            health_result = self.tools.get_document_health()
            steps.append(AgentStep(
                step_number=self._step_counter,
                step_type=StepType.TOOL_CALL,
                content={"tool": "get_document_health"}
            ))
            tools_used.append("get_document_health")

            if health_result.success:
                data = health_result.data
                response = "I couldn't find specific results for that query, but here's what I know about your document corpus:\n\n"
                response += f"- **Documents:** {data.get('document_count', 0)}\n"
                response += f"- **Conflicts:** {data.get('total_conflicts', 0)}\n"
                response += f"- **Staleness Issues:** {data.get('staleness_issues', 0)}\n"
                response += f"- **Coverage Gaps:** {data.get('coverage_gaps', 0)}\n\n"
                response += "Try asking about specific topics like 'password policy', 'conflicts', 'expired documents', or 'coverage gaps'."
                return response
            else:
                return "I couldn't find relevant information for your query. Try asking about 'conflicts', 'expired documents', or 'coverage gaps'."

    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract a topic from the user message."""
        # Common topics to look for
        topics = [
            "password", "security", "remote", "work", "expense",
            "leave", "vacation", "termination", "data", "retention",
            "access", "authentication", "compliance", "policy"
        ]

        message_lower = message.lower()
        for topic in topics:
            if topic in message_lower:
                return topic

        return None

    def _synthesize_consistency_response(
        self,
        search_data: dict,
        consistency_data: dict
    ) -> str:
        """Synthesize a response for consistency queries."""
        inconsistencies = consistency_data.get("inconsistencies", [])
        total = consistency_data.get("total_inconsistencies", 0)
        topic = consistency_data.get("focus_topic", "all topics")

        if total == 0:
            return f"Good news! I found no inconsistencies when checking documents for {topic}."

        response = f"## Consistency Check Results\n\n"
        response += f"I found **{total} inconsistencies** related to {topic}:\n\n"

        for i, issue in enumerate(inconsistencies, 1):
            severity = issue.get("severity", "unknown").upper()
            response += f"### {i}. [{severity}] {issue.get('description', 'Issue')}\n"
            response += f"- **Document A**: {issue.get('document_a', 'Unknown')} states: `{issue.get('value_a', 'N/A')}`\n"
            response += f"- **Document B**: {issue.get('document_b', 'Unknown')} states: `{issue.get('value_b', 'N/A')}`\n\n"

        response += "### Recommended Actions\n"
        response += "1. Review the conflicting documents with stakeholders\n"
        response += "2. Determine which value should be the source of truth\n"
        response += "3. Update the outdated document to match\n"

        return response

    def _synthesize_staleness_response(
        self,
        health_data: dict,
        report_data: dict
    ) -> str:
        """Synthesize a response for staleness queries."""
        expired = health_data.get("expired_documents", 0)
        outdated = health_data.get("outdated_documents", 0)
        total = health_data.get("staleness_issues", 0)

        response = f"## Document Staleness Analysis\n\n"
        response += f"**Summary**: Found {total} staleness issues\n"
        response += f"- Expired documents: {expired}\n"
        response += f"- Outdated documents: {outdated}\n\n"

        sections = report_data.get("sections", [])
        for section in sections[1:6]:  # Skip summary, show up to 5
            response += f"### {section.get('title', 'Document')}\n"
            response += f"{section.get('content', '')}\n"
            if section.get('recommended_action'):
                response += f"**Action**: {section.get('recommended_action')}\n"
            response += "\n"

        recommendations = report_data.get("recommendations", [])
        if recommendations:
            response += "### Recommendations\n"
            for rec in recommendations:
                response += f"- {rec}\n"

        return response

    def _synthesize_gap_response(
        self,
        health_data: dict,
        report_data: dict
    ) -> str:
        """Synthesize a response for gap queries."""
        total_gaps = health_data.get("coverage_gaps", 0)
        critical_gaps = health_data.get("critical_gaps", 0)

        response = f"## Coverage Gap Analysis\n\n"
        response += f"**Summary**: Found {total_gaps} coverage gaps ({critical_gaps} critical/high priority)\n\n"

        sections = report_data.get("sections", [])
        for section in sections[1:8]:  # Skip summary, show up to 7
            response += f"### {section.get('title', 'Gap')}\n"
            response += f"{section.get('content', '')}\n"
            if section.get('covered_in'):
                response += f"- Covered in: {', '.join(section.get('covered_in', []))}\n"
            if section.get('missing_from'):
                response += f"- Missing from: {', '.join(section.get('missing_from', []))}\n"
            response += "\n"

        recommendations = report_data.get("recommendations", [])
        if recommendations:
            response += "### Recommendations\n"
            for rec in recommendations:
                response += f"- {rec}\n"

        return response

    def _synthesize_health_response(
        self,
        health_data: dict,
        report_data: dict
    ) -> str:
        """Synthesize a response for health queries."""
        score = health_data.get("health_score", 0)
        status = health_data.get("status", "unknown")

        response = f"## Document Corpus Health Report\n\n"
        response += f"**Health Score**: {score}/100 ({status.upper()})\n\n"

        response += "### Key Metrics\n"
        response += f"- Total Documents: {health_data.get('document_count', 0)}\n"
        response += f"- Total Chunks: {health_data.get('chunk_count', 0)}\n"
        response += f"- Open Alerts: {health_data.get('open_alerts', 0)}\n"
        response += f"- Conflicts: {health_data.get('total_conflicts', 0)} ({health_data.get('critical_conflicts', 0)} critical)\n"
        response += f"- Staleness Issues: {health_data.get('staleness_issues', 0)}\n"
        response += f"- Coverage Gaps: {health_data.get('coverage_gaps', 0)}\n\n"

        sections = report_data.get("sections", [])
        for section in sections:
            response += f"### {section.get('title', 'Section')}\n"
            response += f"{section.get('content', '')}\n\n"

        recommendations = report_data.get("recommendations", [])
        if recommendations:
            response += "### Priority Actions\n"
            for rec in recommendations:
                response += f"- {rec}\n"

        return response

    def reset_conversation(self):
        """Reset the conversation history."""
        self._conversation_history = []
        self._step_counter = 0
