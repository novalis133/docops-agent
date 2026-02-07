"""DocOps Agent - Streamlit Frontend.

Provides a polished UI for document operations with:
- Real-time corpus health dashboard
- Agent chat with step-by-step trace
- Conflict viewer with side-by-side comparison
- Report generation and download
"""

import streamlit as st
import sys
import json
import time
from pathlib import Path
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings

st.set_page_config(
    page_title="DocOps Agent",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_health_data():
    """Get corpus health data with error handling."""
    try:
        from src.main import get_corpus_health
        return get_corpus_health()
    except Exception as e:
        return {
            "status": "unknown",
            "color": "gray",
            "document_count": 0,
            "chunk_count": 0,
            "open_alerts": 0,
            "critical_alerts": 0,
            "error": str(e)
        }


def main():
    """Main Streamlit application."""
    st.title("DocOps Agent")
    st.markdown("*Intelligent Document Operations Platform*")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Agent Chat", "Conflict Viewer", "Reports", "Search", "Upload"],
            label_visibility="collapsed",
        )

        st.divider()

        # Health status with color coding
        st.header("Corpus Health")
        health = get_health_data()

        status_colors = {
            "healthy": ("green", ":green_circle:"),
            "warning": ("orange", ":large_orange_circle:"),
            "critical": ("red", ":red_circle:"),
            "unknown": ("gray", ":white_circle:")
        }
        color, icon = status_colors.get(health.get("status", "unknown"), ("gray", ":white_circle:"))

        st.markdown(f"{icon} **{health.get('status', 'Unknown').title()}**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", health.get("document_count", 0))
        with col2:
            st.metric("Alerts", health.get("open_alerts", 0))

        if health.get("critical_alerts", 0) > 0:
            st.error(f":warning: {health['critical_alerts']} critical alerts!")

    # Main content
    if page == "Dashboard":
        render_dashboard()
    elif page == "Agent Chat":
        render_agent_chat()
    elif page == "Conflict Viewer":
        render_conflict_viewer()
    elif page == "Reports":
        render_reports()
    elif page == "Search":
        render_search()
    elif page == "Upload":
        render_upload()


def render_dashboard():
    """Render the dashboard page with health metrics."""
    st.header("Document Health Dashboard")

    health = get_health_data()

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Documents",
            health.get("document_count", 0),
            help="Number of documents in the corpus"
        )
    with col2:
        st.metric(
            "Total Chunks",
            health.get("chunk_count", 0),
            help="Number of indexed text chunks"
        )
    with col3:
        open_alerts = health.get("open_alerts", 0)
        st.metric(
            "Open Alerts",
            open_alerts,
            delta=None if open_alerts == 0 else f"-{open_alerts}" if open_alerts < 0 else None,
            delta_color="inverse"
        )
    with col4:
        critical = health.get("critical_alerts", 0)
        st.metric(
            "Critical Issues",
            critical,
            help="Requires immediate attention"
        )

    st.divider()

    # Health score visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Health Score")
        try:
            from src.agent import AgentTools
            tools = AgentTools()
            health_result = tools.get_document_health()

            if health_result.success:
                score = health_result.data.get("health_score", 0)

                # Progress bar with color
                if score >= 80:
                    st.progress(score / 100, text=f"Score: {score}/100 - Healthy")
                elif score >= 50:
                    st.progress(score / 100, text=f"Score: {score}/100 - Warning")
                else:
                    st.progress(score / 100, text=f"Score: {score}/100 - Critical")

                # Issue breakdown
                st.markdown("**Issue Breakdown:**")
                issues = []
                if health_result.data.get("total_conflicts", 0) > 0:
                    issues.append(f"- :warning: {health_result.data['total_conflicts']} conflicts")
                if health_result.data.get("staleness_issues", 0) > 0:
                    issues.append(f"- :clock3: {health_result.data['staleness_issues']} staleness issues")
                if health_result.data.get("coverage_gaps", 0) > 0:
                    issues.append(f"- :hole: {health_result.data['coverage_gaps']} coverage gaps")

                if issues:
                    for issue in issues:
                        st.markdown(issue)
                else:
                    st.success("No issues detected!")

        except Exception as e:
            st.info("Health score analysis unavailable")
            st.caption(str(e)[:100])

    with col2:
        st.subheader("Quick Actions")
        if st.button("Run Conflict Scan", use_container_width=True):
            st.session_state["quick_action"] = "conflict_scan"
            st.rerun()

        if st.button("Check Staleness", use_container_width=True):
            st.session_state["quick_action"] = "staleness_audit"
            st.rerun()

        if st.button("Gap Analysis", use_container_width=True):
            st.session_state["quick_action"] = "gap_analysis"
            st.rerun()

    # Execute quick action if triggered
    if st.session_state.get("quick_action"):
        action = st.session_state.pop("quick_action")
        with st.spinner(f"Running {action}..."):
            try:
                from src.agent import WorkflowEngine
                engine = WorkflowEngine()
                result = engine.execute_workflow(action)

                st.subheader(f"Results: {action}")
                st.write(result.summary)

                with st.expander("View Details"):
                    st.json(result.details)

            except Exception as e:
                st.error(f"Action failed: {e}")

    st.divider()

    # Recent alerts
    st.subheader("Recent Alerts")
    try:
        from src.main import create_indexer
        indexer = create_indexer()
        alerts = indexer.get_alerts(top_k=10)

        if alerts:
            for alert in alerts:
                severity = alert.get("severity", "low")
                severity_icons = {
                    "critical": ":red_circle:",
                    "high": ":large_orange_circle:",
                    "medium": ":yellow_circle:",
                    "low": ":green_circle:"
                }
                icon = severity_icons.get(severity, ":white_circle:")

                with st.expander(f"{icon} {alert.get('title', 'Alert')} ({severity})"):
                    st.write(alert.get("description", "No description"))
                    st.caption(f"Document: {alert.get('document_id')} | Created: {alert.get('created_at', 'Unknown')}")
        else:
            st.success("No open alerts! Your documentation is in good shape.")

    except Exception as e:
        st.info("Unable to load alerts")
        st.caption(str(e)[:50])


def render_agent_chat():
    """Render the agent chat page with step-by-step trace."""
    st.header("Agent Chat")
    st.info("Chat with DocOps Agent to analyze your documents. The agent uses multi-step reasoning with specialized tools.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []

    # Chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show steps for assistant messages
                if message["role"] == "assistant" and i < len(st.session_state.agent_steps):
                    steps = st.session_state.agent_steps[i]
                    if steps:
                        with st.expander("View Agent Steps"):
                            for step in steps:
                                step_type = step.get("step_type", "unknown")
                                if step_type == "tool_call":
                                    st.markdown(f":wrench: **Tool Call**: `{step.get('content', {}).get('tool', 'unknown')}`")
                                elif step_type == "tool_result":
                                    st.markdown(f":white_check_mark: **Result**: {json.dumps(step.get('content', {}), indent=2)[:200]}...")

        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from src.agent import DocOpsAgent
                        agent = DocOpsAgent()
                        response = agent.chat(prompt)

                        # Display response
                        st.markdown(response.answer)

                        # Store steps
                        steps_data = [s.to_dict() for s in response.steps]
                        st.session_state.agent_steps.append(steps_data)

                        # Show step trace
                        if response.steps:
                            with st.expander("View Agent Steps", expanded=True):
                                st.markdown(f"**Tools Used:** {', '.join(response.tools_used)}")
                                st.markdown(f"**Total Steps:** {response.total_steps}")

                                for step in response.steps:
                                    step_dict = step.to_dict()
                                    step_type = step_dict.get("step_type")

                                    if step_type == "user_message":
                                        st.markdown(f":speech_balloon: **Input**: {step_dict.get('content', '')[:100]}...")
                                    elif step_type == "tool_call":
                                        content = step_dict.get("content", {})
                                        st.markdown(f":wrench: **Tool**: `{content.get('tool', 'unknown')}`")
                                        if content.get("query"):
                                            st.caption(f"Query: {content['query']}")
                                    elif step_type == "tool_result":
                                        st.markdown(":white_check_mark: **Result received**")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer
                        })

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.session_state.agent_steps.append([])

    with col2:
        st.subheader("Quick Queries")

        quick_queries = [
            "What is the password policy?",
            "Check for conflicts about passwords",
            "Are any documents expired?",
            "What coverage gaps exist?",
            "Give me a health summary"
        ]

        for query in quick_queries:
            if st.button(query, key=f"quick_{query[:20]}", use_container_width=True):
                # Trigger the chat input programmatically
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()

        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_steps = []
            st.rerun()


def render_conflict_viewer():
    """Render the conflict viewer with side-by-side comparison."""
    st.header("Conflict Viewer")
    st.info("View document conflicts side-by-side to understand discrepancies.")

    try:
        from src.analysis import ConflictDetector

        # Topic filter
        topic = st.selectbox(
            "Filter by Topic",
            ["All Topics", "password", "security", "remote", "data", "expense"]
        )

        with st.spinner("Detecting conflicts..."):
            detector = ConflictDetector()

            if topic == "All Topics":
                conflicts = detector.detect_all_conflicts()
            else:
                conflicts = detector.detect_conflicts(topic=topic)

        if not conflicts:
            st.success("No conflicts found!")
            return

        st.subheader(f"Found {len(conflicts)} Conflicts")

        for i, conflict in enumerate(conflicts):
            severity_colors = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow",
                "low": "green"
            }
            color = severity_colors.get(conflict.severity.value, "gray")

            with st.expander(
                f":{'red' if color == 'red' else 'large_orange' if color == 'orange' else 'yellow' if color == 'yellow' else 'green'}_circle: "
                f"**{conflict.description}** ({conflict.severity.value.upper()})",
                expanded=(i == 0)
            ):
                st.markdown(f"**Topic:** {conflict.topic}")
                st.markdown(f"**Type:** {conflict.conflict_type.value}")

                # Side-by-side comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"#### {conflict.location_a.document_title}")
                    st.markdown(f"*Section: {conflict.location_a.section_title}*")

                    # Highlight the conflicting value
                    st.markdown(f"""
                    <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">
                    <strong>Value:</strong> {conflict.value_a}
                    </div>
                    """, unsafe_allow_html=True)

                    if conflict.location_a.content:
                        with st.expander("View Context"):
                            st.text(conflict.location_a.content[:500])

                with col2:
                    st.markdown(f"#### {conflict.location_b.document_title}")
                    st.markdown(f"*Section: {conflict.location_b.section_title}*")

                    st.markdown(f"""
                    <div style="background-color: #ccccff; padding: 10px; border-radius: 5px;">
                    <strong>Value:</strong> {conflict.value_b}
                    </div>
                    """, unsafe_allow_html=True)

                    if conflict.location_b.content:
                        with st.expander("View Context"):
                            st.text(conflict.location_b.content[:500])

                # Action buttons
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Create Alert", key=f"alert_{conflict.id}"):
                        try:
                            from src.actions import AlertManager
                            manager = AlertManager()
                            alert_id = manager.create_alert(
                                document_id=conflict.location_a.document_id,
                                alert_type="conflict",
                                severity=conflict.severity.value,
                                title=conflict.description,
                                description=f"{conflict.value_a} vs {conflict.value_b}"
                            )
                            if alert_id:
                                st.success(f"Alert created: {alert_id}")
                            else:
                                st.info("Alert already exists (deduplicated)")
                        except Exception as e:
                            st.error(f"Failed to create alert: {e}")

    except Exception as e:
        st.error(f"Unable to load conflict detector: {e}")
        st.info("Make sure Elasticsearch is running and data is indexed.")


def render_reports():
    """Render the reports page with generation and download."""
    st.header("Reports")

    report_type = st.selectbox(
        "Select Report Type",
        ["Conflict Analysis", "Staleness Report", "Coverage Gap Analysis", "Compliance Audit"]
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        include_recommendations = st.checkbox("Include Recommendations", value=True)

    with col2:
        output_format = st.radio("Output Format", ["Markdown", "PDF"], horizontal=True)

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                from src.actions import ReportGenerator
                from src.analysis import ConflictDetector, StalenessChecker, GapAnalyzer

                # Initialize components
                detector = ConflictDetector()
                staleness = StalenessChecker()
                gap_analyzer = GapAnalyzer()

                generator = ReportGenerator(
                    conflict_detector=detector,
                    staleness_checker=staleness,
                    gap_analyzer=gap_analyzer
                )

                # Generate report
                if report_type == "Conflict Analysis":
                    report_content = generator.generate_conflict_report(
                        include_recommendations=include_recommendations
                    )
                elif report_type == "Staleness Report":
                    report_content = generator.generate_staleness_report(
                        include_recommendations=include_recommendations
                    )
                elif report_type == "Coverage Gap Analysis":
                    report_content = generator.generate_gap_report(
                        include_recommendations=include_recommendations
                    )
                else:  # Compliance Audit
                    report_content = generator.generate_compliance_report()

                # Display report
                st.subheader("Generated Report")
                st.markdown(report_content)

                # Download button
                st.divider()

                if output_format == "Markdown":
                    st.download_button(
                        label="Download Markdown",
                        data=report_content,
                        file_name=f"{report_type.lower().replace(' ', '_')}_report.md",
                        mime="text/markdown"
                    )
                else:
                    # Generate PDF
                    try:
                        pdf_bytes = generator.to_pdf(report_content)
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"{report_type.lower().replace(' ', '_')}_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.warning(f"PDF generation failed: {e}")
                        st.download_button(
                            label="Download Markdown Instead",
                            data=report_content,
                            file_name=f"{report_type.lower().replace(' ', '_')}_report.md",
                            mime="text/markdown"
                        )

            except Exception as e:
                st.error(f"Failed to generate report: {e}")
                st.info("Make sure Elasticsearch is running and documents are indexed.")


def render_search():
    """Render the search page."""
    st.header("Document Search")

    query = st.text_input("Search query", placeholder="Enter your search...")

    col1, col2 = st.columns([1, 3])
    with col1:
        use_hybrid = st.checkbox("Use hybrid search", value=True)
        top_k = st.slider("Results", 1, 20, 10)

    if query:
        with st.spinner("Searching..."):
            try:
                from src.main import search

                results = search(query, top_k=top_k, use_hybrid=use_hybrid)

                st.subheader(f"Results ({len(results)})")

                for i, result in enumerate(results, 1):
                    score = result.get("score", 0)
                    with st.expander(
                        f"{i}. {result.get('section_title', 'Result')} "
                        f"(Score: {score:.3f})"
                    ):
                        st.markdown(f"**Document:** {result.get('document_title', 'Unknown')}")
                        st.markdown(f"**Section:** {result.get('section_title', 'Unknown')}")
                        st.text(result.get("content", "")[:500] + "...")

            except Exception as e:
                st.error(f"Search failed: {e}")


def render_upload():
    """Render the upload page."""
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest Documents", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status.text(f"Processing: {uploaded_file.name}")

                try:
                    import tempfile
                    from src.main import ingest_document

                    # Save to temp file
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Ingest
                    result = ingest_document(tmp_path)
                    st.success(
                        f"Ingested: {uploaded_file.name} "
                        f"({result['chunk_count']} chunks)"
                    )

                    # Clean up
                    Path(tmp_path).unlink(missing_ok=True)

                except Exception as e:
                    st.error(f"Failed to ingest {uploaded_file.name}: {e}")

                progress.progress((i + 1) / len(uploaded_files))

            status.text("Processing complete!")


if __name__ == "__main__":
    main()
