"""Report generation for DocOps analysis results.

Generates Markdown and PDF reports from conflict, staleness, gap,
and compliance analysis results.
"""

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

from ..analysis import ConflictDetector, StalenessChecker, GapAnalyzer


@dataclass
class ReportSection:
    """A section in a generated report."""
    title: str
    content: str
    level: int = 2  # Heading level (1-4)


class ReportGenerator:
    """Generate structured reports from analysis results.

    Supports Markdown and PDF output formats.
    """

    def __init__(
        self,
        conflict_detector: Optional[ConflictDetector] = None,
        staleness_checker: Optional[StalenessChecker] = None,
        gap_analyzer: Optional[GapAnalyzer] = None,
    ):
        """Initialize the report generator.

        Args:
            conflict_detector: For conflict reports.
            staleness_checker: For staleness reports.
            gap_analyzer: For gap analysis reports.
        """
        self.conflict_detector = conflict_detector
        self.staleness_checker = staleness_checker
        self.gap_analyzer = gap_analyzer

    def generate_conflict_report(
        self,
        topic: Optional[str] = None,
        include_recommendations: bool = True
    ) -> str:
        """Generate a conflict analysis report.

        Args:
            topic: Optional topic to focus on.
            include_recommendations: Whether to include remediation suggestions.

        Returns:
            Markdown-formatted report string.
        """
        sections = [
            ReportSection(
                title="Conflict Analysis Report",
                content=f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                level=1
            )
        ]

        if self.conflict_detector is None:
            sections.append(ReportSection(
                title="Error",
                content="Conflict detector not configured. Unable to generate report."
            ))
            return self._render_markdown(sections)

        # Get conflicts
        if topic:
            conflicts = self.conflict_detector.detect_conflicts(topic=topic)
            sections.append(ReportSection(
                title="Scope",
                content=f"Analysis focused on topic: **{topic}**"
            ))
        else:
            conflicts = self.conflict_detector.detect_all_conflicts()
            sections.append(ReportSection(
                title="Scope",
                content="Full corpus analysis"
            ))

        # Summary
        if not conflicts:
            sections.append(ReportSection(
                title="Summary",
                content="No conflicts found in the analyzed documents."
            ))
            return self._render_markdown(sections)

        # Count by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for c in conflicts:
            sev = c.severity.value
            if sev in by_severity:
                by_severity[sev].append(c)

        summary_lines = [
            f"Found **{len(conflicts)} conflicts** across the document corpus:",
            "",
            f"- Critical: {len(by_severity['critical'])}",
            f"- High: {len(by_severity['high'])}",
            f"- Medium: {len(by_severity['medium'])}",
            f"- Low: {len(by_severity['low'])}"
        ]
        sections.append(ReportSection(
            title="Summary",
            content="\n".join(summary_lines)
        ))

        # Detailed findings by severity
        for severity in ["critical", "high", "medium", "low"]:
            if by_severity[severity]:
                sections.append(ReportSection(
                    title=f"{severity.upper()} Severity Conflicts",
                    content=self._format_conflicts(by_severity[severity])
                ))

        # Recommendations
        if include_recommendations and conflicts:
            sections.append(ReportSection(
                title="Recommendations",
                content=self._get_conflict_recommendations(conflicts)
            ))

        return self._render_markdown(sections)

    def generate_staleness_report(
        self,
        include_recommendations: bool = True
    ) -> str:
        """Generate a staleness analysis report.

        Args:
            include_recommendations: Whether to include update suggestions.

        Returns:
            Markdown-formatted report string.
        """
        sections = [
            ReportSection(
                title="Document Staleness Report",
                content=f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                level=1
            )
        ]

        if self.staleness_checker is None:
            sections.append(ReportSection(
                title="Error",
                content="Staleness checker not configured. Unable to generate report."
            ))
            return self._render_markdown(sections)

        # Get staleness issues
        issues = self.staleness_checker.check_all_documents()

        if not issues:
            sections.append(ReportSection(
                title="Summary",
                content="No staleness issues found. All documents appear to be current."
            ))
            return self._render_markdown(sections)

        # Count by type
        expired = [i for i in issues if i.staleness_type.value == "expired"]
        outdated = [i for i in issues if i.staleness_type.value == "outdated_year"]
        stale_review = [i for i in issues if i.staleness_type.value == "stale_review"]

        summary_lines = [
            f"Found **{len(issues)} staleness issues**:",
            "",
            f"- Expired documents: {len(expired)}",
            f"- Outdated year references: {len(outdated)}",
            f"- Overdue for review: {len(stale_review)}"
        ]
        sections.append(ReportSection(
            title="Summary",
            content="\n".join(summary_lines)
        ))

        # Detailed findings
        if expired:
            sections.append(ReportSection(
                title="Expired Documents (Immediate Action Required)",
                content=self._format_staleness_issues(expired)
            ))

        if outdated:
            sections.append(ReportSection(
                title="Documents with Outdated Year References",
                content=self._format_staleness_issues(outdated)
            ))

        if stale_review:
            sections.append(ReportSection(
                title="Documents Overdue for Review",
                content=self._format_staleness_issues(stale_review)
            ))

        # Recommendations
        if include_recommendations:
            sections.append(ReportSection(
                title="Recommendations",
                content=self._get_staleness_recommendations(issues)
            ))

        return self._render_markdown(sections)

    def generate_gap_report(
        self,
        include_recommendations: bool = True
    ) -> str:
        """Generate a coverage gap analysis report.

        Args:
            include_recommendations: Whether to include coverage suggestions.

        Returns:
            Markdown-formatted report string.
        """
        sections = [
            ReportSection(
                title="Coverage Gap Analysis Report",
                content=f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                level=1
            )
        ]

        if self.gap_analyzer is None:
            sections.append(ReportSection(
                title="Error",
                content="Gap analyzer not configured. Unable to generate report."
            ))
            return self._render_markdown(sections)

        # Get gaps
        gaps = self.gap_analyzer.analyze_all()

        if not gaps:
            sections.append(ReportSection(
                title="Summary",
                content="No coverage gaps found. Document corpus has consistent topic coverage."
            ))
            return self._render_markdown(sections)

        # Count by severity
        critical = [g for g in gaps if g.severity.value == "critical"]
        high = [g for g in gaps if g.severity.value == "high"]
        medium = [g for g in gaps if g.severity.value == "medium"]
        low = [g for g in gaps if g.severity.value == "low"]

        summary_lines = [
            f"Found **{len(gaps)} coverage gaps**:",
            "",
            f"- Critical: {len(critical)}",
            f"- High: {len(high)}",
            f"- Medium: {len(medium)}",
            f"- Low: {len(low)}"
        ]
        sections.append(ReportSection(
            title="Summary",
            content="\n".join(summary_lines)
        ))

        # Priority gaps
        priority_gaps = critical + high
        if priority_gaps:
            sections.append(ReportSection(
                title="Priority Gaps (Critical/High)",
                content=self._format_gaps(priority_gaps[:10])
            ))

        # Other gaps
        other_gaps = medium + low
        if other_gaps:
            sections.append(ReportSection(
                title="Other Gaps",
                content=self._format_gaps(other_gaps[:10])
            ))

        # Recommendations
        if include_recommendations:
            sections.append(ReportSection(
                title="Recommendations",
                content=self._get_gap_recommendations(gaps)
            ))

        return self._render_markdown(sections)

    def generate_compliance_report(
        self,
        focus_areas: Optional[list[str]] = None
    ) -> str:
        """Generate a comprehensive compliance audit report.

        Args:
            focus_areas: Optional list of areas to focus on.

        Returns:
            Markdown-formatted report string.
        """
        sections = [
            ReportSection(
                title="Compliance Audit Report",
                content=f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                level=1
            )
        ]

        if focus_areas:
            sections.append(ReportSection(
                title="Audit Scope",
                content=f"Focus areas: {', '.join(focus_areas)}"
            ))

        # Conflicts section
        if self.conflict_detector:
            conflicts = self.conflict_detector.detect_all_conflicts()
            critical_conflicts = [c for c in conflicts if c.severity.value == "critical"]

            conflict_summary = [
                f"**{len(conflicts)} total conflicts** identified",
                f"- {len(critical_conflicts)} require immediate attention"
            ]

            if critical_conflicts:
                conflict_summary.append("\n**Critical conflicts:**")
                for c in critical_conflicts[:5]:
                    conflict_summary.append(f"- {c.description}")

            sections.append(ReportSection(
                title="Conflict Analysis",
                content="\n".join(conflict_summary)
            ))

        # Staleness section
        if self.staleness_checker:
            issues = self.staleness_checker.check_all_documents()
            expired = [i for i in issues if i.staleness_type.value == "expired"]

            staleness_summary = [
                f"**{len(issues)} staleness issues** identified",
                f"- {len(expired)} documents have expired"
            ]

            if expired:
                staleness_summary.append("\n**Expired documents:**")
                for i in expired[:5]:
                    staleness_summary.append(f"- {i.document_title}")

            sections.append(ReportSection(
                title="Staleness Analysis",
                content="\n".join(staleness_summary)
            ))

        # Gaps section
        if self.gap_analyzer:
            gaps = self.gap_analyzer.analyze_all()
            critical_gaps = [g for g in gaps if g.severity.value in ["critical", "high"]]

            gap_summary = [
                f"**{len(gaps)} coverage gaps** identified",
                f"- {len(critical_gaps)} are high priority"
            ]

            if critical_gaps:
                gap_summary.append("\n**Priority gaps:**")
                for g in critical_gaps[:5]:
                    gap_summary.append(f"- {g.topic}: {g.description}")

            sections.append(ReportSection(
                title="Coverage Analysis",
                content="\n".join(gap_summary)
            ))

        # Overall assessment
        total_issues = 0
        if self.conflict_detector:
            total_issues += len(conflicts)
        if self.staleness_checker:
            total_issues += len(issues)
        if self.gap_analyzer:
            total_issues += len(gaps)

        if total_issues == 0:
            assessment = "Documentation corpus is in **EXCELLENT** condition."
        elif total_issues < 5:
            assessment = "Documentation corpus is in **GOOD** condition with minor issues to address."
        elif total_issues < 15:
            assessment = "Documentation corpus needs **ATTENTION**. Multiple issues should be addressed."
        else:
            assessment = "Documentation corpus requires **IMMEDIATE ACTION**. Significant compliance risks identified."

        sections.append(ReportSection(
            title="Overall Assessment",
            content=assessment
        ))

        return self._render_markdown(sections)

    def to_pdf(self, markdown_content: str) -> bytes:
        """Convert Markdown report to PDF.

        Args:
            markdown_content: The Markdown report content.

        Returns:
            PDF file as bytes.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Custom styles
            styles.add(ParagraphStyle(
                name='H1Custom',
                parent=styles['Heading1'],
                spaceAfter=12,
            ))
            styles.add(ParagraphStyle(
                name='H2Custom',
                parent=styles['Heading2'],
                spaceAfter=8,
            ))

            story = []

            # Parse markdown and convert to reportlab elements
            lines = markdown_content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['H1Custom']))
                    story.append(Spacer(1, 12))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['H2Custom']))
                    story.append(Spacer(1, 8))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading3']))
                    story.append(Spacer(1, 6))
                elif line.startswith('- '):
                    # Bullet point
                    text = '\u2022 ' + line[2:]
                    text = text.replace('**', '<b>').replace('**', '</b>')
                    story.append(Paragraph(text, styles['Normal']))
                elif line.strip():
                    # Regular text
                    text = line.replace('**', '<b>').replace('**', '</b>')
                    story.append(Paragraph(text, styles['Normal']))
                else:
                    story.append(Spacer(1, 6))

            doc.build(story)
            buffer.seek(0)
            return buffer.read()

        except ImportError:
            # Fallback: return simple text as "PDF"
            return f"PDF generation requires reportlab. Install with: pip install reportlab\n\n{markdown_content}".encode()

    def _render_markdown(self, sections: list[ReportSection]) -> str:
        """Render sections as Markdown.

        Args:
            sections: List of report sections.

        Returns:
            Markdown-formatted string.
        """
        lines = []
        for section in sections:
            # Add heading
            heading_prefix = '#' * section.level
            lines.append(f"{heading_prefix} {section.title}")
            lines.append("")

            # Add content
            lines.append(section.content)
            lines.append("")

        return '\n'.join(lines)

    def _format_conflicts(self, conflicts: list) -> str:
        """Format a list of conflicts as Markdown."""
        lines = []
        for i, c in enumerate(conflicts, 1):
            lines.append(f"### {i}. {c.description}")
            lines.append("")
            lines.append(f"**Type:** {c.conflict_type.value}")
            lines.append(f"**Topic:** {c.topic}")
            lines.append("")
            lines.append(f"**Document A:** {c.location_a.document_title}")
            lines.append(f"> {c.value_a}")
            lines.append("")
            lines.append(f"**Document B:** {c.location_b.document_title}")
            lines.append(f"> {c.value_b}")
            lines.append("")

        return '\n'.join(lines)

    def _format_staleness_issues(self, issues: list) -> str:
        """Format a list of staleness issues as Markdown."""
        lines = []
        for issue in issues:
            lines.append(f"### {issue.document_title}")
            lines.append("")
            lines.append(f"**Issue:** {issue.description}")
            lines.append(f"**Severity:** {issue.severity.value}")
            if issue.recommended_action:
                lines.append(f"**Action:** {issue.recommended_action}")
            lines.append("")

        return '\n'.join(lines)

    def _format_gaps(self, gaps: list) -> str:
        """Format a list of gaps as Markdown."""
        lines = []
        for gap in gaps:
            lines.append(f"### {gap.topic}")
            lines.append("")
            lines.append(f"**Description:** {gap.description}")
            lines.append(f"**Severity:** {gap.severity.value}")
            if gap.covered_in:
                lines.append(f"**Covered in:** {', '.join(gap.covered_in)}")
            if gap.missing_from:
                lines.append(f"**Missing from:** {', '.join(gap.missing_from)}")
            lines.append("")

        return '\n'.join(lines)

    def _get_conflict_recommendations(self, conflicts: list) -> str:
        """Generate recommendations based on conflicts."""
        recs = [
            "1. **Review and resolve critical conflicts immediately** - These represent significant compliance risks.",
            "2. **Establish document ownership** - Assign clear owners to each document to prevent future conflicts.",
            "3. **Implement change management** - Require cross-document review when updating policy values.",
            "4. **Schedule regular consistency audits** - Run conflict detection monthly to catch issues early."
        ]
        return '\n'.join(recs)

    def _get_staleness_recommendations(self, issues: list) -> str:
        """Generate recommendations based on staleness issues."""
        recs = [
            "1. **Update or retire expired documents** - Expired documents should be removed or updated immediately.",
            "2. **Replace year references with current year** - Update all outdated year references.",
            "3. **Establish review cadence** - Set up quarterly or annual review schedules for all documents.",
            "4. **Add metadata** - Include 'last_reviewed' and 'next_review' dates in all documents."
        ]
        return '\n'.join(recs)

    def _get_gap_recommendations(self, gaps: list) -> str:
        """Generate recommendations based on coverage gaps."""
        recs = [
            "1. **Prioritize security-related gaps** - Any gaps related to security, access, or data protection should be addressed first.",
            "2. **Cross-reference related documents** - Ensure consistent topic coverage across related policies.",
            "3. **Create a coverage matrix** - Document which topics should appear in which documents.",
            "4. **Consolidate where appropriate** - Consider merging documents with significant overlap."
        ]
        return '\n'.join(recs)
