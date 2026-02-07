"""Action layer components for DocOps.

This module provides:
- Report generation (Markdown/PDF)
- Alert management (lifecycle, deduplication)
- Kibana dashboard generation
"""

from .report_generator import ReportGenerator, ReportSection
from .alert_manager import AlertManager, Alert
from .dashboard import KibanaDashboardGenerator, export_dashboard_to_file

__all__ = [
    # Report Generation
    "ReportGenerator",
    "ReportSection",
    # Alert Management
    "AlertManager",
    "Alert",
    # Dashboard
    "KibanaDashboardGenerator",
    "export_dashboard_to_file",
]
