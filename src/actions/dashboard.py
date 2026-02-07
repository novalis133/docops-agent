"""Kibana dashboard generation for DocOps.

Generates Kibana saved objects (NDJSON format) for
document health visualization dashboards.
"""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4


class KibanaDashboardGenerator:
    """Generate Kibana dashboard definitions.

    Creates NDJSON-formatted saved objects that can be
    imported into Kibana via Stack Management > Saved Objects.
    """

    def __init__(
        self,
        documents_index: str = "docops-documents",
        chunks_index: str = "docops-chunks",
        alerts_index: str = "docops-alerts",
    ):
        """Initialize the dashboard generator.

        Args:
            documents_index: Name of documents index.
            chunks_index: Name of chunks index.
            alerts_index: Name of alerts index.
        """
        self.documents_index = documents_index
        self.chunks_index = chunks_index
        self.alerts_index = alerts_index

    def generate_dashboard(self, title: str = "DocOps Health Dashboard") -> str:
        """Generate complete dashboard with all visualizations.

        Args:
            title: Dashboard title.

        Returns:
            NDJSON string that can be imported into Kibana.
        """
        objects = []

        # Create index patterns
        objects.extend(self._create_index_patterns())

        # Create visualizations
        viz_ids = []

        # Document count metric
        doc_count_viz = self._create_metric_visualization(
            title="Total Documents",
            index_pattern=self.documents_index,
            field="_id",
            agg_type="count"
        )
        objects.append(doc_count_viz)
        viz_ids.append(doc_count_viz["id"])

        # Chunk count metric
        chunk_count_viz = self._create_metric_visualization(
            title="Total Chunks",
            index_pattern=self.chunks_index,
            field="_id",
            agg_type="count"
        )
        objects.append(chunk_count_viz)
        viz_ids.append(chunk_count_viz["id"])

        # Open alerts metric
        alerts_viz = self._create_metric_visualization(
            title="Open Alerts",
            index_pattern=self.alerts_index,
            field="_id",
            agg_type="count",
            filter_field="status",
            filter_value="open"
        )
        objects.append(alerts_viz)
        viz_ids.append(alerts_viz["id"])

        # Alerts by severity pie chart
        severity_pie = self._create_pie_chart(
            title="Alerts by Severity",
            index_pattern=self.alerts_index,
            field="severity"
        )
        objects.append(severity_pie)
        viz_ids.append(severity_pie["id"])

        # Alerts by type pie chart
        type_pie = self._create_pie_chart(
            title="Alerts by Type",
            index_pattern=self.alerts_index,
            field="alert_type"
        )
        objects.append(type_pie)
        viz_ids.append(type_pie["id"])

        # Alerts by status pie chart
        status_pie = self._create_pie_chart(
            title="Alerts by Status",
            index_pattern=self.alerts_index,
            field="status"
        )
        objects.append(status_pie)
        viz_ids.append(status_pie["id"])

        # Documents by file type bar chart
        filetype_bar = self._create_bar_chart(
            title="Documents by Type",
            index_pattern=self.documents_index,
            field="file_type"
        )
        objects.append(filetype_bar)
        viz_ids.append(filetype_bar["id"])

        # Alerts over time area chart
        alerts_timeline = self._create_timeline_chart(
            title="Alerts Over Time",
            index_pattern=self.alerts_index,
            date_field="created_at"
        )
        objects.append(alerts_timeline)
        viz_ids.append(alerts_timeline["id"])

        # Create the dashboard
        dashboard = self._create_dashboard(
            title=title,
            visualization_ids=viz_ids
        )
        objects.append(dashboard)

        # Convert to NDJSON
        return '\n'.join(json.dumps(obj) for obj in objects)

    def generate_alerts_table(self) -> str:
        """Generate a saved search for alerts table.

        Returns:
            NDJSON string for the saved search.
        """
        saved_search = {
            "id": f"docops-alerts-search-{uuid4().hex[:8]}",
            "type": "search",
            "attributes": {
                "title": "DocOps Alerts Table",
                "description": "All alerts with details",
                "columns": [
                    "severity",
                    "title",
                    "description",
                    "document_id",
                    "status",
                    "created_at"
                ],
                "sort": [["created_at", "desc"]],
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": self.alerts_index,
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": []
        }

        return json.dumps(saved_search)

    def _create_index_patterns(self) -> list[dict]:
        """Create index pattern objects."""
        patterns = []

        for index_name in [self.documents_index, self.chunks_index, self.alerts_index]:
            patterns.append({
                "id": f"index-pattern-{index_name}",
                "type": "index-pattern",
                "attributes": {
                    "title": index_name,
                    "timeFieldName": "created_at" if index_name == self.alerts_index else "indexed_at"
                },
                "references": []
            })

        return patterns

    def _create_metric_visualization(
        self,
        title: str,
        index_pattern: str,
        field: str,
        agg_type: str = "count",
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None
    ) -> dict:
        """Create a metric visualization."""
        viz_id = f"docops-metric-{uuid4().hex[:8]}"

        filters = []
        if filter_field and filter_value:
            filters.append({
                "meta": {
                    "index": index_pattern,
                    "negate": False,
                    "disabled": False,
                    "alias": None,
                    "type": "phrase",
                    "key": filter_field,
                    "params": {"query": filter_value}
                },
                "query": {"match_phrase": {filter_field: filter_value}}
            })

        vis_state = {
            "title": title,
            "type": "metric",
            "params": {
                "metric": {
                    "style": {
                        "bgColor": False,
                        "bgFill": "#000",
                        "fontSize": 60,
                        "subText": ""
                    }
                }
            },
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": agg_type,
                    "schema": "metric",
                    "params": {} if agg_type == "count" else {"field": field}
                }
            ]
        }

        return {
            "id": viz_id,
            "type": "visualization",
            "attributes": {
                "title": title,
                "visState": json.dumps(vis_state),
                "uiStateJSON": "{}",
                "description": "",
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": f"index-pattern-{index_pattern}",
                        "query": {"query": "", "language": "kuery"},
                        "filter": filters
                    })
                }
            },
            "references": [
                {
                    "id": f"index-pattern-{index_pattern}",
                    "name": "kibanaSavedObjectMeta.searchSourceJSON.index",
                    "type": "index-pattern"
                }
            ]
        }

    def _create_pie_chart(
        self,
        title: str,
        index_pattern: str,
        field: str
    ) -> dict:
        """Create a pie chart visualization."""
        viz_id = f"docops-pie-{uuid4().hex[:8]}"

        vis_state = {
            "title": title,
            "type": "pie",
            "params": {
                "type": "pie",
                "addTooltip": True,
                "addLegend": True,
                "legendPosition": "right",
                "isDonut": True
            },
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric",
                    "params": {}
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "terms",
                    "schema": "segment",
                    "params": {
                        "field": field,
                        "size": 10,
                        "order": "desc",
                        "orderBy": "1"
                    }
                }
            ]
        }

        return {
            "id": viz_id,
            "type": "visualization",
            "attributes": {
                "title": title,
                "visState": json.dumps(vis_state),
                "uiStateJSON": "{}",
                "description": "",
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": f"index-pattern-{index_pattern}",
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": [
                {
                    "id": f"index-pattern-{index_pattern}",
                    "name": "kibanaSavedObjectMeta.searchSourceJSON.index",
                    "type": "index-pattern"
                }
            ]
        }

    def _create_bar_chart(
        self,
        title: str,
        index_pattern: str,
        field: str
    ) -> dict:
        """Create a horizontal bar chart visualization."""
        viz_id = f"docops-bar-{uuid4().hex[:8]}"

        vis_state = {
            "title": title,
            "type": "horizontal_bar",
            "params": {
                "type": "horizontal_bar",
                "addTooltip": True,
                "addLegend": True,
                "legendPosition": "right"
            },
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric",
                    "params": {}
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "terms",
                    "schema": "segment",
                    "params": {
                        "field": field,
                        "size": 10,
                        "order": "desc",
                        "orderBy": "1"
                    }
                }
            ]
        }

        return {
            "id": viz_id,
            "type": "visualization",
            "attributes": {
                "title": title,
                "visState": json.dumps(vis_state),
                "uiStateJSON": "{}",
                "description": "",
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": f"index-pattern-{index_pattern}",
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": [
                {
                    "id": f"index-pattern-{index_pattern}",
                    "name": "kibanaSavedObjectMeta.searchSourceJSON.index",
                    "type": "index-pattern"
                }
            ]
        }

    def _create_timeline_chart(
        self,
        title: str,
        index_pattern: str,
        date_field: str
    ) -> dict:
        """Create a timeline/area chart visualization."""
        viz_id = f"docops-timeline-{uuid4().hex[:8]}"

        vis_state = {
            "title": title,
            "type": "area",
            "params": {
                "type": "area",
                "addTooltip": True,
                "addLegend": True,
                "legendPosition": "right"
            },
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric",
                    "params": {}
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "date_histogram",
                    "schema": "segment",
                    "params": {
                        "field": date_field,
                        "interval": "auto",
                        "min_doc_count": 1
                    }
                }
            ]
        }

        return {
            "id": viz_id,
            "type": "visualization",
            "attributes": {
                "title": title,
                "visState": json.dumps(vis_state),
                "uiStateJSON": "{}",
                "description": "",
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": f"index-pattern-{index_pattern}",
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": [
                {
                    "id": f"index-pattern-{index_pattern}",
                    "name": "kibanaSavedObjectMeta.searchSourceJSON.index",
                    "type": "index-pattern"
                }
            ]
        }

    def _create_dashboard(
        self,
        title: str,
        visualization_ids: list[str]
    ) -> dict:
        """Create the main dashboard object."""
        dashboard_id = f"docops-dashboard-{uuid4().hex[:8]}"

        # Create panel layout
        panels = []
        grid_data = []
        col = 0
        row = 0

        for i, viz_id in enumerate(visualization_ids):
            # 3 panels per row, 16 columns wide each
            panel_w = 16
            panel_h = 12

            if col + panel_w > 48:
                col = 0
                row += panel_h

            panels.append({
                "version": "8.0.0",
                "type": "visualization",
                "gridData": {
                    "x": col,
                    "y": row,
                    "w": panel_w,
                    "h": panel_h,
                    "i": str(i)
                },
                "panelIndex": str(i),
                "embeddableConfig": {},
                "panelRefName": f"panel_{i}"
            })

            col += panel_w

        references = []
        for i, viz_id in enumerate(visualization_ids):
            references.append({
                "id": viz_id,
                "name": f"panel_{i}",
                "type": "visualization"
            })

        return {
            "id": dashboard_id,
            "type": "dashboard",
            "attributes": {
                "title": title,
                "description": "DocOps Agent health monitoring dashboard",
                "panelsJSON": json.dumps(panels),
                "optionsJSON": json.dumps({
                    "useMargins": True,
                    "hidePanelTitles": False
                }),
                "timeRestore": False,
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": references
        }


def export_dashboard_to_file(output_path: str) -> None:
    """Export the dashboard to a NDJSON file.

    Args:
        output_path: Path for the output file.
    """
    generator = KibanaDashboardGenerator()
    dashboard_ndjson = generator.generate_dashboard()

    with open(output_path, 'w') as f:
        f.write(dashboard_ndjson)

    print(f"Dashboard exported to: {output_path}")
    print("Import via Kibana > Stack Management > Saved Objects > Import")
