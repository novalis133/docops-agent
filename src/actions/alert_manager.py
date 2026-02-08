"""Alert management for DocOps.

Provides deduplication, acknowledgment, resolution,
and severity-based routing for document alerts.
Includes Slack webhook integration for notifications.
"""

import hashlib
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Slack webhook URL (set via environment variable)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


@dataclass
class Alert:
    """Represents a document alert."""
    id: str
    document_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    metadata: Optional[dict] = None
    # Resolution lifecycle fields
    resolution_status: str = "open"  # open, in_progress, resolved, wont_fix
    resolution_notes: Optional[str] = None
    verification_status: str = "pending"  # pending, verified, failed
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    remediation: Optional[dict] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Alert":
        """Create an Alert from a dictionary."""
        return cls(
            id=data.get("id", ""),
            document_id=data.get("document_id", ""),
            alert_type=data.get("alert_type", ""),
            severity=data.get("severity", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=data.get("status", "open"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) and data.get("updated_at") else None,
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if isinstance(data.get("resolved_at"), str) and data.get("resolved_at") else None,
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if isinstance(data.get("acknowledged_at"), str) and data.get("acknowledged_at") else None,
            acknowledged_by=data.get("acknowledged_by"),
            resolved_by=data.get("resolved_by"),
            metadata=data.get("metadata"),
            # Resolution lifecycle fields
            resolution_status=data.get("resolution_status", "open"),
            resolution_notes=data.get("resolution_notes"),
            verification_status=data.get("verification_status", "pending"),
            verified_at=datetime.fromisoformat(data["verified_at"]) if isinstance(data.get("verified_at"), str) and data.get("verified_at") else None,
            verified_by=data.get("verified_by"),
            remediation=data.get("remediation"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "metadata": self.metadata,
            # Resolution lifecycle fields
            "resolution_status": self.resolution_status,
            "resolution_notes": self.resolution_notes,
            "verification_status": self.verification_status,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verified_by": self.verified_by,
            "remediation": self.remediation,
        }


class AlertManager:
    """Manage document alerts with deduplication and lifecycle tracking.

    Features:
    - Deduplication based on document_id + alert_type + description hash
    - Acknowledgment workflow
    - Resolution with audit trail
    - Severity-based filtering
    - Bulk operations
    """

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        alerts_index: str = "docops-alerts",
    ):
        """Initialize the alert manager.

        Args:
            es_client: Existing Elasticsearch client.
            host: ES host.
            port: ES port.
            scheme: http or https.
            alerts_index: Name of alerts index.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.alerts_index = alerts_index

    def create_alert(
        self,
        document_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        metadata: Optional[dict] = None,
        deduplicate: bool = True
    ) -> Optional[str]:
        """Create a new alert with optional deduplication.

        Args:
            document_id: ID of the affected document.
            alert_type: Type of alert (e.g., "conflict", "staleness", "gap").
            severity: Alert severity (critical, high, medium, low).
            title: Short alert title.
            description: Detailed description.
            metadata: Additional metadata.
            deduplicate: Whether to skip if duplicate exists.

        Returns:
            Alert ID if created, None if deduplicated.
        """
        # Generate deduplication key
        dedup_key = self._generate_dedup_key(document_id, alert_type, description)

        if deduplicate:
            # Check for existing alert with same dedup key
            existing = self._find_by_dedup_key(dedup_key)
            if existing:
                return None  # Duplicate - skip creation

        # Create alert
        now = datetime.utcnow()
        alert_id = f"alert-{now.strftime('%Y%m%d%H%M%S')}-{dedup_key[:8]}"

        alert_doc = {
            "document_id": document_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "status": "open",
            "created_at": now.isoformat(),
            "dedup_key": dedup_key,
            "metadata": metadata or {},
        }

        self.es.index(
            index=self.alerts_index,
            id=alert_id,
            document=alert_doc,
            refresh=True
        )

        # Send Slack notification for critical/high severity alerts
        if severity in ("critical", "high"):
            self._send_slack_notification(alert_id, severity, title, description)

        return alert_id

    def _send_slack_notification(
        self,
        alert_id: str,
        severity: str,
        title: str,
        description: str
    ) -> bool:
        """Send alert notification to Slack webhook.

        Args:
            alert_id: The alert ID.
            severity: Alert severity level.
            title: Alert title.
            description: Alert description.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not SLACK_WEBHOOK_URL:
            logger.debug("Slack webhook not configured, skipping notification")
            return False

        try:
            import httpx

            # Color based on severity
            color = "#DC3545" if severity == "critical" else "#FD7E14"
            emoji = ":rotating_light:" if severity == "critical" else ":warning:"

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": f"{emoji} DocOps Alert: {title}",
                                    "emoji": True
                                }
                            },
                            {
                                "type": "section",
                                "fields": [
                                    {
                                        "type": "mrkdwn",
                                        "text": f"*Severity:*\n{severity.upper()}"
                                    },
                                    {
                                        "type": "mrkdwn",
                                        "text": f"*Alert ID:*\n{alert_id[:20]}..."
                                    }
                                ]
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Description:*\n{description[:500]}"
                                }
                            },
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": f"Sent by DocOps Agent | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

            response = httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Slack notification sent for alert {alert_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")
            return False

    def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: ID of the alert.
            acknowledged_by: User/system acknowledging.

        Returns:
            True if acknowledged, False if not found or already acknowledged.
        """
        try:
            alert = self.es.get(index=self.alerts_index, id=alert_id)
            source = alert["_source"]

            if source.get("acknowledged_at"):
                return False  # Already acknowledged

            now = datetime.utcnow()
            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc={
                    "status": "acknowledged",
                    "acknowledged_at": now.isoformat(),
                    "acknowledged_by": acknowledged_by,
                    "updated_at": now.isoformat(),
                },
                refresh=True
            )
            return True

        except Exception:
            return False

    def resolve(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Resolve an alert.

        Args:
            alert_id: ID of the alert.
            resolved_by: User/system resolving.
            resolution_note: Optional note about resolution.

        Returns:
            True if resolved, False if not found or already resolved.
        """
        try:
            alert = self.es.get(index=self.alerts_index, id=alert_id)
            source = alert["_source"]

            if source.get("status") == "resolved":
                return False  # Already resolved

            now = datetime.utcnow()
            update_doc = {
                "status": "resolved",
                "resolved_at": now.isoformat(),
                "resolved_by": resolved_by,
                "updated_at": now.isoformat(),
            }

            if resolution_note:
                metadata = source.get("metadata", {})
                metadata["resolution_note"] = resolution_note
                update_doc["metadata"] = metadata

            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc=update_doc,
                refresh=True
            )
            return True

        except Exception:
            return False

    def reopen(self, alert_id: str, reason: Optional[str] = None) -> bool:
        """Reopen a resolved alert.

        Args:
            alert_id: ID of the alert.
            reason: Optional reason for reopening.

        Returns:
            True if reopened, False if not found or not resolved.
        """
        try:
            alert = self.es.get(index=self.alerts_index, id=alert_id)
            source = alert["_source"]

            if source.get("status") != "resolved":
                return False  # Not resolved

            now = datetime.utcnow()
            update_doc = {
                "status": "open",
                "resolved_at": None,
                "resolved_by": None,
                "updated_at": now.isoformat(),
            }

            if reason:
                metadata = source.get("metadata", {})
                metadata["reopen_reason"] = reason
                metadata["reopened_at"] = now.isoformat()
                update_doc["metadata"] = metadata

            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc=update_doc,
                refresh=True
            )
            return True

        except Exception:
            return False

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID.

        Args:
            alert_id: ID of the alert.

        Returns:
            Alert object or None if not found.
        """
        try:
            result = self.es.get(index=self.alerts_index, id=alert_id)
            source = result["_source"]
            source["id"] = result["_id"]
            return Alert.from_dict(source)
        except Exception:
            return None

    def get_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        document_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        top_k: int = 100
    ) -> list[Alert]:
        """Get alerts with optional filtering.

        Args:
            status: Filter by status (open, acknowledged, resolved).
            severity: Filter by severity (critical, high, medium, low).
            document_id: Filter by document ID.
            alert_type: Filter by alert type.
            top_k: Maximum number of alerts to return.

        Returns:
            List of Alert objects.
        """
        must_clauses = []

        if status:
            must_clauses.append({"term": {"status": status}})
        if severity:
            must_clauses.append({"term": {"severity": severity}})
        if document_id:
            must_clauses.append({"term": {"document_id": document_id}})
        if alert_type:
            must_clauses.append({"term": {"alert_type": alert_type}})

        query = {"match_all": {}}
        if must_clauses:
            query = {"bool": {"must": must_clauses}}

        try:
            result = self.es.search(
                index=self.alerts_index,
                query=query,
                size=top_k,
                sort=[
                    {"severity": {"order": "asc", "unmapped_type": "keyword"}},
                    {"created_at": {"order": "desc"}}
                ]
            )

            alerts = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                source["id"] = hit["_id"]
                alerts.append(Alert.from_dict(source))

            return alerts

        except Exception:
            return []

    def get_open_alerts(self, severity: Optional[str] = None) -> list[Alert]:
        """Get all open alerts.

        Args:
            severity: Optional severity filter.

        Returns:
            List of open alerts.
        """
        return self.get_alerts(status="open", severity=severity)

    def get_alerts_by_document(self, document_id: str) -> list[Alert]:
        """Get all alerts for a specific document.

        Args:
            document_id: Document ID.

        Returns:
            List of alerts for the document.
        """
        return self.get_alerts(document_id=document_id)

    def bulk_resolve(
        self,
        alert_ids: list[str],
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> dict:
        """Resolve multiple alerts at once.

        Args:
            alert_ids: List of alert IDs.
            resolved_by: User/system resolving.
            resolution_note: Optional note about resolution.

        Returns:
            Dictionary with success/failure counts.
        """
        results = {"resolved": 0, "failed": 0, "already_resolved": 0}

        for alert_id in alert_ids:
            if self.resolve(alert_id, resolved_by, resolution_note):
                results["resolved"] += 1
            else:
                alert = self.get_alert(alert_id)
                if alert and alert.status == "resolved":
                    results["already_resolved"] += 1
                else:
                    results["failed"] += 1

        return results

    def get_alert_counts(self) -> dict:
        """Get alert counts by status and severity.

        Returns:
            Dictionary with count breakdowns.
        """
        try:
            result = self.es.search(
                index=self.alerts_index,
                size=0,
                aggs={
                    "by_status": {
                        "terms": {"field": "status"}
                    },
                    "by_severity": {
                        "terms": {"field": "severity"}
                    },
                    "open_by_severity": {
                        "filter": {"term": {"status": "open"}},
                        "aggs": {
                            "severity": {
                                "terms": {"field": "severity"}
                            }
                        }
                    }
                }
            )

            aggs = result.get("aggregations", {})

            counts = {
                "total": result["hits"]["total"]["value"],
                "by_status": {},
                "by_severity": {},
                "open_by_severity": {}
            }

            for bucket in aggs.get("by_status", {}).get("buckets", []):
                counts["by_status"][bucket["key"]] = bucket["doc_count"]

            for bucket in aggs.get("by_severity", {}).get("buckets", []):
                counts["by_severity"][bucket["key"]] = bucket["doc_count"]

            for bucket in aggs.get("open_by_severity", {}).get("severity", {}).get("buckets", []):
                counts["open_by_severity"][bucket["key"]] = bucket["doc_count"]

            return counts

        except Exception:
            return {"total": 0, "by_status": {}, "by_severity": {}, "open_by_severity": {}}

    def cleanup_resolved(self, older_than_days: int = 90) -> int:
        """Delete resolved alerts older than specified days.

        Args:
            older_than_days: Delete alerts resolved more than this many days ago.

        Returns:
            Number of alerts deleted.
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)

        try:
            result = self.es.delete_by_query(
                index=self.alerts_index,
                query={
                    "bool": {
                        "must": [
                            {"term": {"status": "resolved"}},
                            {"range": {"resolved_at": {"lt": cutoff.isoformat()}}}
                        ]
                    }
                },
                refresh=True
            )

            return result.get("deleted", 0)

        except Exception:
            return 0

    # =========================================================================
    # Resolution Lifecycle Methods
    # =========================================================================

    def update_resolution_status(
        self,
        alert_id: str,
        resolution_status: str,
        resolution_notes: Optional[str] = None,
        resolved_by: Optional[str] = None
    ) -> bool:
        """Update the resolution status of an alert.

        Args:
            alert_id: ID of the alert.
            resolution_status: New status (open, in_progress, resolved, wont_fix).
            resolution_notes: Optional notes about the resolution.
            resolved_by: User/system resolving.

        Returns:
            True if updated, False if not found.
        """
        valid_statuses = {"open", "in_progress", "resolved", "wont_fix"}
        if resolution_status not in valid_statuses:
            logger.warning(f"Invalid resolution status: {resolution_status}")
            return False

        try:
            now = datetime.utcnow()
            update_doc = {
                "resolution_status": resolution_status,
                "updated_at": now.isoformat(),
            }

            if resolution_notes:
                update_doc["resolution_notes"] = resolution_notes

            if resolution_status in ("resolved", "wont_fix"):
                update_doc["status"] = "resolved"
                update_doc["resolved_at"] = now.isoformat()
                if resolved_by:
                    update_doc["resolved_by"] = resolved_by

            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc=update_doc,
                refresh=True
            )
            logger.info(f"Updated alert {alert_id} resolution status to {resolution_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update resolution status: {e}")
            return False

    def set_remediation_suggestion(
        self,
        alert_id: str,
        remediation: dict
    ) -> bool:
        """Set the remediation suggestion for an alert.

        Args:
            alert_id: ID of the alert.
            remediation: Remediation suggestion dictionary.

        Returns:
            True if set, False if not found.
        """
        try:
            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc={
                    "remediation": remediation,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                refresh=True
            )
            logger.info(f"Set remediation suggestion for alert {alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set remediation: {e}")
            return False

    def update_verification_status(
        self,
        alert_id: str,
        verification_status: str,
        verified_by: Optional[str] = None
    ) -> bool:
        """Update the verification status of an alert.

        Args:
            alert_id: ID of the alert.
            verification_status: New status (pending, verified, failed).
            verified_by: User/system that verified.

        Returns:
            True if updated, False if not found.
        """
        valid_statuses = {"pending", "verified", "failed"}
        if verification_status not in valid_statuses:
            logger.warning(f"Invalid verification status: {verification_status}")
            return False

        try:
            now = datetime.utcnow()
            update_doc = {
                "verification_status": verification_status,
                "updated_at": now.isoformat(),
            }

            if verification_status in ("verified", "failed"):
                update_doc["verified_at"] = now.isoformat()
                if verified_by:
                    update_doc["verified_by"] = verified_by

            # If verification failed, reopen the alert
            if verification_status == "failed":
                update_doc["resolution_status"] = "open"
                update_doc["status"] = "open"
                update_doc["resolved_at"] = None

            self.es.update(
                index=self.alerts_index,
                id=alert_id,
                doc=update_doc,
                refresh=True
            )
            logger.info(f"Updated alert {alert_id} verification status to {verification_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update verification status: {e}")
            return False

    def get_alerts_by_lifecycle(self) -> dict:
        """Get alerts grouped by resolution and verification status.

        Returns:
            Dictionary with alerts grouped by lifecycle stage.
        """
        try:
            result = self.es.search(
                index=self.alerts_index,
                size=0,
                aggs={
                    "by_resolution_status": {
                        "terms": {"field": "resolution_status", "missing": "open"}
                    },
                    "by_verification_status": {
                        "terms": {"field": "verification_status", "missing": "pending"}
                    },
                    "lifecycle_stages": {
                        "filters": {
                            "filters": {
                                "open": {"term": {"resolution_status": "open"}},
                                "in_progress": {"term": {"resolution_status": "in_progress"}},
                                "resolved_pending_verification": {
                                    "bool": {
                                        "must": [
                                            {"term": {"resolution_status": "resolved"}},
                                            {"term": {"verification_status": "pending"}}
                                        ]
                                    }
                                },
                                "verified": {"term": {"verification_status": "verified"}},
                                "verification_failed": {"term": {"verification_status": "failed"}},
                                "wont_fix": {"term": {"resolution_status": "wont_fix"}}
                            }
                        }
                    }
                }
            )

            aggs = result.get("aggregations", {})

            lifecycle = {
                "total": result["hits"]["total"]["value"],
                "by_resolution_status": {},
                "by_verification_status": {},
                "lifecycle_stages": {}
            }

            for bucket in aggs.get("by_resolution_status", {}).get("buckets", []):
                lifecycle["by_resolution_status"][bucket["key"]] = bucket["doc_count"]

            for bucket in aggs.get("by_verification_status", {}).get("buckets", []):
                lifecycle["by_verification_status"][bucket["key"]] = bucket["doc_count"]

            stages = aggs.get("lifecycle_stages", {}).get("buckets", {})
            for stage, data in stages.items():
                lifecycle["lifecycle_stages"][stage] = data.get("doc_count", 0)

            return lifecycle

        except Exception as e:
            logger.error(f"Failed to get lifecycle stats: {e}")
            return {"total": 0, "by_resolution_status": {}, "by_verification_status": {}, "lifecycle_stages": {}}

    def get_pending_verifications(self, top_k: int = 50) -> list[Alert]:
        """Get alerts that are resolved but pending verification.

        Args:
            top_k: Maximum number of alerts to return.

        Returns:
            List of alerts pending verification.
        """
        try:
            result = self.es.search(
                index=self.alerts_index,
                query={
                    "bool": {
                        "must": [
                            {"term": {"resolution_status": "resolved"}},
                            {"term": {"verification_status": "pending"}}
                        ]
                    }
                },
                size=top_k,
                sort=[{"resolved_at": {"order": "asc"}}]
            )

            alerts = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                source["id"] = hit["_id"]
                alerts.append(Alert.from_dict(source))

            return alerts

        except Exception as e:
            logger.error(f"Failed to get pending verifications: {e}")
            return []

    def _generate_dedup_key(
        self,
        document_id: str,
        alert_type: str,
        description: str
    ) -> str:
        """Generate a deduplication key for an alert.

        Args:
            document_id: Document ID.
            alert_type: Alert type.
            description: Alert description.

        Returns:
            Hash-based deduplication key.
        """
        content = f"{document_id}:{alert_type}:{description}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _find_by_dedup_key(self, dedup_key: str) -> Optional[str]:
        """Find an open alert by deduplication key.

        Args:
            dedup_key: The deduplication key.

        Returns:
            Alert ID if found, None otherwise.
        """
        try:
            result = self.es.search(
                index=self.alerts_index,
                query={
                    "bool": {
                        "must": [
                            {"term": {"dedup_key": dedup_key}},
                            {"term": {"status": "open"}}
                        ]
                    }
                },
                size=1
            )

            if result["hits"]["total"]["value"] > 0:
                return result["hits"]["hits"][0]["_id"]
            return None

        except Exception:
            return None
