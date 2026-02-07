#!/usr/bin/env python3
"""Seed demo data for DocOps Agent."""

import argparse
import logging
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import ingest_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Demo documents content
DEMO_DOCUMENTS = [
    {
        "filename": "employee_handbook_2024.md",
        "content": """# Employee Handbook 2024

## Introduction

Welcome to Acme Corporation! This handbook outlines our policies and procedures.

## Working Hours

Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday.
Remote work is permitted with manager approval.

### Flexible Hours Policy

Employees may request flexible hours arrangements. Core hours are 10:00 AM to 3:00 PM.

## Leave Policy

### Annual Leave

All full-time employees are entitled to 20 days of paid annual leave per year.
Leave must be requested at least 2 weeks in advance for periods longer than 3 days.

### Sick Leave

Employees are entitled to 10 days of paid sick leave per year.
A doctor's note is required for absences longer than 3 consecutive days.

## Code of Conduct

All employees must adhere to our code of conduct which includes:
- Treating colleagues with respect
- Maintaining confidentiality
- Avoiding conflicts of interest
- Following safety protocols

## IT Security Policy

### Password Requirements

Passwords must be:
- At least 12 characters long
- Changed every 90 days
- Unique and not reused

### Data Handling

Confidential data must be encrypted when transmitted externally.
USB drives are prohibited for storing company data.
""",
    },
    {
        "filename": "security_policy_2024.md",
        "content": """# Information Security Policy 2024

## Purpose

This policy establishes the security requirements for all Acme Corporation systems and data.

## Scope

This policy applies to all employees, contractors, and third-party users.

## Password Policy

### Requirements

All passwords must meet the following criteria:
- Minimum 14 characters (updated from previous 12-character requirement)
- Must include uppercase, lowercase, numbers, and special characters
- Changed every 60 days (updated from 90-day rotation)
- Cannot reuse last 12 passwords

### Multi-Factor Authentication

MFA is required for:
- VPN access
- Cloud services
- Administrative accounts
- Email access from new devices

## Data Classification

### Confidential

Data that could cause significant harm if disclosed:
- Customer personal information
- Financial records
- Strategic plans

### Internal

Data for internal use only:
- Internal communications
- Project documentation
- Employee directories

### Public

Data approved for public release:
- Marketing materials
- Press releases
- Public documentation

## Incident Response

Security incidents must be reported to IT Security within 1 hour of discovery.
The incident response team will assess and contain threats.

## Compliance

Violations of this policy may result in disciplinary action, up to and including termination.
""",
    },
    {
        "filename": "expense_policy_2023.md",
        "content": """# Expense Reimbursement Policy 2023

## Overview

This policy governs the reimbursement of business-related expenses.

## Eligible Expenses

### Travel

- Economy class airfare for flights under 6 hours
- Business class permitted for flights over 6 hours
- Hotel accommodations up to $200/night in standard markets
- Meals up to $75/day

### Office Supplies

- Pre-approved purchases up to $100
- Manager approval required for purchases over $100

### Client Entertainment

- Meals with clients: up to $150 per person
- Events: pre-approval required

## Submission Process

1. Submit expenses within 30 days of incurrence
2. Attach original receipts
3. Include business justification
4. Obtain manager approval

## Reimbursement Timeline

Approved expenses will be reimbursed within 14 business days.

## Policy Violations

Fraudulent expense claims will result in immediate termination and potential legal action.

**Note: This policy expires December 31, 2023 and is pending review.**
""",
    },
    {
        "filename": "remote_work_policy.md",
        "content": """# Remote Work Policy

## Eligibility

Remote work is available to employees who:
- Have completed their probationary period
- Have roles suitable for remote work
- Have manager approval

## Equipment

The company will provide:
- Laptop computer
- Monitor (upon request)
- Keyboard and mouse
- Headset for calls

Employees are responsible for:
- Reliable internet connection (minimum 50 Mbps)
- Appropriate workspace

## Working Hours

Remote employees must:
- Be available during core hours (10:00 AM - 3:00 PM)
- Attend all scheduled meetings
- Respond to communications within 2 hours during work hours

## Security Requirements

Remote workers must:
- Use company VPN for all work activities
- Ensure home network is secured with WPA3 encryption
- Never use public WiFi for work without VPN
- Keep work devices locked when not in use

## Communication

- Check Slack at least every 30 minutes during work hours
- Enable video for team meetings
- Update calendar with working hours and availability

## Performance Monitoring

Remote work continues at management discretion based on:
- Meeting deadlines
- Quality of work
- Communication responsiveness
- Team collaboration
""",
    },
    {
        "filename": "data_retention_policy.md",
        "content": """# Data Retention Policy

## Purpose

This policy defines how long various types of data must be retained and when they should be deleted.

## Retention Periods

### Financial Records

- Tax records: 7 years
- Invoices: 7 years
- Expense reports: 5 years
- Bank statements: 7 years

### Employee Records

- Active employee files: Duration of employment + 7 years
- Terminated employee files: 7 years after termination
- Payroll records: 7 years
- I-9 forms: 3 years after hire or 1 year after termination (whichever is later)

### Customer Data

- Active customer records: Duration of relationship
- Inactive customer records: 3 years after last activity
- Transaction records: 7 years
- Support tickets: 2 years after resolution

### Communications

- Business emails: 5 years
- Chat logs: 1 year
- Meeting recordings: 90 days (unless specifically retained)

## Deletion Procedures

When retention periods expire:
1. Verify no legal holds apply
2. Confirm no ongoing investigations
3. Securely delete electronic records
4. Shred physical documents

## Legal Holds

When litigation is anticipated or commenced:
- Suspend all deletion of potentially relevant records
- Notify Legal department immediately
- Document the scope of the hold

## Compliance

Failure to follow retention schedules may result in:
- Legal penalties
- Regulatory fines
- Disciplinary action
""",
    },
]


def create_demo_documents(output_dir: Path) -> list[Path]:
    """Create demo document files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for doc in DEMO_DOCUMENTS:
        file_path = output_dir / doc["filename"]
        file_path.write_text(doc["content"], encoding="utf-8")
        created_files.append(file_path)
        logger.info(f"Created: {file_path}")

    return created_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed demo data for DocOps Agent")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store demo documents (default: temp directory)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Also ingest the demo documents into Elasticsearch",
    )

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="docops_demo_"))

    logger.info(f"Creating demo documents in: {output_dir}")

    # Create demo documents
    created_files = create_demo_documents(output_dir)
    logger.info(f"Created {len(created_files)} demo documents")

    # Optionally ingest them
    if args.ingest:
        logger.info("Ingesting demo documents into Elasticsearch...")
        success_count = 0
        failure_count = 0

        for file_path in created_files:
            try:
                result = ingest_document(file_path)
                logger.info(f"Ingested: {file_path.name} ({result['chunk_count']} chunks)")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
                failure_count += 1

        logger.info(f"Ingestion complete: {success_count} succeeded, {failure_count} failed")

    logger.info("Demo data seeding complete!")
    logger.info(f"Documents location: {output_dir}")


if __name__ == "__main__":
    main()
