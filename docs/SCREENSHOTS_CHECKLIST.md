# Screenshots Checklist for DevPost Submission

This document provides a checklist and guidelines for capturing screenshots for the Elasticsearch Agent Builder Hackathon submission.

---

## DevPost Image Requirements

- **First image = Thumbnail:** This appears in search results and browsing. Make it count!
- **Recommended dimensions:** 1280x720 (16:9) or 1920x1080
- **File format:** PNG preferred (better quality), JPG acceptable
- **File size:** Keep under 5MB per image
- **GIFs:** Can show interactions (agent typing, step-by-step progression)

---

## Required Screenshots

### 1. Dashboard Overview
**File:** `docs/screenshots/01-dashboard-overview.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Full dashboard with health score, document count, chunk count, alert counts by severity, pie/bar charts |
| **Why it matters** | Demonstrates production-ready UI, shows "at a glance" value |
| **Key elements** | Health score prominently visible, measurable impact metrics ("40 hrs → 2 min"), colorful visualizations |
| **Capture tips** | Use demo data with realistic numbers, ensure all widgets are populated |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 2. Agent Chat - Multi-Step Reasoning
**File:** `docs/screenshots/02-agent-multi-step.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Agent chat with visible step trace showing 3+ tools being chained |
| **Why it matters** | **CRITICAL for judging** - proves multi-step reasoning, not just search |
| **Key elements** | User query visible, step trace with tool names, final synthesized response |
| **Suggested query** | "Are there any conflicts about password requirements?" |
| **Capture tips** | Ensure step trace shows: search_documents → run_consistency_check → create_alert |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 3. Agent Chat - Tool Execution Details
**File:** `docs/screenshots/03-agent-tool-details.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Expanded view of tool execution with parameters and results |
| **Why it matters** | Shows technical depth, Elasticsearch integration |
| **Key elements** | Tool name, input parameters, ES query details, returned results |
| **Capture tips** | If expandable sections exist, expand them |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 4. Conflict Viewer
**File:** `docs/screenshots/04-conflict-viewer.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Side-by-side conflict comparison with document excerpts |
| **Why it matters** | Visual "wow factor" - makes abstract problem tangible |
| **Key elements** | Two document sections shown, conflicting values highlighted, severity badge, AI resolution suggestion |
| **Capture tips** | Choose a clear conflict (e.g., "12 chars" vs "14 chars"), ensure both sides are readable |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 5. Search Results with Hybrid Scores
**File:** `docs/screenshots/05-search-results.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Search results showing both BM25 and vector similarity scores |
| **Why it matters** | Demonstrates Elasticsearch hybrid search implementation |
| **Key elements** | Query input, result list with scores, source documents, relevance indicators |
| **Capture tips** | Use a query that returns mixed results (some exact matches, some semantic) |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 6. Generated Report
**File:** `docs/screenshots/06-generated-report.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Compliance report with findings, recommendations, export options |
| **Why it matters** | Shows actionable output, not just analysis |
| **Key elements** | Report title, summary stats, findings list, download buttons (Markdown, Excel, PDF) |
| **Capture tips** | Generate a report with real findings, show export options |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 7. Workflow Execution
**File:** `docs/screenshots/07-workflow-execution.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Pre-built workflow (e.g., "compliance_audit") running with progress |
| **Why it matters** | Shows automation capability, "one-click" value proposition |
| **Key elements** | Workflow name, execution status, results summary |
| **Capture tips** | Capture during execution if possible to show progress |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 8. Welcome Screen (Thumbnail Candidate)
**File:** `docs/screenshots/08-welcome-screen.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Clean welcome screen with product name, tagline, key features |
| **Why it matters** | **Best thumbnail candidate** - clean, branded, inviting |
| **Key elements** | DocOps Agent title, "Get Started" button, feature highlights |
| **Capture tips** | Fresh browser, no clutter, full width |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

### 9. Social Proof (Required for 10% criteria)
**File:** `docs/screenshots/09-social-proof.png`

| Aspect | Details |
|--------|---------|
| **What to show** | Screenshot of Twitter/X post with @elastic_devs tag |
| **Why it matters** | Required for Social Sharing criteria (10% of score) |
| **Key elements** | Post visible, @elastic_devs tag, project name/description |
| **Capture tips** | Post before deadline, capture shortly after for engagement visibility |

**Status:** [ ] Not captured  [ ] Captured  [ ] Approved

---

## Directory Structure

```
docs/
└── screenshots/
    ├── 01-dashboard-overview.png
    ├── 02-agent-multi-step.png
    ├── 03-agent-tool-details.png
    ├── 04-conflict-viewer.png
    ├── 05-search-results.png
    ├── 06-generated-report.png
    ├── 07-workflow-execution.png
    ├── 08-welcome-screen.png       ← Best thumbnail candidate
    └── 09-social-proof.png
```

---

## GIF Recommendations

Consider creating GIFs for these interactions:

1. **Agent Reasoning Flow** (`agent-reasoning.gif`)
   - User types query → Agent shows step trace → Final response
   - Duration: 10-15 seconds
   - Tools: ScreenToGif, LICEcap, or Gifski

2. **Document Upload** (`document-upload.gif`)
   - Drag file → Upload progress → Success message → Appears in corpus
   - Duration: 5-8 seconds

3. **Conflict Discovery** (`conflict-discovery.gif`)
   - Run workflow → Conflicts appear → Click to view details
   - Duration: 8-12 seconds

---

## Capture Checklist

Before capturing screenshots:

- [ ] Elasticsearch running with demo data loaded
- [ ] Frontend running at localhost:8501
- [ ] Browser at 100% zoom (no scaling)
- [ ] Browser dev tools closed
- [ ] No personal/sensitive data visible
- [ ] Consistent browser window size (1280x720 recommended)
- [ ] Light mode preferred (better visibility)

---

## Thumbnail Selection Priority

1. **Welcome Screen** - Clean, branded, professional
2. **Dashboard** - Shows product value immediately
3. **Agent Multi-Step** - Demonstrates core differentiator
4. **Conflict Viewer** - Visual "wow factor"

---

## Post-Capture Checklist

- [ ] All screenshots captured at consistent dimensions
- [ ] File sizes optimized (under 5MB each)
- [ ] Filenames match convention (01-, 02-, etc.)
- [ ] No sensitive data visible
- [ ] Screenshots reviewed for clarity
- [ ] GIFs tested for smooth playback
- [ ] Uploaded to DevPost in correct order
