"""
DocuMind — Bulletproof Flowchart Generator
Guaranteed valid Mermaid output for any document.
"""

import os
import re
import streamlit.components.v1 as components
from pathlib import Path

# Load env manually
_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV.exists():
    with open(_ENV) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


# ══ Safe label cleaner ═════════════════════════════════════════════════════════

def _safe(text: str, max_len: int = 42) -> str:
    """Strip every character that breaks Mermaid."""
    if not text:
        return "Step"
    text = text.replace('"','').replace("'",'').replace('`','')
    text = re.sub(r'[<>{}\[\]\\|()&@#$%^*+=]', '', text)
    text = text.replace(':', ' -').replace(';', ',').replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove file extension patterns
    text = re.sub(r'\.\w{2,5}$', '', text).strip()
    if len(text) > max_len:
        text = text[:max_len-3] + '...'
    return text or "Step"

# ══ Mermaid renderer ═══════════════════════════════════════════════════════════

def render_mermaid(mermaid_code: str):
    """Render Mermaid diagram with guaranteed-safe code."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
<style>
  body {{ margin:0; background:#0f172a; padding:16px; box-sizing:border-box; }}
  .mermaid {{ width:100%; }}
  .mermaid svg {{ max-width:100% !important; }}
</style>
</head>
<body>
<div class="mermaid">
{mermaid_code}
</div>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'dark',
    securityLevel: 'loose',
    flowchart: {{ htmlLabels: false, curve: 'basis' }},
    mindmap: {{ padding: 20 }},
    themeVariables: {{
      primaryColor: '#6366f1',
      primaryTextColor: '#f1f5f9',
      primaryBorderColor: '#818cf8',
      lineColor: '#94a3b8',
      secondaryColor: '#1e293b',
      tertiaryColor: '#0f172a',
      background: '#0f172a',
      mainBkg: '#1e293b'
    }}
  }});
</script>
</body>
</html>"""
    components.html(html, height=600, scrolling=True)


# ══ AI-powered generation ══════════════════════════════════════════════════════

def _generate_with_ai(text: str, diagram_type: str) -> str | None:
    """Use Gemini to generate valid Mermaid code."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=key)

        if "research" in diagram_type.lower():
            task = "a flowchart showing the research paper structure: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion"
        elif "mind" in diagram_type.lower() or "concept" in diagram_type.lower():
            task = "a flowchart (NOT mindmap) showing the top 6-8 key concepts and how they connect"
        else:
            task = "a flowchart showing the main process steps in order"

        prompt = f"""Analyze this document and generate {task}.

DOCUMENT:
{text[:2500]}

OUTPUT RULES — follow EXACTLY:
1. Output ONLY the Mermaid code. No explanation. No markdown fences. No ``` symbols.
2. Start with exactly: flowchart TD
3. Node IDs: use only A, B, C, D, E, F, G, H, I, J (single or double letter)
4. Labels: short (under 35 chars), wrapped in double quotes, NO special chars
5. Forbidden in labels: ( ) [ ] {{ }} < > | \\ : ; & '
6. Connections: use -->
7. Maximum 8 nodes total
8. No style or classDef lines

CORRECT EXAMPLE:
flowchart TD
    A["Image Acquisition"]
    B["Face Detection"]
    C["Feature Extraction"]
    D["CNN Classification"]
    E["Drowsiness Score"]
    F["Safety Alert System"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F

Generate the diagram now:"""

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        raw = (resp.text or "").strip()

        # Strip any markdown fences
        raw = re.sub(r'```(?:mermaid)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw)
        raw = raw.strip()

        # Validate — must start with flowchart or graph
        if raw and (raw.startswith("flowchart") or raw.startswith("graph")):
            # Final safety pass
            return _validate_and_fix(raw)

        return None

    except Exception as e:
        print(f"[Flowchart] AI error: {e}")
        return None


def _validate_and_fix(code: str) -> str:
    """
    Post-process AI-generated Mermaid to fix any remaining issues.
    """
    lines  = code.strip().split("\n")
    output = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Directive line — keep as-is
        if s.startswith(("flowchart", "graph", "mindmap")):
            output.append(s)
            continue

        # Arrow line — keep but clean
        if "-->" in s or "---" in s:
            # Remove arrow labels like |text|
            s = re.sub(r'\|[^|]*\|', '', s)
            output.append("    " + s.strip())
            continue

        # Node line — sanitize label
        # Match: ID["label"] or ID[label] or ID("label") or ID([label])
        m = re.match(r'^(\w+)([\[\(]{1,2})"?([^"\]\)]*)"?([\]\)]{1,2})$', s)
        if m:
            nid    = m.group(1)
            ob     = m.group(2)
            label  = _safe(m.group(3))
            cb     = m.group(4)
            output.append(f'    {nid}{ob}"{label}"{cb}')
            continue

        # style/classDef lines — skip (can cause issues)
        if s.startswith(("style ", "classDef ", "class ")):
            continue

        output.append("    " + s)

    return "\n".join(output)


# ══ Smart local extraction ═════════════════════════════════════════════════════

def _extract_steps_smart(text: str) -> list[str]:
    """
    Extract meaningful steps from any document.
    Returns clean strings safe for Mermaid labels.
    """
    steps = []

    # Strategy 1: numbered lists
    matches = re.findall(
        r'(?:^|\n)\s*\d+[\.\)]\s+([A-Z][^\n]{15,100})',
        text, re.MULTILINE
    )
    for m in matches:
        clean = _safe(m.strip(), 40)
        if clean and len(clean.split()) >= 2:
            steps.append(clean)
    if len(steps) >= 3:
        return steps[:8]

    # Strategy 2: bullet points
    matches = re.findall(
        r'(?:^|\n)\s*[•\-\*]\s+([A-Z][^\n]{15,100})',
        text, re.MULTILINE
    )
    for m in matches:
        clean = _safe(m.strip(), 40)
        if clean and len(clean.split()) >= 2:
            steps.append(clean)
    if len(steps) >= 3:
        return steps[:8]

    # Strategy 3: action sentences
    actions = [
        'collect', 'capture', 'preprocess', 'resize', 'normalize',
        'extract', 'detect', 'classify', 'train', 'evaluate', 'apply',
        'generate', 'compute', 'implement', 'process', 'analyse',
        'install', 'configure', 'deploy', 'test', 'validate'
    ]
    sentences = re.split(r'[.!?]\s+', text)
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if 3 <= len(words) <= 18:
            if any(a in sent.lower() for a in actions):
                steps.append(_safe(sent, 40))
    if len(steps) >= 3:
        return steps[:8]

    # Strategy 4: capitalized phrases
    phrases = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b', text)
    seen = set()
    for p in phrases:
        if p not in seen and len(p) > 8:
            seen.add(p)
            steps.append(_safe(p, 40))
        if len(steps) >= 8:
            break

    # Final fallback
    return steps[:8] if len(steps) >= 3 else [
        "Data Collection",
        "Preprocessing",
        "Feature Extraction",
        "Model Application",
        "Evaluation",
        "Output Generation"
    ]


def _extract_concepts_smart(text: str) -> list[str]:
    """Extract top concepts for flowchart (replaces broken mindmap)."""
    # Count capitalized multi-word phrases
    phrases = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})\b', text)
    freq = {}
    for p in phrases:
        if len(p) > 5 and len(p.split()) >= 2:
            freq[p] = freq.get(p, 0) + 1

    top = sorted(freq.items(), key=lambda x: -x[1])[:7]
    concepts = [_safe(t[0], 35) for t in top if t[0]]

    return concepts if len(concepts) >= 3 else [
        "Document Analysis",
        "Key Concepts",
        "Methodology",
        "Results",
        "Conclusions"
    ]


# ══ Flowchart builders ═════════════════════════════════════════════════════════

def _build_process_flowchart(steps: list[str], title: str = "") -> str:
    """Build guaranteed-valid Mermaid flowchart from actual content steps."""
    lines = ["flowchart TD"]

    # Never use filename as start — use a generic meaningful label
    lines.append('    ST["Document Process Flow"]')

    prev     = "ST"
    node_ids = list("ABCDEFGHIJ")

    for i, step in enumerate(steps[:8]):
        nid   = node_ids[i] if i < len(node_ids) else f"N{i}"
        label = _safe(step, 42)

        # Skip if label looks like a filename
        if re.search(r'\w+\.\w{2,5}$', label) or len(label.split()) < 2:
            continue

        lines.append(f'    {nid}["{label}"]')
        lines.append(f"    {prev} --> {nid}")
        prev = nid

    lines.append('    END["Output / Result"]')
    lines.append(f"    {prev} --> END")

    return "\n".join(lines)


def _build_concept_flowchart(concepts: list[str]) -> str:
    """
    Build a concept map as a flowchart (mindmap syntax is unreliable).
    Uses a hub-and-spoke layout.
    """
    lines = ["flowchart TD"]
    lines.append('    HUB["Key Concepts"]')

    node_ids = list("ABCDEFGHIJ")

    for i, concept in enumerate(concepts[:8]):
        nid   = node_ids[i] if i < len(node_ids) else f"C{i}"
        label = _safe(concept, 35)
        lines.append(f'    {nid}["{label}"]')
        lines.append(f"    HUB --> {nid}")

    return "\n".join(lines)


def _build_research_flowchart(text: str) -> str:
    """Build research paper structure flowchart."""
    sections = [
        ("ABSTRACT",      "Abstract"),
        ("INTRODUCTION",  "Introduction"),
        ("RELATED",       "Related Work"),
        ("LITERATURE",    "Literature Review"),
        ("METHODOLOGY",   "Methodology"),
        ("DATASET",       "Dataset"),
        ("EXPERIMENT",    "Experiments"),
        ("RESULT",        "Results"),
        ("DISCUSSION",    "Discussion"),
        ("CONCLUSION",    "Conclusion"),
    ]

    text_up = text.upper()
    found   = [label for keyword, label in sections if keyword in text_up]

    if len(found) < 3:
        found = ["Introduction", "Methodology", "Experiments", "Results", "Conclusion"]

    lines = ["flowchart TD"]
    lines.append('    PAPER["Research Paper"]')
    prev = "PAPER"

    node_ids = list("ABCDEFGHIJ")
    for i, section in enumerate(found[:8]):
        nid   = node_ids[i]
        label = _safe(section, 35)
        lines.append(f'    {nid}["{label}"]')
        lines.append(f"    {prev} --> {nid}")
        prev = nid

    return "\n".join(lines)


# ══ Public API ══════════════════════════════════════════════════════════════════

def generate_flowchart_from_text(text: str, title: str = "Process",
                                  diagram_type: str = "process") -> str:
    # Strip filename from title
    title = re.sub(r'\.\w{2,5}$', '', title).strip()
    title = re.sub(r'[_\-]+', ' ', title).strip()

    # Try AI first
    ai = _generate_with_ai(text, diagram_type)
    if ai:
        return ai

    dtype = diagram_type.lower()
    if "research" in dtype:
        return _build_research_flowchart(text)
    elif "mind" in dtype or "concept" in dtype:
        concepts = _extract_concepts_smart(text)
        return _build_concept_flowchart(concepts)
    else:
        steps = _extract_steps_smart(text)
        return _build_process_flowchart(steps, title)

def generate_research_flowchart(text: str) -> str:
    ai = _generate_with_ai(text, "research paper structure")
    return ai if ai else _build_research_flowchart(text)


def generate_concept_map(text: str) -> str:
    """
    Generate concept map as a flowchart (mindmap avoided — too many syntax issues).
    """
    ai = _generate_with_ai(text, "concept map flowchart hub and spoke")
    if ai:
        return ai
    concepts = _extract_concepts_smart(text)
    return _build_concept_flowchart(concepts)