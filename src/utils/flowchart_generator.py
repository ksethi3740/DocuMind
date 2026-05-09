"""
DocuMind — Bulletproof Flowchart Generator
Compact, properly-sized nodes. AI-powered with smart fallback.
"""

import os
import re
import streamlit.components.v1 as components
from pathlib import Path

# Load env
_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV.exists():
    with open(_ENV) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


# ══ Label cleaner ══════════════════════════════════════════════════════════════

def _safe(text: str, max_len: int = 38) -> str:
    """Strip every character that breaks Mermaid. Keep labels SHORT."""
    if not text:
        return "Step"
    text = text.replace('"','').replace("'",'').replace('`','')
    text = re.sub(r'[<>{}\[\]\\|()&@#$%^*+=]', '', text)
    text = text.replace(':', ' -').replace(';', ',').replace('_', ' ')
    # Remove file extensions (e.g. .docx .pdf)
    text = re.sub(r'\b\w+\.\w{2,5}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_len:
        # Cut at word boundary
        text = text[:max_len].rsplit(' ', 1)[0].strip()
        if text:
            text += '...'
    return text or "Step"


# ══ Mermaid renderer ═══════════════════════════════════════════════════════════

def render_mermaid(mermaid_code: str):
    """
    Render compact Mermaid diagram that fits without scrolling.
    Forces proper SVG scaling via viewBox manipulation.
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%;
    background: #0f172a;
    overflow: hidden;
  }}
  #container {{
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
  }}
  .mermaid {{
    width: 100%;
    max-width: 760px;
  }}
  /* Force ALL svg elements to scale down */
  .mermaid svg {{
    width: 100% !important;
    height: auto !important;
    max-height: 420px !important;
    display: block;
    margin: 0 auto;
  }}
  /* Compact node text */
  .mermaid .nodeLabel,
  .mermaid .label,
  .mermaid text {{
    font-size: 12px !important;
    font-family: Inter, sans-serif !important;
  }}
  /* Reduce node padding */
  .mermaid .node rect,
  .mermaid .node polygon {{
    padding: 4px 8px !important;
  }}
</style>
</head>
<body>
<div id="container">
  <div class="mermaid" id="diagram">
{mermaid_code}
  </div>
</div>
<script>
  mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
    flowchart: {{
      htmlLabels: false,
      curve: 'basis',
      nodeSpacing: 30,
      rankSpacing: 40,
      padding: 8,
      useMaxWidth: true,
      diagramPadding: 8
    }},
    themeVariables: {{
      fontSize: '12px',
      primaryColor: '#4f46e5',
      primaryTextColor: '#e2e8f0',
      primaryBorderColor: '#6366f1',
      lineColor: '#64748b',
      secondaryColor: '#1e293b',
      tertiaryColor: '#0f172a',
      background: '#0f172a',
      mainBkg: '#1e293b',
      nodeBorder: '#6366f1',
      clusterBkg: '#1e293b',
      fontFamily: 'Inter, sans-serif',
      nodeTextColor: '#e2e8f0',
      labelBoxBkgColor: '#1e293b',
      labelBoxBorderColor: '#6366f1'
    }}
  }});

  async function renderDiagram() {{
    try {{
      const el = document.getElementById('diagram');
      const {{ svg }} = await mermaid.render('mermaid-svg', el.textContent.trim());
      el.innerHTML = svg;

      // Force the SVG to scale properly
      const svgEl = el.querySelector('svg');
      if (svgEl) {{
        // Get natural dimensions
        const w = svgEl.getAttribute('width')  || svgEl.viewBox?.baseVal?.width  || 600;
        const h = svgEl.getAttribute('height') || svgEl.viewBox?.baseVal?.height || 400;

        // Set viewBox if not set
        if (!svgEl.getAttribute('viewBox')) {{
          svgEl.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
        }}

        // Remove fixed dimensions — let CSS control
        svgEl.removeAttribute('width');
        svgEl.removeAttribute('height');
        svgEl.style.width  = '100%';
        svgEl.style.height = 'auto';
        svgEl.style.maxHeight = '420px';

        // Notify parent of actual rendered height
        const rect = svgEl.getBoundingClientRect();
        const actualH = Math.min(Math.ceil(rect.height) + 40, 460);
        window.parent.postMessage({{ type: 'mermaid-height', height: actualH }}, '*');
      }}
    }} catch(err) {{
      document.getElementById('diagram').innerHTML =
        '<div style="color:#f87171;padding:20px;font-family:monospace;font-size:12px;">'
        + '⚠️ Diagram error: ' + err.message + '<br><br>'
        + '<pre style=\\"font-size:10px;white-space:pre-wrap;\\">{mermaid_code[:200]}</pre>'
        + '</div>';
    }}
  }}

  renderDiagram();
</script>
</body>
</html>"""
    components.html(html, height=460, scrolling=False)


# ══ AI generation ══════════════════════════════════════════════════════════════

def _generate_with_ai(text: str, diagram_type: str) -> str | None:
    """Use Groq or Gemini to generate proper Mermaid code."""
    groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not groq_key and not gemini_key:
        return None

    if "research" in diagram_type.lower():
        task = "a flowchart showing the research paper structure with sections like Abstract, Introduction, Methodology, Results, Conclusion"
    elif "mind" in diagram_type.lower() or "concept" in diagram_type.lower():
        task = "a hub-and-spoke flowchart showing the 6 main concepts/topics from the document"
    else:
        task = "a flowchart showing the main process steps in the correct order"

    prompt = f"""Analyze this document and generate {task}.

DOCUMENT TEXT:
{text[:2500]}

STRICT OUTPUT RULES — follow EXACTLY or it will break:
1. Output ONLY Mermaid code. NO explanation. NO markdown. NO backticks.
2. Start with exactly: flowchart TD
3. Node IDs: single letters only — A, B, C, D, E, F, G, H
4. Labels: max 30 characters, in double quotes, NO special characters
5. FORBIDDEN in labels: ( ) [ ] {{ }} < > | \\ : ; & ' ` _ . #
6. Use --> for all connections
7. Maximum 7 nodes total
8. No style, classDef, or subgraph lines

CORRECT EXAMPLE OUTPUT:
flowchart TD
    A["Video Frame Capture"]
    B["Face Detection"]
    C["Feature Extraction"]
    D["CNN Classification"]
    E["Drowsiness Detection"]
    F["Emotion Recognition"]
    G["Safety Alert System"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

Generate the diagram now (flowchart code only):"""

    # Try Groq
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp   = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r'```(?:mermaid)?\s*', '', raw)
            raw = re.sub(r'```\s*$', '', raw).strip()
            if raw.startswith("flowchart") or raw.startswith("graph"):
                validated = _validate(raw)
                if validated:
                    return validated
        except Exception as e:
            print(f"[Flowchart] Groq error: {e}")

    # Try Gemini
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp   = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            raw = (resp.text or "").strip()
            raw = re.sub(r'```(?:mermaid)?\s*', '', raw)
            raw = re.sub(r'```\s*$', '', raw).strip()
            if raw.startswith("flowchart") or raw.startswith("graph"):
                validated = _validate(raw)
                if validated:
                    return validated
        except Exception as e:
            print(f"[Flowchart] Gemini error: {e}")

    return None


def _validate(code: str) -> str | None:
    """Validate and fix AI-generated Mermaid code."""
    lines  = code.strip().split("\n")
    output = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Directive
        if s.startswith(("flowchart", "graph", "mindmap")):
            output.append(s)
            continue

        # Arrow
        if "-->" in s or "---" in s:
            s = re.sub(r'\|[^|]*\|', '', s)  # remove arrow labels
            output.append("    " + s.strip())
            continue

        # Style lines — skip
        if s.startswith(("style ", "classDef ", "class ", "subgraph", "end")):
            continue

        # Node definition
        m = re.match(r'^(\w+)([\[\(]{1,2})"?([^"\]\)]*)"?([\]\)]{1,2})$', s)
        if m:
            nid   = m.group(1)
            ob    = m.group(2)
            label = _safe(m.group(3), 35)
            cb    = m.group(4)
            output.append(f'    {nid}{ob}"{label}"{cb}')
            continue

        # Plain text node (no brackets) — skip to avoid errors
        if re.match(r'^\w+$', s):
            continue

        output.append("    " + s)

    result = "\n".join(output)
    # Must have at least one arrow to be valid
    return result if "-->" in result else None


# ══ Local extraction fallback ══════════════════════════════════════════════════

def _extract_steps(text: str) -> list[str]:
    """Extract meaningful process steps — returns clean SHORT strings."""
    steps = []

    # Strategy 1: numbered lists
    matches = re.findall(
        r'(?:^|\n)\s*\d+[\.\)]\s+([A-Z][^\n]{15,80})',
        text, re.MULTILINE
    )
    for m in matches:
        clean = _safe(m.strip(), 35)
        if clean and len(clean.split()) >= 2 and clean != "Step":
            steps.append(clean)
    if len(steps) >= 3:
        return steps[:7]

    # Strategy 2: bullet points
    matches = re.findall(
        r'(?:^|\n)\s*[•\-\*]\s+([A-Z][^\n]{15,80})',
        text, re.MULTILINE
    )
    for m in matches:
        clean = _safe(m.strip(), 35)
        if clean and len(clean.split()) >= 2:
            steps.append(clean)
    if len(steps) >= 3:
        return steps[:7]

    # Strategy 3: action-verb sentences
    actions = [
        'capture', 'collect', 'preprocess', 'resize', 'normalize',
        'extract', 'detect', 'classify', 'train', 'evaluate', 'apply',
        'generate', 'compute', 'implement', 'process', 'analyse',
        'install', 'configure', 'deploy', 'test', 'validate', 'use'
    ]
    sentences = re.split(r'[.!?]\s+', text)
    for sent in sentences:
        sent = sent.strip()
        if 3 <= len(sent.split()) <= 15:
            if any(a in sent.lower() for a in actions):
                clean = _safe(sent, 35)
                if clean and clean != "Step":
                    steps.append(clean)
    if len(steps) >= 3:
        return steps[:7]

    # Fallback
    return [
        "Data Collection",
        "Preprocessing",
        "Feature Extraction",
        "Model Application",
        "Evaluation",
        "Result Generation"
    ]


def _extract_concepts(text: str) -> list[str]:
    """Extract top concepts for hub-and-spoke diagram."""
    phrases = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})\b', text)
    freq = {}
    for p in phrases:
        if 5 < len(p) < 35 and len(p.split()) >= 2:
            freq[p] = freq.get(p, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:7]
    return [_safe(t[0], 32) for t in top] or [
        "Document Analysis", "Key Methods",
        "Results", "Conclusions", "Future Work"
    ]


# ══ Flowchart builders ══════════════════════════════════════════════════════════

def _build_process(steps: list[str]) -> str:
    lines = ["flowchart TD"]
    lines.append('    ST["Start"]')
    prev     = "ST"
    node_ids = list("ABCDEFG")

    for i, step in enumerate(steps[:7]):
        nid   = node_ids[i] if i < len(node_ids) else f"N{i}"
        label = _safe(step, 35)
        lines.append(f'    {nid}["{label}"]')
        lines.append(f"    {prev} --> {nid}")
        prev = nid

    lines.append('    END["End"]')
    lines.append(f"    {prev} --> END")
    return "\n".join(lines)


def _build_research(text: str) -> str:
    sections_check = [
        ("ABSTRACT",     "Abstract"),
        ("INTRODUCTION", "Introduction"),
        ("RELATED",      "Related Work"),
        ("METHODOLOGY",  "Methodology"),
        ("DATASET",      "Dataset"),
        ("EXPERIMENT",   "Experiments"),
        ("RESULT",       "Results"),
        ("DISCUSSION",   "Discussion"),
        ("CONCLUSION",   "Conclusion"),
    ]
    text_up = text.upper()
    found   = [label for kw, label in sections_check if kw in text_up]

    if len(found) < 3:
        found = ["Introduction","Methodology","Experiments","Results","Conclusion"]

    lines = ["flowchart TD"]
    lines.append('    P["Research Paper"]')
    prev     = "P"
    node_ids = list("ABCDEFGHI")

    for i, sec in enumerate(found[:8]):
        nid = node_ids[i] if i < len(node_ids) else f"S{i}"
        lines.append(f'    {nid}["{_safe(sec, 30)}"]')
        lines.append(f"    {prev} --> {nid}")
        prev = nid

    return "\n".join(lines)


def _build_concept(concepts: list[str]) -> str:
    lines = ["flowchart TD"]
    lines.append('    HUB["Key Concepts"]')
    node_ids = list("ABCDEFG")

    for i, c in enumerate(concepts[:7]):
        nid = node_ids[i] if i < len(node_ids) else f"C{i}"
        lines.append(f'    {nid}["{_safe(c, 30)}"]')
        lines.append(f"    HUB --> {nid}")

    return "\n".join(lines)


# ══ Public API ══════════════════════════════════════════════════════════════════

def generate_flowchart_from_text(text: str, title: str = "Process",
                                  diagram_type: str = "process") -> str:
    # Try AI first
    ai = _generate_with_ai(text, diagram_type)
    if ai:
        return ai

    # Local fallback
    dtype = diagram_type.lower()
    if "research" in dtype:
        return _build_research(text)
    elif "mind" in dtype or "concept" in dtype:
        return _build_concept(_extract_concepts(text))
    else:
        return _build_process(_extract_steps(text))


def generate_research_flowchart(text: str) -> str:
    ai = _generate_with_ai(text, "research paper structure")
    return ai if ai else _build_research(text)


def generate_concept_map(text: str) -> str:
    ai = _generate_with_ai(text, "concept map hub spoke")
    return ai if ai else _build_concept(_extract_concepts(text))