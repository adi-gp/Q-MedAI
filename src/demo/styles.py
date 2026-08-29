"""Contained visual styling for the combined faculty demonstration."""

THEME_CSS = """
<style>
:root {
  --qm-bg: #f4f8fc;
  --qm-surface: #ffffff;
  --qm-surface-2: #f8fbff;
  --qm-navy: #0b1f33;
  --qm-navy-2: #173a5e;
  --qm-text: #16263a;
  --qm-muted: #60758a;
  --qm-border: #d9e5ef;
  --qm-cyan: #16c7c0;
  --qm-cyan-dark: #0e8f8a;
  --qm-violet: #6558e8;
  --qm-green: #1a9b68;
  --qm-amber: #a66a00;
  --qm-red: #c23b45;
  --qm-shadow: 0 10px 30px rgba(22, 43, 67, .08);
}

/* ------------------------------
   Global canvas / spacing
   ------------------------------ */
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
               "Segoe UI", sans-serif;
}

.stApp { color: var(--qm-navy);
  background:
    radial-gradient(circle at 90% 0%, rgba(101,88,232,.07), transparent 32rem),
    radial-gradient(circle at 15% 10%, rgba(22,199,192,.07), transparent 26rem),
    var(--qm-bg);
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1480px !important;
  padding-top: 4.75rem !important;
  padding-bottom: 4rem !important;
  padding-left: 2.75rem !important;
  padding-right: 2.75rem !important;
}

header[data-testid="stHeader"] {
  background: rgba(244,248,252,.88) !important;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(217,229,239,.75);
}

/* ------------------------------
   Typography
   ------------------------------ */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
  color: var(--qm-navy) !important;
  letter-spacing: -.02em;
}

h1 {
  font-size: clamp(2rem, 4vw, 3.1rem) !important;
  line-height: 1.05 !important;
}

p, li, span, div {
  text-rendering: optimizeLegibility;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
.stCaption {
  color: var(--qm-muted) !important;
}

/* Native widget labels: fixes the almost-white labels visible in the old UI */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stRadio label,
.stCheckbox label {
  color: #32485f !important;
  font-weight: 650 !important;
  opacity: 1 !important;
}

/* Native metric contrast */
[data-testid="stMetric"] {
  background: var(--qm-surface);
  border: 1px solid var(--qm-border);
  border-radius: 16px;
  padding: .85rem 1rem;
  box-shadow: 0 5px 18px rgba(22,43,67,.055);
}

[data-testid="stMetricLabel"] p {
  color: var(--qm-muted) !important;
  font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
  color: var(--qm-navy) !important;
  font-weight: 800 !important;
}

/* ------------------------------
   Sidebar
   ------------------------------ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1f33 0%, #122942 100%) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
  color: #edf6ff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 11px;
  padding: .32rem .45rem;
  transition: background .16s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,.08);
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #afc2d3 !important;
}

/* ------------------------------
   Form controls
   ------------------------------ */
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  border-radius: 10px !important;
}

[data-testid="stForm"] {
  background: rgba(255,255,255,.72);
  border: 1px solid var(--qm-border);
  border-radius: 20px;
  padding: 1.25rem 1.25rem .9rem;
  box-shadow: var(--qm-shadow);
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
  border-radius: 11px !important;
  font-weight: 750 !important;
  min-height: 2.8rem;
  border: 0 !important;
  box-shadow: 0 8px 18px rgba(101,88,232,.14);
}

[data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
  color: white !important;
  background: linear-gradient(115deg, #0f8f8a, #6558e8) !important;
}

[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}

/* ------------------------------
   Alerts: consistent contrast
   ------------------------------ */
[data-testid="stAlert"] {
  border-radius: 14px !important;
  border: 1px solid var(--qm-border) !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
  color: #203247 !important;
}

/* ------------------------------
   Tables / code
   ------------------------------ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--qm-border);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 5px 18px rgba(22,43,67,.05);
}

[data-testid="stCodeBlock"] {
  border-radius: 14px;
  overflow: hidden;
}

/* ------------------------------
   Custom Q-MedAI components
   ------------------------------ */
.qm-shell {
  margin-top: .25rem;
}

.qm-hero {
  position: relative;
  overflow: hidden;
  padding: 1.6rem 1.75rem;
  border-radius: 24px;
  color: #fff;
  background: linear-gradient(118deg, #0b1f33 0%, #173f66 58%, #5143b6 100%);
  box-shadow: 0 18px 44px rgba(23,50,77,.18);
  margin-bottom: 1.45rem;
}

.qm-hero:after {
  content: "";
  position: absolute;
  width: 240px;
  height: 240px;
  border-radius: 999px;
  right: -90px;
  top: -120px;
  background: rgba(22,199,192,.17);
}

.qm-hero-title {
  color: #fff !important;
  font-size: clamp(1.65rem, 3vw, 2.35rem);
  line-height: 1.14;
  font-weight: 820;
  margin: .25rem 0 .45rem;
  max-width: 850px;
}

.qm-hero-copy {
  color: #dbe9f5;
  font-size: 1rem;
  max-width: 850px;
  line-height: 1.55;
}

.qm-eyebrow {
  color: #88f0e9;
  font-size: .76rem;
  letter-spacing: .14em;
  font-weight: 820;
  text-transform: uppercase;
}

.qm-section {
  margin: 1.75rem 0 .8rem;
}

.qm-section-title {
  color: var(--qm-navy);
  font-size: 1.55rem;
  font-weight: 820;
  letter-spacing: -.025em;
  margin-bottom: .25rem;
}

.qm-section-copy {
  color: var(--qm-muted);
  line-height: 1.55;
  max-width: 900px;
}

.qm-card { color: var(--qm-navy);
  background: var(--qm-surface);
  border: 1px solid var(--qm-border);
  border-radius: 18px;
  padding: 1.05rem 1.15rem;
  box-shadow: var(--qm-shadow);
}

.qm-card-title {
  color: var(--qm-navy);
  font-size: 1rem;
  font-weight: 800;
  margin-bottom: .3rem;
}

.qm-card-copy {
  color: var(--qm-muted);
  line-height: 1.5;
  font-size: .94rem;
}

.qm-stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: .9rem;
  margin: 1rem 0 1.35rem;
}

.qm-stat {
  background: var(--qm-surface);
  border: 1px solid var(--qm-border);
  border-radius: 16px;
  padding: .95rem 1rem;
  box-shadow: 0 6px 20px rgba(22,43,67,.055);
}

.qm-stat-label {
  color: var(--qm-muted);
  font-size: .76rem;
  text-transform: uppercase;
  letter-spacing: .07em;
  font-weight: 760;
}

.qm-stat-value {
  color: var(--qm-navy);
  font-size: 1.65rem;
  line-height: 1.05;
  margin-top: .38rem;
  font-weight: 850;
}

.qm-stat-note {
  color: var(--qm-muted);
  font-size: .78rem;
  margin-top: .25rem;
}

.qm-verdict { color: var(--qm-navy);
  border: 1px solid var(--qm-border);
  border-left: 5px solid var(--qm-cyan);
  background: var(--qm-surface);
  padding: .95rem 1rem;
  border-radius: 14px;
  box-shadow: 0 5px 18px rgba(22,43,67,.045);
}

.qm-verdict-label {
  color: var(--qm-muted);
  text-transform: uppercase;
  letter-spacing: .06em;
  font-size: .72rem;
  font-weight: 780;
}

.qm-verdict-headline {
  color: var(--qm-navy);
  font-size: 1rem;
  font-weight: 720;
  margin-top: .35rem;
}

.qm-pill {
  display: inline-block;
  padding: .25rem .52rem;
  border-radius: 999px;
  font-size: .72rem;
  letter-spacing: .035em;
  font-weight: 820;
  margin-right: .35rem;
  background: #e9f7f6;
  color: #0d736f;
  border: 1px solid #bfeae7;
}

.qm-pill-purple {
  background: #f0edff;
  color: #5746c9;
  border-color: #d9d1ff;
}

.qm-pill-dark {
  background: #edf2f7;
  color: #34495f;
  border-color: #d8e2eb;
}

.qm-flow {
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: .7rem;
  margin: .95rem 0 1.25rem;
}

.qm-flow-step {
  position: relative;
  min-height: 108px;
  padding: .85rem .9rem;
  border-radius: 15px;
  background: var(--qm-surface);
  border: 1px solid var(--qm-border);
}

.qm-flow-num {
  color: var(--qm-cyan-dark);
  font-size: .73rem;
  font-weight: 850;
  letter-spacing: .07em;
}

.qm-flow-title {
  color: var(--qm-navy);
  font-weight: 800;
  margin-top: .28rem;
}

.qm-flow-copy {
  color: var(--qm-muted);
  font-size: .8rem;
  line-height: 1.4;
  margin-top: .2rem;
}

.qm-result-primary {
  border-radius: 20px;
  padding: 1.35rem 1.45rem;
  color: #fff;
  background: linear-gradient(118deg, #0c776f 0%, #0f938b 48%, #5143b6 100%);
  box-shadow: 0 16px 38px rgba(24,83,103,.18);
  margin: .9rem 0 1rem;
}

.qm-result-kicker {
  color: #bff9f5;
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 800;
  font-size: .72rem;
}

.qm-result-score {
  color: #fff;
  font-size: clamp(2.4rem, 6vw, 4rem);
  font-weight: 900;
  line-height: 1;
  margin: .45rem 0;
}

.qm-result-copy {
  color: #e4f6f4;
  font-size: .92rem;
}

.qm-score-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: .85rem;
  margin: .8rem 0 1.2rem;
}

.qm-score-card {
  background: var(--qm-surface);
  border: 1px solid var(--qm-border);
  border-radius: 16px;
  padding: .95rem 1rem;
}

.qm-score-label {
  color: var(--qm-muted);
  font-size: .76rem;
  font-weight: 760;
  text-transform: uppercase;
  letter-spacing: .055em;
}

.qm-score-value {
  color: var(--qm-navy);
  font-weight: 860;
  font-size: 1.65rem;
  margin: .25rem 0;
}

.qm-score-note {
  color: var(--qm-muted);
  font-size: .78rem;
}

.qm-field-group {
  color: var(--qm-navy);
  font-weight: 820;
  font-size: .95rem;
  margin: .25rem 0 .45rem;
}

.qm-footnote {
  color: var(--qm-muted);
  font-size: .83rem;
  line-height: 1.5;
}

.qm-divider {
  height: 1px;
  background: var(--qm-border);
  margin: 1.4rem 0;
}

.qm-feature-row {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin: .55rem 0 1rem;
}

.qm-empty {
  border: 1px dashed #bfd0df;
  background: rgba(255,255,255,.54);
  border-radius: 16px;
  padding: 1.25rem;
  color: var(--qm-muted);
}

@media (max-width: 900px) {
  .qm-stat-grid,
  .qm-score-grid,
  .qm-flow {
    grid-template-columns: 1fr;
  }

  [data-testid="stMainBlockContainer"],
  .block-container {
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
    padding-top: 4.25rem !important;
  }
}


/* ---------- Faculty presentation scale ---------- */

[data-testid="stSidebar"] {
    min-width: 250px !important;
    width: 250px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    font-size: 0.96rem !important;
    padding: 0.42rem 0.55rem !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
}

[data-testid="stWidgetLabel"] p {
    font-size: 0.88rem !important;
}

[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    min-height: 42px !important;
    font-size: 0.92rem !important;
}

[data-testid="stDataFrame"] {
    font-size: 0.90rem !important;
}

.qm-hero {
    padding: 1.8rem 2rem !important;
}

.qm-hero-title {
    font-size: clamp(1.9rem, 3vw, 2.65rem) !important;
}

.qm-hero-copy {
    font-size: 1.02rem !important;
}

.qm-card {
    padding: 1.2rem 1.3rem !important;
}

.qm-stat {
    padding: 1.1rem 1.15rem !important;
}

.qm-stat-value {
    font-size: 1.85rem !important;
}

.qm-flow-step {
    min-height: 118px !important;
    padding: 1rem !important;
}

.qm-section-title {
    font-size: 1.7rem !important;
}

.qm-section-copy {
    font-size: 0.98rem !important;
}

[data-testid="stFormSubmitButton"] > button {
    min-height: 3.2rem !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
}


/* Primary button text visibility */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button *,
.stButton > button,
.stButton > button * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

/* Disabled buttons should still be readable */
.stButton > button:disabled,
.stButton > button:disabled * {
    color: rgba(255,255,255,.72) !important;
}

</style>
"""
