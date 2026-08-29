"""Contained visual styling for the combined faculty demonstration."""


THEME_CSS = """
<style>
:root { --qm-navy:#071A2F; --qm-cyan:#24D6D0; --qm-violet:#8B7CFF; --qm-panel:#102A43; }
.stApp { background: linear-gradient(145deg, #F7FBFF 0%, #EEF6FF 55%, #F7F4FF 100%); }
.qm-hero { padding: 1.4rem 1.6rem; border-radius: 22px; color: white;
  background: linear-gradient(120deg, var(--qm-navy), #123E67 62%, #3D2D78); box-shadow: 0 18px 44px #17324d25; }
.qm-eyebrow { color: #8FF8F1; font-size: .78rem; letter-spacing: .14em; font-weight: 700; }
.qm-card { background: #ffffffd9; border: 1px solid #D8E8F5; border-radius: 18px;
  padding: 1rem 1.1rem; min-height: 132px; box-shadow: 0 8px 24px #17324d12; }
.qm-verdict { border-left: 5px solid var(--qm-cyan); background: white; padding: .85rem 1rem; border-radius: 12px; }
.qm-footnote { color:#52677B; font-size:.86rem; }
</style>
"""
