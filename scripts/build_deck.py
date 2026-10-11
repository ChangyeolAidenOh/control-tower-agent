"""Stage 7 deck generator (v2 -- flat design system).

Reads deck/deck_content.yaml, renders outputs/deck/stage7_deck.pptx and
outputs/deck/tom.pptx. Value system unchanged: {ref: "file:dotted.path"}
from frozen JSONs, {frozen, source} literals audited at the end.

v2 design system:
- one header grammar on every slide: kicker / title / full-width hairline
  at a fixed y / bold headline; the short floating accent bar is removed
- flat square panels (no rounded corners, no top bars): light fill, hairline
  border, a 4.5pt left accent strip inside the panel edge, title + hairline
  + evenly distributed item rows -- panels stretch to the content band bottom
- every content block lives in the band y = 2.2 .. 6.95; sources at 7.06
- shadows stripped explicitly on every shape

Run: python -m scripts.build_deck
Deps: pip install python-pptx pyyaml
"""

import json
from pathlib import Path

import yaml
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

CONTENT_PATH = Path("deck/deck_content.yaml")
OUT_DIR = Path("outputs/deck")
DECK_PATH = OUT_DIR / "stage7_deck.pptx"
TOM_PATH = OUT_DIR / "tom.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)
CONTENT_W = Emu(int(SLIDE_W - 2 * MARGIN))
BAND_TOP = Inches(2.2)
BAND_BOT = Inches(6.95)
GAP = Inches(0.3)

INK = RGBColor(0x20, 0x21, 0x33)
MUTED = RGBColor(0x60, 0x61, 0x72)
ACCENT = RGBColor(0x0F, 0x62, 0xFE)
WARN = RGBColor(0xB4, 0x23, 0x18)
OK = RGBColor(0x1A, 0x7F, 0x4B)
LINE = RGBColor(0xDE, 0xDF, 0xE6)
SEP = RGBColor(0xEC, 0xED, 0xF2)
PANEL_BG = RGBColor(0xF6, 0xF7, 0xFA)
AGENT_C = RGBColor(0xE0, 0x7B, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TONES = {"ok": OK, "warn": WARN, "muted": MUTED, "ink": INK, "accent": ACCENT, "agent": AGENT_C}

POLICY_COLORS = {
    "b": WARN, "a4": ACCENT, "a4_sl": RGBColor(0x08, 0x47, 0xB0),
    "apto": RGBColor(0x6F, 0x2D, 0xBD), "l0_rule_ct": INK,
    "a1": RGBColor(0xA6, 0xA7, 0xB3), "a2": RGBColor(0xA6, 0xA7, 0xB3),
    "a3": RGBColor(0xA6, 0xA7, 0xB3), "a5": OK, "agent": AGENT_C,
}

FONT = "Malgun Gothic"


class Resolver:
    def __init__(self, file_map):
        self.data = {k: json.loads(Path(p).read_text()) for k, p in file_map.items()}
        self.frozen_audit = []

    def get_path(self, spec):
        file_key, dotted = spec.split(":", 1)
        node = self.data[file_key]
        for part in dotted.split("."):
            node = node[part]
        return node

    def value(self, v, context=""):
        if isinstance(v, dict) and "ref" in v:
            raw = self.get_path(v["ref"])
            fmt = v.get("format")
            return fmt.format(raw) if fmt else str(raw)
        if isinstance(v, dict) and "frozen" in v:
            self.frozen_audit.append((context, v["frozen"], v.get("source", "?")))
            return v["frozen"]
        return str(v)


def strip_shadow(shape):
    shape.shadow.inherit = False


def rs(run, size, bold=False, color=INK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT


def textbox(slide, x, y, w, h, anchor=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    if anchor:
        tf.vertical_anchor = anchor
    return tf


def para(tf, first):
    return tf.paragraphs[0] if first else tf.add_paragraph()


def rect(slide, x, y, w, h, fill=None, line_color=None, line_w=0.75):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line_color is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(line_w)
    strip_shadow(sh)
    return sh


def chevron(slide, x, y, w, h, fill, line_color, line_w):
    sh = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line_color
    sh.line.width = Pt(line_w)
    strip_shadow(sh)
    return sh


def header(slide, spec):
    tf = textbox(slide, MARGIN, Inches(0.36), CONTENT_W, Inches(0.26))
    r = para(tf, True).add_run()
    r.text = spec.get("kicker", "").upper()
    rs(r, 10.5, bold=True, color=ACCENT)
    tf = textbox(slide, MARGIN, Inches(0.62), CONTENT_W, Inches(0.52))
    r = para(tf, True).add_run()
    r.text = spec["title"]
    rs(r, 22, bold=True)
    rect(slide, MARGIN, Inches(1.26), CONTENT_W, Pt(1), fill=LINE)
    if spec.get("headline"):
        tf = textbox(slide, MARGIN, Inches(1.4), CONTENT_W, Inches(0.72))
        r = para(tf, True).add_run()
        r.text = spec["headline"]
        rs(r, 13.5, bold=True, color=INK)


def source_line(slide, sources):
    if not sources:
        return
    tf = textbox(slide, MARGIN, Inches(7.06), CONTENT_W, Inches(0.3))
    r = para(tf, True).add_run()
    r.text = "출처  " + " · ".join(sources)
    rs(r, 8.5, color=MUTED)


def panel(slide, x, y, w, h, title, items, tone="accent", item_size=11.5,
          title_color=None, dist=True):
    """Flat panel: left accent strip, title, hairline, distributed item rows."""
    rect(slide, x, y, w, h, fill=PANEL_BG, line_color=LINE)
    rect(slide, x, y, Pt(4.5), h, fill=TONES.get(tone, ACCENT))
    pad = Inches(0.26)
    tx = Emu(int(x + pad))
    tw = Emu(int(w - 2 * pad))
    tf = textbox(slide, tx, Emu(int(y + Inches(0.18))), tw, Inches(0.34))
    r = para(tf, True).add_run()
    r.text = title
    rs(r, 12.5, bold=True, color=title_color or INK)
    rect(slide, tx, Emu(int(y + Inches(0.56))), tw, Pt(1), fill=SEP)
    top = Emu(int(y + Inches(0.66)))
    avail = Emu(int(y + h - Inches(0.14) - top))
    n = max(1, len(items))
    if n == 1:
        tf = textbox(slide, tx, Emu(int(top + Inches(0.08))), tw, Emu(int(avail - Inches(0.08))))
        r = para(tf, True).add_run()
        r.text = items[0]
        rs(r, item_size, color=INK)
        return
    row_h = Emu(int(avail / n)) if dist else None
    cy = top
    for i, item in enumerate(items):
        h_i = row_h if dist else Inches(0.5)
        tf = textbox(slide, tx, cy, tw, h_i, anchor=MSO_ANCHOR.MIDDLE)
        r = para(tf, True).add_run()
        r.text = item
        rs(r, item_size, color=INK)
        if i < n - 1 and dist:
            rect(slide, tx, Emu(int(cy + h_i)), tw, Pt(0.75), fill=SEP)
        cy = Emu(int(cy + (h_i if dist else Inches(0.5))))


def grid_positions(n_cols, n_rows, top=BAND_TOP, bot=BAND_BOT):
    w = Emu(int((CONTENT_W - (n_cols - 1) * GAP) / n_cols))
    h = Emu(int((bot - top - (n_rows - 1) * GAP) / n_rows))
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            out.append((Emu(int(MARGIN + c * (w + GAP))), Emu(int(top + r * (h + GAP))), w, h))
    return out


def set_cell(cell, text, size=11, bold=False, color=INK, fill=None, align=None):
    cell.margin_left = Inches(0.12)
    cell.margin_right = Inches(0.08)
    cell.margin_top = Inches(0.03)
    cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.text = text
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    p = cell.text_frame.paragraphs[0]
    if align:
        p.alignment = align
    for r in p.runs:
        rs(r, size, bold=bold, color=color)


# ---------------------------------------------------------------- builders

def build_stat_cards(slide, spec, res):
    header(slide, spec)
    stats = spec["stats"]
    n = len(stats)
    w = Emu(int((CONTENT_W - (n - 1) * GAP) / n))
    y, h = Inches(2.5), Inches(4.0)
    for i, st in enumerate(stats):
        x = Emu(int(MARGIN + i * (w + GAP)))
        tone = st.get("tone", "ink")
        rect(slide, x, y, w, h, fill=PANEL_BG, line_color=LINE)
        rect(slide, x, y, Pt(4.5), h, fill=TONES[tone])
        ix = Emu(int(x + Inches(0.32)))
        iw = Emu(int(w - Inches(0.64)))
        tf = textbox(slide, ix, Emu(int(y + Inches(0.55))), iw, Inches(0.4))
        r = para(tf, True).add_run()
        r.text = st["label"]
        rs(r, 12.5, bold=True, color=MUTED)
        tf = textbox(slide, ix, Emu(int(y + Inches(1.35))), iw, Inches(0.85))
        r = para(tf, True).add_run()
        r.text = res.value(st["value"], f"{spec['id']}:{st['label']}:value")
        rs(r, 40, bold=True, color=TONES[tone])
        tf = textbox(slide, ix, Emu(int(y + Inches(2.55))), iw, Inches(1.1))
        r = para(tf, True).add_run()
        r.text = res.value(st["sub"], f"{spec['id']}:{st['label']}:sub")
        rs(r, 11.5, color=MUTED)
    source_line(slide, spec.get("sources"))


def build_definition(slide, spec, res):
    header(slide, spec)
    y = BAND_TOP
    st_h = Inches(0.78)
    bar = rect(slide, MARGIN, y, CONTENT_W, st_h, fill=INK)
    tf = bar.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.32)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run()
    r.text = spec["statement"]
    rs(r, 16.5, bold=True, color=WHITE)
    rows = spec["rows"]
    ty = Emu(int(y + st_h + Inches(0.25)))
    th = Emu(int(BAND_BOT - ty))
    gframe = slide.shapes.add_table(len(rows), 2, MARGIN, ty, CONTENT_W, th)
    tbl = gframe.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.columns[0].width = Inches(1.9)
    tbl.columns[1].width = Emu(int(CONTENT_W - Inches(1.9)))
    for i, (k, v) in enumerate(rows):
        fillc = WHITE if i % 2 else PANEL_BG
        set_cell(tbl.cell(i, 0), k, size=12, bold=True, color=ACCENT, fill=fillc)
        set_cell(tbl.cell(i, 1), v, size=12, fill=fillc)
        tbl.rows[i].height = Emu(int(th / len(rows)))
    source_line(slide, spec.get("sources"))


def band_row(slide, y, label, steps, highlight, approvals):
    tf = textbox(slide, MARGIN, y, Inches(1.05), Inches(0.75), anchor=MSO_ANCHOR.MIDDLE)
    first = True
    for part in label.split("\n"):
        r = para(tf, first).add_run()
        first = False
        r.text = part
        rs(r, 11, bold=True, color=MUTED)
    n = len(steps)
    gap = Inches(0.1)
    x0 = Emu(int(MARGIN + Inches(1.15)))
    w_total = Emu(int(CONTENT_W - Inches(1.15)))
    step_w = Emu(int((w_total - (n - 1) * gap) / n))
    h = Inches(0.62)
    for i, s in enumerate(steps):
        x = Emu(int(x0 + i * (step_w + gap)))
        fill = AGENT_C if i in highlight else PANEL_BG
        line_c = INK if i in approvals else LINE
        sh = chevron(slide, x, y, step_w, h, fill, line_c, 2.0 if i in approvals else 0.75)
        tf2 = sh.text_frame
        tf2.word_wrap = True
        p = tf2.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = s
        rs(r, 10.5, bold=i in highlight, color=WHITE if i in highlight else INK)


def build_before_after_band(slide, spec, res):
    header(slide, spec)
    band_row(slide, Inches(2.25), "AS-IS\n수작업", spec["asis_steps"], set(), set())
    band_row(slide, Inches(3.1), "To-Be\n승인 흐름", spec["asis_steps"],
             set(spec["agent_steps"]), set(spec["approval_steps"]))
    tf = textbox(slide, Emu(int(MARGIN + Inches(1.15))), Inches(3.85),
                 Emu(int(CONTENT_W - Inches(1.15))), Inches(0.3))
    r = para(tf, True).add_run()
    r.text = spec["legend"]
    rs(r, 10, color=MUTED)
    panels = spec["panels"]
    pw = Emu(int((CONTENT_W - GAP) / 2))
    py = Inches(4.35)
    ph = Emu(int(BAND_BOT - py))
    for i, p in enumerate(panels):
        x = Emu(int(MARGIN + i * (pw + GAP)))
        panel(slide, x, py, pw, ph, p["name"], p["items"], tone=p.get("tone", "accent"))
    source_line(slide, spec.get("sources"))


def build_criteria(slide, spec, res):
    header(slide, spec)
    layers = spec["layers"]
    lw = Emu(int((CONTENT_W - GAP) / 2))
    lh = Inches(1.55)
    for i, layer in enumerate(layers):
        x = Emu(int(MARGIN + i * (lw + GAP)))
        panel(slide, x, BAND_TOP, lw, lh, layer["name"], [layer["text"]], tone="accent" if i == 0 else "warn")
    ty = Emu(int(BAND_TOP + lh + Inches(0.25)))
    th = Emu(int(BAND_BOT - ty))
    tiers = spec["tiers"]
    gframe = slide.shapes.add_table(len(tiers) + 1, 2, MARGIN, ty, CONTENT_W, th)
    tbl = gframe.table
    tbl.first_row = True
    tbl.horz_banding = False
    tbl.columns[0].width = Inches(2.4)
    tbl.columns[1].width = Emu(int(CONTENT_W - Inches(2.4)))
    set_cell(tbl.cell(0, 0), "판정 (3-tier)", size=12, bold=True, color=WHITE, fill=INK)
    set_cell(tbl.cell(0, 1), "조건", size=12, bold=True, color=WHITE, fill=INK)
    for i, (name, cond) in enumerate(tiers, start=1):
        fillc = WHITE if i % 2 else PANEL_BG
        set_cell(tbl.cell(i, 0), name, size=12, bold=True, fill=fillc)
        set_cell(tbl.cell(i, 1), cond, size=12, fill=fillc)
    for i in range(len(tiers) + 1):
        tbl.rows[i].height = Emu(int(th / (len(tiers) + 1)))
    source_line(slide, spec.get("sources"))


def build_scatter_chart(slide, spec, res):
    header(slide, spec)
    cfg = spec["chart"]
    policies = res.get_path(cfg["points_ref"])
    frontier = res.get_path(cfg["frontier_ref"])
    agent = res.get_path(cfg["agent_ref"])
    refs = set(res.get_path(cfg["reference_policies_ref"]))
    xs = [v["mean_episode_cost"] for v in policies.values()]
    x_min, x_max = min(xs), max(xs)
    pad = (x_max - x_min) * 0.08

    data = XyChartData()
    s_line = data.add_series("프런티어")
    for p in frontier:
        s_line.add_data_point(policies[p]["mean_episode_cost"], policies[p]["pooled_sku_day_csl"])
    s_guide = data.add_series(cfg["hline"]["label"])
    s_guide.add_data_point(x_min - pad, cfg["hline"]["y"])
    s_guide.add_data_point(x_max + pad, cfg["hline"]["y"])
    series_keys = []
    for key, v in sorted(policies.items()):
        label = v["display_name"] + (" (참고)" if key in refs else "")
        s = data.add_series(label)
        s.add_data_point(v["mean_episode_cost"], v["pooled_sku_day_csl"])
        series_keys.append(key)
    s_agent = data.add_series("Agent")
    s_agent.add_data_point(agent["mean_episode_cost"], agent["pooled_sku_day_csl"])
    series_keys.append("agent")

    gframe = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
        MARGIN, Inches(2.1), Inches(8.35), Inches(4.25), data,
    )
    chart = gframe.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.RIGHT
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(9)
    chart.legend.font.name = FONT
    plot = chart.plots[0]
    for idx, series in enumerate(plot.series):
        fmt = series.format.line
        if idx == 0:
            fmt.color.rgb = ACCENT
            fmt.width = Pt(1.75)
        elif idx == 1:
            fmt.color.rgb = RGBColor(0xB9, 0xBA, 0xC4)
            fmt.width = Pt(1.0)
        else:
            fmt.fill.background()
            key = series_keys[idx - 2]
            series.marker.style = 8
            series.marker.size = 10 if key == "agent" else 7
            series.marker.format.fill.solid()
            series.marker.format.fill.fore_color.rgb = POLICY_COLORS.get(key, MUTED)
    vax = chart.value_axis
    vax.minimum_scale = 0.3
    vax.maximum_scale = 1.0
    vax.has_major_gridlines = True
    vax.major_gridlines.format.line.color.rgb = RGBColor(0xEB, 0xEC, 0xF1)
    cax = chart.category_axis
    cax.minimum_scale = max(0.0, (x_min - pad) // 100 * 100)
    cax.maximum_scale = -((-(x_max + pad)) // 100) * 100
    for ax in (vax, cax):
        ax.tick_labels.font.size = Pt(9)
        ax.tick_labels.font.name = FONT
        ax.format.line.color.rgb = LINE

    nx = Inches(9.25)
    nw = Emu(int(SLIDE_W - MARGIN - nx))
    panel(slide, nx, Inches(2.1), nw, Inches(4.25), "읽는 법 (기술 관찰)",
          spec["talk_notes"], tone="ink", item_size=10.5)
    tf = textbox(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.5))
    first = True
    for n in spec.get("footnotes", []):
        r = para(tf, first).add_run()
        first = False
        r.text = "주  " + n
        rs(r, 8.5, color=MUTED)
    source_line(slide, spec.get("sources"))


def build_judgment_matrix(slide, spec, res):
    header(slide, spec)
    comps = [("vs A2", "a2"), ("vs A3", "a3"), ("vs A4", "a4")]
    rs_keys = ["r3", "r5", "r10"]

    def h1_cell(rkey, comp):
        node = res.get_path(f"confirmatory:results.{rkey}.h1.{comp}")
        return (f"{node['point_estimate']:.1f} ({node['primary']['lower_5pct']:.1f})",
                bool(node["pass"]))

    def hb_cell(rkey):
        node = res.get_path(f"confirmatory:results.{rkey}.h1_b")
        return f"{node['point_estimate']:.1f}", bool(node["pass"])

    n_rows = 1 + len(comps) + 1 + len(spec["matrix_frozen_rows"])
    tw = Inches(7.35)
    ty = BAND_TOP
    th = Inches(3.6)
    gframe = slide.shapes.add_table(n_rows, 4, MARGIN, ty, tw, th)
    tbl = gframe.table
    tbl.first_row = True
    tbl.horz_banding = False
    widths = [Inches(2.35), Inches(1.68), Inches(1.68), Inches(1.64)]
    for i, wdt in enumerate(widths):
        tbl.columns[i].width = wdt
    for j, htxt in enumerate(["H1 (비용 Δ, 하한)", "r = 3", "r = 5", "r = 10"]):
        set_cell(tbl.cell(0, j), htxt, size=11, bold=True, color=WHITE, fill=INK)
    ri = 1
    for label, comp in comps:
        fillc = WHITE if ri % 2 else PANEL_BG
        set_cell(tbl.cell(ri, 0), label, size=10.5, bold=True, fill=fillc)
        for j, rkey in enumerate(rs_keys, start=1):
            txt, ok = h1_cell(rkey, comp)
            set_cell(tbl.cell(ri, j), f"{txt}  {'PASS' if ok else 'FAIL'}",
                     size=10, color=OK if ok else WARN, fill=fillc)
        ri += 1
    fillc = WHITE if ri % 2 else PANEL_BG
    set_cell(tbl.cell(ri, 0), "H1-b vs A-PTO", size=10.5, bold=True, fill=fillc)
    for j, rkey in enumerate(rs_keys, start=1):
        txt, ok = hb_cell(rkey)
        set_cell(tbl.cell(ri, j), f"{txt}  {'PASS' if ok else 'FAIL'}",
                 size=10, color=OK if ok else WARN, fill=fillc)
    ri += 1
    for row in spec["matrix_frozen_rows"]:
        res.frozen_audit.append((f"{spec['id']}:matrix:{row[0]}", " / ".join(row[1:]), "stage6_report.md §4"))
        fillc = WHITE if ri % 2 else PANEL_BG
        set_cell(tbl.cell(ri, 0), row[0], size=10.5, bold=True, fill=fillc)
        for j, vtxt in enumerate(row[1:], start=1):
            good = "PASS" in vtxt
            set_cell(tbl.cell(ri, j), vtxt, size=10, bold=True,
                     color=OK if good else WARN, fill=fillc)
        ri += 1
    for i in range(n_rows):
        tbl.rows[i].height = Emu(int(th / n_rows))

    note = res.value(spec["matrix_note"], f"{spec['id']}:matrix_note")
    tf = textbox(slide, MARGIN, Emu(int(ty + th + Inches(0.1))), tw, Inches(0.4))
    r = para(tf, True).add_run()
    r.text = "주  " + note
    rs(r, 9, color=MUTED)

    cx = Emu(int(MARGIN + tw + GAP))
    cw = Emu(int(SLIDE_W - MARGIN - cx))
    card = spec["h3_card"]
    lines = [res.value(l, f"{spec['id']}:h3:{i}") for i, l in enumerate(card["lines"])]
    panel(slide, cx, ty, cw, Emu(int(BAND_BOT - ty)), card["title"], lines, tone="accent", item_size=11)
    source_line(slide, spec.get("sources"))


def build_h5_breakdown(slide, spec, res):
    header(slide, spec)
    segs = spec["segments"]
    total = sum(s["n"] for s in segs)
    y, bar_h = BAND_TOP, Inches(0.78)
    x = Emu(int(MARGIN))
    for s in segs:
        w = Emu(int(CONTENT_W * s["n"] / total))
        sh = rect(slide, x, y, w, bar_h, fill=TONES[s["tone"]], line_color=WHITE, line_w=1.5)
        tf = sh.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = s["label"]
        rs(r, 13.5, bold=True, color=WHITE)
        if s.get("sub"):
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            r2 = p2.add_run()
            r2.text = s["sub"]
            rs(r2, 9, color=WHITE)
        x = Emu(int(x + w))
    tf = textbox(slide, MARGIN, Emu(int(y + bar_h + Inches(0.07))), CONTENT_W, Inches(0.28))
    r = para(tf, True).add_run()
    r.text = spec["threshold_note"]
    rs(r, 10, color=MUTED)
    by = Emu(int(y + bar_h + Inches(0.5)))
    beats = spec["beats"]
    bw = Emu(int((CONTENT_W - 2 * GAP) / 3))
    bh = Emu(int(BAND_BOT - by))
    tones = ["ink", "accent", "agent"]
    for i, (name, items) in enumerate(beats):
        bx = Emu(int(MARGIN + i * (bw + GAP)))
        panel(slide, bx, by, bw, bh, f"{i + 1}. {name}", items, tone=tones[i], item_size=11.5)
    source_line(slide, spec.get("sources"))


def build_grid_2x2(slide, spec, res):
    header(slide, spec)
    panels = spec["panels"]
    top = BAND_TOP if spec.get("headline") else Inches(1.62)
    pos = grid_positions(2, 2, top=top)
    for (x, y, w, h), p in zip(pos, panels):
        items = [res.value(it, f"{spec['id']}:{p['name']}:{j}")
                 for j, it in enumerate(p["items"])]
        panel(slide, x, y, w, h, p["name"], items, tone=p.get("tone", "accent"))
    source_line(slide, spec.get("sources"))


def build_tom_lanes(slide, spec, res):
    header(slide, spec)
    lanes = spec["lanes"]
    lw = Emu(int((CONTENT_W - (len(lanes) - 1) * GAP) / len(lanes)))
    lh = Inches(2.9)
    for i, lane in enumerate(lanes):
        x = Emu(int(MARGIN + i * (lw + GAP)))
        panel(slide, x, BAND_TOP, lw, lh, lane["name"], lane["items"],
              tone=lane.get("tone", "ink"), item_size=12)
    ty = Emu(int(BAND_TOP + lh + Inches(0.25)))
    th = Emu(int(BAND_BOT - ty))
    resp = spec["responsibilities"]
    gframe = slide.shapes.add_table(len(resp), 2, MARGIN, ty, CONTENT_W, th)
    tbl = gframe.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.columns[0].width = Inches(1.9)
    tbl.columns[1].width = Emu(int(CONTENT_W - Inches(1.9)))
    for i, (k, v) in enumerate(resp):
        fillc = WHITE if i % 2 else PANEL_BG
        set_cell(tbl.cell(i, 0), k, size=11.5, bold=True, color=ACCENT, fill=fillc)
        set_cell(tbl.cell(i, 1), v, size=11.5, fill=fillc)
        tbl.rows[i].height = Emu(int(th / len(resp)))
    source_line(slide, spec.get("sources"))


BUILDERS = {
    "stat_cards": build_stat_cards,
    "definition": build_definition,
    "before_after_band": build_before_after_band,
    "criteria": build_criteria,
    "scatter_chart": build_scatter_chart,
    "judgment_matrix": build_judgment_matrix,
    "h5_breakdown": build_h5_breakdown,
    "grid_2x2": build_grid_2x2,
    "tom_lanes": build_tom_lanes,
}


def render(slides, res, out_path):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]
    for i, spec in enumerate(slides, start=1):
        slide = prs.slides.add_slide(blank)
        BUILDERS[spec["layout"]](slide, spec, res)
        tf = textbox(slide, Emu(int(SLIDE_W - Inches(0.85))), Inches(7.06), Inches(0.5), Inches(0.3))
        r = tf.paragraphs[0].add_run()
        r.text = f"{i:02d}"
        rs(r, 9, color=MUTED)
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"wrote {out_path} ({len(slides)} slides)")


def main():
    content = yaml.safe_load(CONTENT_PATH.read_text())
    res = Resolver(content["meta"]["files"])
    global FONT
    FONT = content["meta"].get("font", FONT)
    render(content["slides"], res, DECK_PATH)
    render([content["tom"]], res, TOM_PATH)
    print()
    print(f"frozen literal audit ({len(res.frozen_audit)} entries):")
    for ctx, val, src in res.frozen_audit:
        print(f"  [{ctx}] {val}  <- {src}")


if __name__ == "__main__":
    main()
