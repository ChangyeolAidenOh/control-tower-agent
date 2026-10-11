"""Stage 7 deck generator (v0 skeleton).

Reads deck/deck_content.yaml and renders outputs/deck/stage7_deck.pptx plus
outputs/deck/tom.pptx with python-pptx. Numbers are injected through a
resolver: {ref: "file_key:dotted.path"} reads from the frozen JSONs declared
in meta.files; {frozen: "...", source: "..."} is a frozen literal whose JSON
pointer is not wired yet. Every frozen literal is printed as an audit list at
the end of the build -- the target is to convert them to refs over time.

Implemented builders: stat_cards (S1), scatter_chart (S5, native XY chart
with frontier line, agent marker, A5 reference, CSL 90% guide), generic
(title + headline + body + footnotes + sources) for everything else.
Upgrade targets noted in the yaml (before_after_band, judgment_matrix,
h5_breakdown_bar, three_lane_tom) are the next iteration.

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
from pptx.util import Emu, Inches, Pt

CONTENT_PATH = Path("deck/deck_content.yaml")
OUT_DIR = Path("outputs/deck")
DECK_PATH = OUT_DIR / "stage7_deck.pptx"
TOM_PATH = OUT_DIR / "tom.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)

COLORS = {
    "ink": RGBColor(0x1A, 0x1A, 0x2E),
    "muted": RGBColor(0x5A, 0x5A, 0x6E),
    "accent": RGBColor(0x0F, 0x62, 0xFE),
    "warn": RGBColor(0xB4, 0x23, 0x18),
    "ok": RGBColor(0x1A, 0x7F, 0x4B),
    "line": RGBColor(0xC9, 0xC9, 0xD4),
    "bg_card": RGBColor(0xF2, 0xF4, 0xF8),
}

POLICY_COLORS = {
    "b": RGBColor(0xB4, 0x23, 0x18),
    "a4": RGBColor(0x0F, 0x62, 0xFE),
    "a4_sl": RGBColor(0x08, 0x47, 0xB0),
    "apto": RGBColor(0x6F, 0x2D, 0xBD),
    "l0_rule_ct": RGBColor(0x1A, 0x1A, 0x2E),
    "a1": RGBColor(0x9A, 0x9A, 0xAA),
    "a2": RGBColor(0x9A, 0x9A, 0xAA),
    "a3": RGBColor(0x9A, 0x9A, 0xAA),
    "a5": RGBColor(0x1A, 0x7F, 0x4B),
    "agent": RGBColor(0xE0, 0x7B, 0x00),
}


class Resolver:
    """Resolve yaml value specs; collect an audit trail of frozen literals."""

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


def set_font(run, size, bold=False, color=COLORS["ink"], font_name=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if font_name:
        run.font.name = font_name


def add_textbox(slide, x, y, w, h):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    return box, tf


def add_title_block(slide, title, font_name):
    box, tf = add_textbox(slide, MARGIN, Inches(0.35), SLIDE_W - 2 * MARGIN, Inches(0.6))
    run = tf.paragraphs[0].add_run()
    run.text = title
    set_font(run, 24, bold=True, font_name=font_name)
    line = slide.shapes.add_shape(1, MARGIN, Inches(1.0), SLIDE_W - 2 * MARGIN, Emu(1))
    line.fill.solid()
    line.fill.fore_color.rgb = COLORS["line"]
    line.line.fill.background()


def add_headline(slide, text, font_name, y=Inches(1.15), h=Inches(0.95)):
    box, tf = add_textbox(slide, MARGIN, y, SLIDE_W - 2 * MARGIN, h)
    run = tf.paragraphs[0].add_run()
    run.text = text
    set_font(run, 14, bold=True, color=COLORS["muted"], font_name=font_name)


def add_source_line(slide, sources, font_name):
    if not sources:
        return
    box, tf = add_textbox(slide, MARGIN, SLIDE_H - Inches(0.45), SLIDE_W - 2 * MARGIN, Inches(0.3))
    run = tf.paragraphs[0].add_run()
    run.text = "출처: " + " · ".join(sources)
    set_font(run, 9, color=COLORS["muted"], font_name=font_name)


def add_footnotes(slide, notes, font_name, y):
    if not notes:
        return
    box, tf = add_textbox(slide, MARGIN, y, SLIDE_W - 2 * MARGIN, SLIDE_H - y - Inches(0.5))
    first = True
    for n in notes:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = "주) " + n
        set_font(run, 9.5, color=COLORS["muted"], font_name=font_name)


def build_generic(slide, spec, res, font_name):
    add_title_block(slide, spec["title"], font_name)
    y = Inches(1.15)
    if spec.get("headline"):
        add_headline(slide, spec["headline"], font_name, y=y)
        y = Inches(2.15)
    body = spec.get("body", [])
    if body:
        box, tf = add_textbox(slide, MARGIN, y, SLIDE_W - 2 * MARGIN, Inches(4.0))
        first = True
        for item in body:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_after = Pt(6)
            if isinstance(item, dict):
                ctx = f"{spec['id']}:{item['key']}"
                text = f"{item['key']}: {res.value(item['value'], ctx)}"
            else:
                text = str(item)
            run = p.add_run()
            run.text = "— " + text
            set_font(run, 13, font_name=font_name)
    add_footnotes(slide, spec.get("footnotes"), font_name, Inches(6.1))
    add_source_line(slide, spec.get("sources"), font_name)


def build_stat_cards(slide, spec, res, font_name):
    add_title_block(slide, spec["title"], font_name)
    add_headline(slide, spec["headline"], font_name, y=Inches(1.15), h=Inches(1.1))
    stats = spec["stats"]
    n = len(stats)
    gap = Inches(0.35)
    card_w = Emu(int((SLIDE_W - 2 * MARGIN - (n - 1) * gap) / n))
    card_h = Inches(2.6)
    y = Inches(2.6)
    for i, st in enumerate(stats):
        x = Emu(int(MARGIN + i * (card_w + gap)))
        card = slide.shapes.add_shape(5, x, y, card_w, card_h)  # ROUNDED_RECTANGLE
        card.fill.solid()
        card.fill.fore_color.rgb = COLORS["bg_card"]
        card.line.color.rgb = COLORS["line"]
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = st["label"]
        set_font(r0, 12, bold=True, color=COLORS["muted"], font_name=font_name)
        p1 = tf.add_paragraph()
        p1.space_before = Pt(10)
        r1 = p1.add_run()
        r1.text = res.value(st["value"], f"{spec['id']}:{st['label']}:value")
        color = COLORS["warn"] if "−" in r1.text or "미달" in st.get("sub", {}).get("frozen", "") else COLORS["ink"]
        set_font(r1, 30, bold=True, color=color, font_name=font_name)
        p2 = tf.add_paragraph()
        p2.space_before = Pt(8)
        r2 = p2.add_run()
        r2.text = res.value(st["sub"], f"{spec['id']}:{st['label']}:sub")
        set_font(r2, 11, color=COLORS["muted"], font_name=font_name)
    add_source_line(slide, spec.get("sources"), font_name)


def build_scatter_chart(slide, spec, res, font_name):
    add_title_block(slide, spec["title"], font_name)
    add_headline(slide, spec["headline"], font_name)
    cfg = spec["chart"]
    policies = res.get_path(cfg["points_ref"])
    frontier = res.get_path(cfg["frontier_ref"])
    agent = res.get_path(cfg["agent_ref"])
    refs = set(res.get_path(cfg["reference_policies_ref"]))

    data = XyChartData()
    x_vals = [v["mean_episode_cost"] for v in policies.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    pad = (x_max - x_min) * 0.06

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
    s_agent = data.add_series("Agent (ST8-02)")
    s_agent.add_data_point(agent["mean_episode_cost"], agent["pooled_sku_day_csl"])
    series_keys.append("agent")

    gframe = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
        MARGIN, Inches(2.1), Inches(8.6), Inches(4.3), data,
    )
    chart = gframe.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.RIGHT
    chart.legend.include_in_layout = False
    plot = chart.plots[0]
    # series 0 = frontier (line, no markers), 1 = guide line, rest = single-point markers
    for idx, series in enumerate(plot.series):
        fmt = series.format.line
        if idx == 0:
            fmt.color.rgb = COLORS["accent"]
            fmt.width = Pt(1.75)
        elif idx == 1:
            fmt.color.rgb = COLORS["line"]
            fmt.width = Pt(1.0)
        else:
            fmt.fill.background()
            key = series_keys[idx - 2]
            series.marker.style = 8  # circle
            series.marker.size = 9 if key == "agent" else 7
            series.marker.format.fill.solid()
            series.marker.format.fill.fore_color.rgb = POLICY_COLORS.get(key, COLORS["muted"])
    vax = chart.value_axis
    vax.minimum_scale = 0.3
    vax.maximum_scale = 1.0
    vax.has_major_gridlines = True

    box, tf = add_textbox(slide, Inches(9.4), Inches(2.1), Inches(3.3), Inches(4.3))
    first = True
    for note in spec.get("talk_notes", []):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = "· " + note
        set_font(run, 10.5, font_name=font_name)
    add_footnotes(slide, spec.get("footnotes"), font_name, Inches(6.5))
    add_source_line(slide, spec.get("sources"), font_name)


BUILDERS = {
    "generic": build_generic,
    "stat_cards": build_stat_cards,
    "scatter_chart": build_scatter_chart,
}


def new_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def render(slides, res, font_name, out_path):
    prs = new_presentation()
    blank = prs.slide_layouts[6]
    for spec in slides:
        slide = prs.slides.add_slide(blank)
        BUILDERS[spec.get("layout", "generic")](slide, spec, res, font_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"wrote {out_path} ({len(slides)} slides)")


def main():
    content = yaml.safe_load(CONTENT_PATH.read_text())
    meta = content["meta"]
    res = Resolver(meta["files"])
    font_name = meta.get("font", "Malgun Gothic")

    render(content["slides"], res, font_name, DECK_PATH)
    render([content["tom"]], res, font_name, TOM_PATH)

    print()
    print(f"frozen literal audit ({len(res.frozen_audit)} entries -- convert to refs over time):")
    for ctx, val, src in res.frozen_audit:
        print(f"  [{ctx}] {val}  <- {src}")


if __name__ == "__main__":
    main()
