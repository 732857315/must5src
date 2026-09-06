"""Self-contained SVG/HTML views of two 5x5 policy probability maps.

Quantization controls display color only. Probability labels always retain the
original probabilities, including when a map is scaled by its own maximum.
"""

from html import escape
from collections.abc import Mapping
import unicodedata

import numpy as np

from unet_codec import (
    normalize_grid, EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB,
    RED_RGB, GREEN_RGB,
)


EXAMPLE_STATUS = "仅显示示例，尚未训练"
_PALETTE = np.array([EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB], dtype=np.uint8)
_CELL_NAMES = ("空位", "黑棋", "白棋", "边界 / 禁下")


def _probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape not in ((25,), (5, 5)):
        raise ValueError("probabilities must have shape (25,) or (5, 5)")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    return values.reshape(25).copy()


def quantize_levels(probabilities, legal_mask, relative=False):
    """Return 25 integers: zero for zero/illegal cells, 1..25 for positive ones.

    Absolute colors use ceil(probability * 25). Relative colors first divide by
    the maximum probability among legal cells. This is not rank quantization.
    """
    probabilities = _probabilities(probabilities)
    legal_mask = np.asarray(legal_mask)
    if legal_mask.shape not in ((25,), (5, 5)) or not np.isin(legal_mask, (False, True)).all():
        raise ValueError("legal_mask must contain 25 boolean values")
    legal_mask = legal_mask.reshape(25).astype(bool)
    probabilities[~legal_mask] = 0
    maximum = probabilities.max()
    if relative and maximum > 0:
        probabilities /= maximum
    levels = np.zeros(25, dtype=np.uint8)
    positive = probabilities > 0
    # Remove floating-point noise at exact 1/25 boundaries; every positive value
    # still receives at least level 1, even when its displayed intensity is tiny.
    levels[positive] = np.clip(np.ceil(probabilities[positive] * 25 - 1e-6), 1, 25).astype(np.uint8)
    return levels


def quantized_rgb(state, probabilities, color, *, relative=False):
    """Return uint8 HWC display colors, preserving occupied/forbidden RGB values."""
    grid = normalize_grid(state)
    target = np.asarray(color, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 255):
        raise ValueError("color must be an RGB triple in [0, 255]")
    levels = quantize_levels(probabilities, grid == 0, relative)
    colors = _PALETTE[grid].copy().reshape(25, 3)
    legal = grid.reshape(25) == 0
    alpha = levels[legal, None] / 25
    colors[legal] = np.rint(np.asarray(EMPTY_RGB) * (1 - alpha) + target * alpha).astype(np.uint8)
    return colors.reshape(5, 5, 3)


def _hex(rgb):
    return "#" + "".join(f"{int(value):02x}" for value in rgb)


def _percent(value):
    return f"{float(value) * 100:.4g}%"


def _wrap_text(value, width=90):
    """Wrap mixed CJK/ASCII text by approximate glyph width without truncation."""
    lines, line, used = [], "", 0.0
    for char in str(value):
        size = 1.0 if unicodedata.east_asian_width(char) in "WF" else 0.56
        if char == "\n" or (line and used + size > width):
            lines.append(line)
            line, used = "", 0.0
            if char == "\n":
                continue
        line += char
        used += size
    lines.append(line)
    return lines


def _text_lines(lines, x, y, *, size=13, step=19, fill="#514f4a", weight=400):
    return "".join(
        f'<text x="{x}" y="{y + i * step}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        for i, line in enumerate(lines)
    )


def _stats_text(training_stats):
    if training_stats is None:
        return ""
    if isinstance(training_stats, Mapping):
        return "训练统计 · " + "   |   ".join(f"{key}：{value}" for key, value in training_stats.items())
    return "训练统计 · " + str(training_stats)


def _board_svg(grid, probabilities, color, *, x, y, name, heading, relative):
    probabilities = _probabilities(probabilities)
    legal = grid.reshape(25) == 0
    levels = quantize_levels(probabilities, legal, relative)
    colors = quantized_rgb(grid, probabilities, color, relative=relative).reshape(25, 3)
    maximum = float(probabilities[legal].max()) if legal.any() else 0
    cell, board_x, board_y = 76, x + 83, y + 107
    accent = _hex(color)
    parts = [
        f'<g data-board="{name}">',
        f'<rect x="{x}" y="{y}" width="545" height="598" rx="18" fill="#ffffff" stroke="#ded9ce"/>',
        f'<rect x="{x + 26}" y="{y + 27}" width="5" height="24" rx="2" fill="{accent}"/>',
        f'<text x="{x + 43}" y="{y + 45}" font-size="21" font-weight="700" fill="#232824">{escape(heading)}</text>',
        f'<text x="{x + 28}" y="{y + 69}" font-size="12" fill="#77766e">每格显示原始概率值 · 颜色强度分 25 级</text>',
    ]
    for column in range(5):
        parts.append(f'<text x="{board_x + column * cell + cell / 2}" y="{board_y - 12}" text-anchor="middle" font-size="12" fill="#77766e">{chr(65 + column)}</text>')
    for row in range(5):
        parts.append(f'<text x="{board_x - 18}" y="{board_y + row * cell + cell / 2 + 4}" text-anchor="middle" font-size="12" fill="#77766e">{row + 1}</text>')
    for index, value in enumerate(grid.reshape(25)):
        row, column = divmod(index, 5)
        left, top = board_x + column * cell, board_y + row * cell
        center_x, center_y = left + cell / 2, top + cell / 2
        level = int(levels[index])
        coordinate = f"{chr(65 + column)}{row + 1}"
        annotation = f"{coordinate}，{_CELL_NAMES[value]}"
        if value == 0:
            annotation += f"，原始概率 {_percent(probabilities[index])}，显示等级 {level}/25"
        else:
            annotation += "，不可落子，不着概率色"
        background = _hex(colors[index]) if value in (0, 3) else _hex(EMPTY_RGB)
        parts.extend([
            f'<g data-cell="{index}" data-state="{int(value)}" data-level="{level}">',
            f'<title>{escape(annotation)}</title>',
            f'<rect x="{left}" y="{top}" width="{cell}" height="{cell}" fill="{background}" stroke="#aca797" stroke-width="1"/>',
        ])
        if value in (1, 2):
            stone_color = _hex(BLACK_RGB if value == 1 else WHITE_RGB)
            label_color = "#ffffff" if value == 1 else _hex(BLACK_RGB)
            parts.append(f'<circle cx="{center_x}" cy="{center_y}" r="25" fill="{stone_color}" stroke="#333c40" stroke-width="1.6"/>')
            parts.append(f'<text x="{center_x}" y="{center_y + 5}" text-anchor="middle" font-size="14" font-weight="600" fill="{label_color}">{"黑" if value == 1 else "白"}</text>')
        elif value == 3:
            parts.append(f'<path d="M {center_x - 11} {center_y - 18} L {center_x + 11} {center_y + 4} M {center_x + 11} {center_y - 18} L {center_x - 11} {center_y + 4}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text x="{center_x}" y="{top + 62}" text-anchor="middle" font-size="11" fill="#ffffff"># 禁下</text>')
        else:
            foreground = "#ffffff" if name == "opponent" and level >= 16 else "#233528"
            if level == 0:
                foreground = "#979084"
            parts.append(f'<text x="{center_x}" y="{top + 35}" text-anchor="middle" font-size="16" font-weight="600" fill="{foreground}">{_percent(probabilities[index])}</text>')
            parts.append(f'<text x="{center_x}" y="{top + 54}" text-anchor="middle" font-size="10" fill="{foreground}">{level} / 25</text>')
        parts.append("</g>")
    legend_y = board_y + 415
    parts.append(f'<text x="{board_x}" y="{legend_y - 12}" font-size="11" fill="#64665f">25 级颜色强度</text>')
    for index in range(25):
        alpha = (index + 1) / 25
        rgb = np.rint(np.asarray(EMPTY_RGB) * (1 - alpha) + np.asarray(color) * alpha)
        parts.append(f'<rect data-legend-level="{index + 1}" x="{board_x + index * 15.2:.1f}" y="{legend_y}" width="15.2" height="16" fill="{_hex(rgb)}"><title>显示等级 {index + 1}/25</title></rect>')
    parts.append(f'<text x="{board_x}" y="{legend_y + 33}" font-size="11" fill="#64665f">1（浅）</text>')
    parts.append(f'<text x="{board_x + 380}" y="{legend_y + 33}" text-anchor="end" font-size="11" fill="#64665f">25（深）</text>')
    legend_caption = f"相对本图最大值：{_percent(maximum)} → 25 级" if relative else "绝对概率：0% 无色，100% → 25 级"
    if relative and maximum == 0:
        legend_caption = "本图无正概率：所有空位均为 0 级"
    parts.append(f'<text x="{x + 272.5}" y="{y + 578}" text-anchor="middle" font-size="12" fill="#575f53">{escape(legend_caption)}</text>')
    parts.append("</g>")
    return "".join(parts)


def render_svg(state, opponent_probabilities, play_probabilities, *,
               title="5×5 双 U-Net 策略图", subtitle="红色：对手落点概率 · 绿色：己方推荐概率",
               status=EXAMPLE_STATUS, checkpoint_source=None, training_stats=None, relative=False):
    """Return a standalone paired-board SVG; status/source are caller supplied."""
    grid = normalize_grid(state)
    opponent_probabilities = _probabilities(opponent_probabilities)
    play_probabilities = _probabilities(play_probabilities)
    title_lines = _wrap_text(title, 39)
    subtitle_lines = _wrap_text(subtitle, 84)
    status_lines = _wrap_text(status, 77)
    source = f"Checkpoint 来源：{checkpoint_source}" if checkpoint_source else "Checkpoint 来源：未提供；默认图为示例，不代表已训练预测"
    source_lines = _wrap_text(source, 94)
    stats = _stats_text(training_stats)
    stats_lines = _wrap_text(stats, 94) if stats else []
    subtitle_y = 49 + len(title_lines) * 32
    status_y = subtitle_y + len(subtitle_lines) * 19 + 4
    status_height = 22 + len(status_lines) * 20
    source_y = status_y + status_height + 23
    stats_y = source_y + len(source_lines) * 19 + 3
    board_y = stats_y + len(stats_lines) * 19 + 19
    height = board_y + 656
    mode = "相对本图最大值" if relative else "绝对概率"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 {height}" width="1180" height="{height}" role="img" aria-label="{escape(str(title), quote=True)}">',
        f'<title>{escape(str(title))}</title>',
        f'<desc>{escape(str(status))}。{escape(source)}。显示方式：{mode}；每格百分比为原始概率。</desc>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC","Segoe UI",sans-serif}</style>',
        f'<rect width="1180" height="{height}" fill="#f4f1eb"/>',
        _text_lines(title_lines, 36, 47, size=28, step=32, fill="#202d25", weight=700),
        _text_lines(subtitle_lines, 36, subtitle_y, size=13),
        f'<rect x="36" y="{status_y}" width="1108" height="{status_height}" rx="10" fill="#fff3d4" stroke="#dfc892"/>',
        _text_lines(status_lines, 54, status_y + 27, size=15, step=20, fill="#765117", weight=700),
        _text_lines(source_lines, 38, source_y, size=12),
        _text_lines(stats_lines, 38, stats_y, size=12),
        _board_svg(grid, opponent_probabilities, RED_RGB, x=36, y=board_y,
                   name="opponent", heading="对手下一步 · 红色", relative=relative),
        _board_svg(grid, play_probabilities, GREEN_RGB, x=599, y=board_y,
                   name="play", heading="己方推荐 · 绿色", relative=relative),
        f'<text x="36" y="{board_y + 630}" font-size="12" fill="#676960">显示方式：{mode}。百分比始终为原始概率；黑 / 白棋和 # 禁下格不着概率色。</text>',
        '</svg>',
    ]
    return "".join(parts)


def _probability_table(grid, opponent_probabilities, play_probabilities):
    legal = grid.reshape(25) == 0
    probabilities = (_probabilities(opponent_probabilities), _probabilities(play_probabilities))
    absolute = [quantize_levels(p, legal) for p in probabilities]
    relative = [quantize_levels(p, legal, True) for p in probabilities]
    rows = []
    for index, value in enumerate(grid.reshape(25)):
        row, column = divmod(index, 5)
        coordinate = f"{chr(65 + column)}{row + 1}"
        if value == 0:
            data = [_percent(p[index]) for p in probabilities]
            data += [f"{a[index]} / {r[index]}" for a, r in zip(absolute, relative)]
        else:
            data = ["—", "—", "0 / 0", "0 / 0"]
        cells = [coordinate, _CELL_NAMES[value], *data]
        rows.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>")
    return "".join(rows)


def render_html(state, opponent_probabilities, play_probabilities, *,
                title="5×5 双 U-Net 策略图", subtitle="红色：对手落点概率 · 绿色：己方推荐概率",
                status=EXAMPLE_STATUS, checkpoint_source=None, training_stats=None):
    """Return an offline Chinese report with an absolute/relative color switch."""
    grid = normalize_grid(state)
    # Validate once before constructing either view and the numeric table.
    opponent_probabilities, play_probabilities = _probabilities(opponent_probabilities), _probabilities(play_probabilities)
    options = dict(title=title, subtitle=subtitle, status=status,
                   checkpoint_source=checkpoint_source, training_stats=training_stats)
    absolute = render_svg(grid, opponent_probabilities, play_probabilities, **options)
    relative = render_svg(grid, opponent_probabilities, play_probabilities, relative=True, **options)
    table = _probability_table(grid, opponent_probabilities, play_probabilities)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(str(title))}</title>
<style>
:root{{font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:#26372c;background:#e8e6de;color-scheme:light}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1240px;margin:24px auto;padding:0 20px 32px}}
.toolbar{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;background:#243e30;color:#fff;padding:18px 24px;border-radius:16px 16px 0 0}}
.brand{{font-size:12px;letter-spacing:2px;font-weight:700}}fieldset{{margin:0;border:0;padding:0;display:flex;flex-wrap:wrap;gap:10px}}legend{{float:left;margin:6px 12px 0 0;font-size:12px;color:#d2dccb}}
label{{padding:8px 12px;border:1px solid #708773;border-radius:8px;font-size:12px;cursor:pointer}}label:has(input:checked){{background:#ebf2e5;color:#253f2d;border-color:#ebf2e5}}input{{accent-color:#447447;vertical-align:middle}}
.note{{margin:0;background:#fff;padding:15px 24px;font-size:12px;line-height:1.8;color:#65705f;border-bottom:1px solid #e5e3d9}}
.graphic{{background:#f4f1eb;overflow:auto}}.graphic svg{{display:block;width:100%;height:auto;min-width:700px}}[hidden]{{display:none!important}}
details{{background:white;border-radius:0 0 16px 16px;padding:20px 24px}}summary{{font-size:13px;cursor:pointer;font-weight:700}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:18px}}th,td{{border-bottom:1px solid #e4e8de;padding:10px;text-align:left;white-space:nowrap}}th{{background:#f3f6ef}}.fine{{font-size:12px;line-height:1.8;color:#77806e}}
@media(max-width:600px){{main{{padding:0;margin:0}}.toolbar{{border-radius:0;padding:18px}}.note{{padding:12px 18px}}details{{border-radius:0;padding:18px}}}}
</style></head><body><main>
<div class="toolbar"><span class="brand">5×5 / DUAL U-NET</span><fieldset aria-label="颜色强度显示方式"><legend>颜色显示</legend>
<label><input type="radio" name="scale" value="absolute" checked> 绝对概率</label>
<label><input type="radio" name="scale" value="relative"> 相对本图最大值</label></fieldset></div>
<p class="note">每格百分比始终保留输入的原始概率值。是否来自已训练模型，以图中的状态和 Checkpoint 来源为准。25 级仅控制颜色强度；相对模式将每张图的最高合法概率映射到第 25 级，不改变概率、不按排名着色。将鼠标停在格子上可查看坐标、原始概率与显示等级。</p>
<section class="graphic" data-mode="absolute" aria-label="绝对概率视图">{absolute}</section>
<section class="graphic" data-mode="relative" aria-label="相对本图最大值视图" hidden>{relative}</section>
<details><summary>查看 25 格原始数值与显示等级</summary><p class="fine">等级列依次为“绝对 / 相对”，范围为 0–25。0 表示零概率或不可落子；1–25 表示由浅到深。黑白棋和灰色禁下格保留原有标记。</p>
<div class="table-wrap"><table><thead><tr><th>坐标</th><th>格子状态</th><th>对手概率</th><th>己方概率</th><th>红色等级（绝对 / 相对）</th><th>绿色等级（绝对 / 相对）</th></tr></thead><tbody>{table}</tbody></table></div></details>
</main><script>
document.querySelectorAll('input[name="scale"]').forEach(function(input){{
  input.addEventListener('change',function(){{
    document.querySelectorAll('[data-mode]').forEach(function(panel){{panel.hidden=panel.dataset.mode!==input.value;}});
  }});
}});
</script></body></html>'''


def render_png(state, opponent_probabilities, play_probabilities, *,
               title="5×5 双 U-Net 策略图", subtitle="红色：对手落点概率 · 绿色：己方推荐概率",
               status=EXAMPLE_STATUS, checkpoint_source=None, training_stats=None,
               relative=False, scale=1.5, font_path=None):
    """Rasterize this module's own SVG primitives to PNG bytes using Pillow.

    This supports only the shapes emitted above, with no browser or network.
    """
    from io import BytesIO
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw, ImageFont

    if not np.isfinite(scale) or not 0.25 <= scale <= 4:
        raise ValueError("scale must be between 0.25 and 4")
    svg = render_svg(state, opponent_probabilities, play_probabilities, title=title,
                     subtitle=subtitle, status=status, checkpoint_source=checkpoint_source,
                     training_stats=training_stats, relative=relative)
    root = ET.fromstring(svg)
    image = Image.new("RGB", (round(float(root.attrib["width"]) * scale),
                              round(float(root.attrib["height"]) * scale)), "white")
    draw = ImageDraw.Draw(image)
    if font_path is not None:
        regular = bold = str(font_path)
    else:
        candidates = (
            ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"),
            ("C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simhei.ttf"),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc"),
        )
        pair = next(((a, b) for a, b in candidates if Path(a).is_file()), None)
        if pair is None:
            raise RuntimeError("A Chinese font is required for PNG rendering; pass font_path")
        regular, bold = pair
        if not Path(bold).is_file():
            bold = regular
    fonts = {}

    def number(node, name, default=0):
        return float(node.attrib.get(name, default)) * scale

    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        fill = node.attrib.get("fill", "black")
        stroke = node.attrib.get("stroke")
        stroke_width = max(1, round(number(node, "stroke-width", 1)))
        if tag == "rect":
            x, y = number(node, "x"), number(node, "y")
            box = (x, y, x + number(node, "width"), y + number(node, "height"))
            radius = number(node, "rx")
            if radius:
                draw.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=stroke_width)
            else:
                draw.rectangle(box, fill=fill, outline=stroke, width=stroke_width)
        elif tag == "circle":
            x, y, radius = number(node, "cx"), number(node, "cy"), number(node, "r")
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=fill, outline=stroke, width=stroke_width)
        elif tag == "path":
            tokens = node.attrib["d"].split()
            previous = None
            for index in range(0, len(tokens), 3):
                command = tokens[index]
                point = (float(tokens[index + 1]) * scale, float(tokens[index + 2]) * scale)
                if command == "L" and previous is not None:
                    draw.line((previous, point), fill=stroke, width=stroke_width)
                previous = point
        elif tag == "text":
            size = max(1, round(number(node, "font-size", 13)))
            weight = int(node.attrib.get("font-weight", 400))
            key = (size, weight >= 600)
            if key not in fonts:
                fonts[key] = ImageFont.truetype(bold if key[1] else regular, size)
            anchor = {"start": "ls", "middle": "ms", "end": "rs"}[node.attrib.get("text-anchor", "start")]
            draw.text((number(node, "x"), number(node, "y")), node.text or "",
                      font=fonts[key], fill=fill, anchor=anchor)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
