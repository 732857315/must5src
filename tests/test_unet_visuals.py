from io import BytesIO
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from unet_codec import EMPTY_RGB, BLACK_RGB, WHITE_RGB, FORBIDDEN_RGB, RED_RGB, GREEN_RGB
from unet_visuals import quantize_levels, quantized_rgb, render_svg, render_html, render_png


class DisplayQuantizationTests(unittest.TestCase):
    def test_all_25_levels_are_monotonic_and_zero_is_uncolored(self):
        probabilities = np.arange(1, 26) / 25
        levels = quantize_levels(probabilities, np.ones(25, dtype=bool))
        np.testing.assert_array_equal(levels, np.arange(1, 26))
        np.testing.assert_array_equal(quantize_levels(np.zeros(25), np.ones(25)), np.zeros(25))
        probabilities[0] = 1e-20
        self.assertEqual(quantize_levels(probabilities, np.ones(25))[0], 1)

    def test_relative_color_uses_legal_maximum_and_keeps_equal_values_equal(self):
        probabilities = np.zeros(25)
        probabilities[:4] = (0.01, 0.01, 0.02, 1)
        legal = np.ones(25, dtype=bool)
        legal[3] = False
        np.testing.assert_array_equal(quantize_levels(probabilities, legal, True)[:4], [13, 13, 25, 0])
        np.testing.assert_array_equal(probabilities[:4], [0.01, 0.01, 0.02, 1])

    def test_illegal_cells_are_zero_even_with_large_probabilities(self):
        legal = np.zeros(25, dtype=bool)
        np.testing.assert_array_equal(quantize_levels(np.ones(25), legal, True), np.zeros(25))

    def test_probabilities_are_validated_before_quantization(self):
        for probabilities in (np.zeros(24), np.full(25, -0.01), np.full(25, 1.01),
                              np.full(25, np.nan), np.full(25, np.inf)):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                quantize_levels(probabilities, np.ones(25))
        with self.assertRaises(ValueError):
            quantize_levels(np.zeros(25), np.full(25, 2))

    def test_quantized_rgb_preserves_stones_and_forbidden_cells_exactly(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[0, :4] = (0, 1, 2, 3)
        probabilities = np.ones(25)
        for color in (RED_RGB, GREEN_RGB):
            for relative in (False, True):
                rgb = quantized_rgb(grid, probabilities, color, relative=relative)
                self.assertEqual(rgb.dtype, np.uint8)
                self.assertEqual(rgb.shape, (5, 5, 3))
                np.testing.assert_array_equal(rgb[0, 0], color)
                np.testing.assert_array_equal(rgb[0, 1], BLACK_RGB)
                np.testing.assert_array_equal(rgb[0, 2], WHITE_RGB)
                np.testing.assert_array_equal(rgb[0, 3], FORBIDDEN_RGB)
        zero = quantized_rgb(grid, np.zeros(25), RED_RGB)
        np.testing.assert_array_equal(zero[0, 0], EMPTY_RGB)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((5, 5), dtype=np.uint8)
        self.grid[0, :3] = (1, 2, 3)
        self.opponent = np.linspace(0, 0.12, 25)
        self.play = np.linspace(0.12, 0, 25)

    def test_svg_has_two_boards_25_level_legends_and_unambiguous_pieces(self):
        root = ET.fromstring(render_svg(self.grid, self.opponent, self.play))
        boards = [node for node in root.iter() if "data-board" in node.attrib]
        self.assertEqual(len(boards), 2)
        legend = [node for node in root.iter() if "data-legend-level" in node.attrib]
        self.assertEqual(len(legend), 50)
        for board in boards:
            cells = [node for node in board if "data-cell" in node.attrib]
            self.assertEqual(len(cells), 25)
            for cell in cells[:3]:
                self.assertEqual(cell.attrib["data-level"], "0")
                self.assertNotIn("%", "".join(cell.itertext()))
            white = [node for node in cells[1] if node.tag.endswith("circle")][0]
            black = [node for node in cells[0] if node.tag.endswith("circle")][0]
            self.assertEqual(white.attrib["fill"], "#ffffff")
            self.assertEqual(black.attrib["fill"], "#181b21")
            self.assertIn("白", "".join(cells[1].itertext()))
            self.assertIn("# 禁下", "".join(cells[2].itertext()))

    def test_default_status_never_claims_trained_predictions(self):
        svg = render_svg(self.grid, self.opponent, self.play)
        self.assertIn("仅显示示例，尚未训练", svg)
        self.assertIn("未提供", svg)

    def test_checkpoint_and_training_statistics_are_visible_and_escaped(self):
        malicious = '<script>alert("x")</script>'
        svg = render_svg(self.grid, self.opponent, self.play, title=malicious,
                         status="已加载训练 checkpoint", checkpoint_source="model<&>.pt",
                         training_stats={"阶段": "三阶段", "损失": 0.02})
        ET.fromstring(svg)
        self.assertNotIn(malicious, svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("model&lt;&amp;&gt;.pt", svg)
        self.assertIn("训练统计", svg)
        self.assertIn("三阶段", svg)

    def test_relative_mode_keeps_original_probability_labels(self):
        svg = render_svg(self.grid, self.opponent, self.play, relative=True)
        self.assertIn("相对本图最大值", svg)
        self.assertIn("12%", svg)
        self.assertNotIn(">100%</text>", svg)

    def test_html_is_offline_and_user_text_cannot_create_html_or_script(self):
        malicious = '</script><img src="https://example.com" onerror="alert(1)">'
        html = render_html(self.grid, self.opponent, self.play, title=malicious,
                           subtitle=malicious, status=malicious, checkpoint_source=malicious)
        self.assertNotIn(malicious, html)
        self.assertNotIn('<img src=', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link ', html)
        self.assertIn('data-mode="absolute"', html)
        self.assertIn('data-mode="relative"', html)
        self.assertIn("相对本图最大值", html)
        self.assertIn("不按排名着色", html)
        self.assertEqual(html.count("<script>"), 1)

    def test_packed_state_matches_grid(self):
        packed = sum(int(value) << (2 * i) for i, value in enumerate(self.grid.reshape(25)))
        self.assertEqual(render_svg(packed, self.opponent, self.play),
                         render_svg(self.grid, self.opponent, self.play))

    def test_all_forbidden_window_has_no_probability_coloring(self):
        root = ET.fromstring(render_svg(np.full((5, 5), 3), np.ones(25), np.ones(25), relative=True))
        cells = [node for node in root.iter() if "data-cell" in node.attrib]
        self.assertTrue(all(node.attrib["data-level"] == "0" for node in cells))
        self.assertIn("本图无正概率", "".join(root.itertext()))

    def test_png_uses_same_report_geometry(self):
        from PIL import Image
        try:
            content = render_png(self.grid, self.opponent, self.play, scale=0.5)
        except RuntimeError as exc:
            if "Chinese font" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))
        png = Image.open(BytesIO(content))
        root = ET.fromstring(render_svg(self.grid, self.opponent, self.play))
        self.assertEqual(png.size, (590, round(float(root.attrib["height"]) / 2)))
        self.assertEqual(png.mode, "RGB")
        with self.assertRaises(ValueError):
            render_png(self.grid, self.opponent, self.play, scale=0)


if __name__ == "__main__":
    unittest.main()
