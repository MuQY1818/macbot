"""Generate the README hero figure: a pixel-art scene of the macbot loop (light and dark SVG).

    python tools/brand/make_scene.py --font path/to/Monocraft.ttf   # writes assets/scene.svg, assets/scene-dark.svg

The arms, camera, MacBook, dataset and GPU are drawn on a pixel grid in the palette of the logo. Labels are
set in Monocraft (SIL Open Font License) and converted to paths on the same grid, so the SVG has no font
dependency and renders the same on GitHub, in browsers and in rsvg.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PX = 5  # one art pixel

THEMES = {
    "light": dict(K="#1B1A2E", S="#2B2A4C", s="#35345E", O="#FF7A1A", Y="#FFC93C", T="#2EC4B6", t="#1F8C83",
                  B="#C9C3B6", b="#9E978A", G="#3BB273", W="#FFFFFF", L="#E9E5DB", D="#8A8796", R="#FF4D4D",
                  text="#1B1A2E", muted="#8A8796", arrow_mac="#FF7A1A", arrow_gpu="#9E978A"),
    "dark": dict(K="#8E8BAE", S="#2B2A4C", s="#35345E", O="#FF7A1A", Y="#FFC93C", T="#2EC4B6", t="#1F8C83",
                 B="#C9C3B6", b="#9E978A", G="#3BB273", W="#F3F1EA", L="#3A3946", D="#A9A6B8", R="#FF4D4D",
                 text="#ECEBF0", muted="#A9A6B8", arrow_mac="#FF7A1A", arrow_gpu="#8A8796"),
}


class Canvas:
    def __init__(self, theme):
        self.t = theme
        self.cells = {}  # (cx, cy) -> color key

    def put(self, x, y, c):
        self.cells[(x, y)] = c

    def box(self, x0, y0, x1, y1, c):  # inclusive cell rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.put(x, y, c)

    def sprite(self, x0, y0, art, keymap=None):
        for dy, row in enumerate(art.strip("\n").splitlines()):
            for dx, ch in enumerate(row):
                if ch != ".":
                    self.put(x0 + dx, y0 + dy, (keymap or {}).get(ch, ch))

    def line(self, x0, y0, x1, y1, c, w=2):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            x = round(x0 + (x1 - x0) * i / n)
            y = round(y0 + (y1 - y0) * i / n)
            self.box(x, y, x + w - 1, y + w - 1, c)

    def svg_rects(self):
        out, rows = [], {}
        for (x, y), c in self.cells.items():
            rows.setdefault(y, {})[x] = c
        for y in sorted(rows):
            xs = sorted(rows[y])
            i = 0
            while i < len(xs):
                x, c = xs[i], rows[y][xs[i]]
                j = i
                while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1 and rows[y][xs[j + 1]] == c:
                    j += 1
                out.append(f'<rect x="{x * PX}" y="{y * PX}" width="{(j - i + 1) * PX}" height="{PX}" '
                           f'fill="{self.t[c]}"/>')
                i = j + 1
        return out


# ---------------------------------------------------------------- sprites (cell units)

def arm(cv, bx, by, joints, link, joint="Y", scale=2, gripper=True):
    """Base at (bx, by) = bottom-centre; joints are offsets from the shoulder, in cells."""
    cv.box(bx - 6, by - 1, bx + 6, by, "K")          # foot
    cv.box(bx - 5, by - 3, bx + 5, by - 2, "D")
    cv.box(bx - 3, by - 5, bx + 3, by - 4, "K")
    pts = [(bx - 1, by - 6)] + [(bx - 1 + dx, by - 6 + dy) for dx, dy in joints]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv.line(x0, y0, x1, y1, link, w=scale)
    for x, y in pts[:-1]:
        cv.box(x - 1, y - 1, x + scale, y + scale, "K")
        cv.box(x, y, x + scale - 1, y + scale - 1, joint)
    if gripper:
        x, y = pts[-1]
        cv.box(x - 1, y, x + 2, y + 1, "K")
        cv.box(x - 2, y + 2, x - 1, y + 5, "K")
        cv.box(x + 2, y + 2, x + 3, y + 5, "K")


CAMERA = """
.KKKKKKKK.
KDDDDDDDDK
KDDKKKKDDK
KDKTTTTKDK
KDKTWTTKDK
KDKTTTTKDK
KDDKKKKDDK
KDDDDDDDRK
.KKKKKKKK.
"""

GPU = """
KKKKKKKKKKKKKKKKK
KLLLLLLLLLLLLLLLK
KLKKKKKKKKKKKKKLK
KLKSSSSSSSSSGGKLK
KLKKKKKKKKKKKKKLK
KLKSSSSSSSSSGGKLK
KLKKKKKKKKKKKKKLK
KLKSSSSSSSSSGGKLK
KLKKKKKKKKKKKKKLK
KLKSSSSSSSSSGGKLK
KLKKKKKKKKKKKKKLK
KLKsssssssssssKLK
KLKsssssssssssKLK
KLKKKKKKKKKKKKKLK
KLLLLLLLLLLLLLLLK
KKKKKKKKKKKKKKKKK
.KK...........KK.
"""


def frame(cv, x, y):
    """One recorded camera frame: 14 x 10 cells."""
    cv.box(x, y, x + 13, y + 9, "K")
    cv.box(x + 1, y + 1, x + 12, y + 8, "W")
    cv.box(x + 2, y + 2, x + 11, y + 7, "t")
    cv.box(x + 2, y + 6, x + 11, y + 7, "T")
    cv.box(x + 8, y + 4, x + 9, y + 5, "O")
    cv.box(x + 4, y + 3, x + 5, y + 5, "Y")


def dataset(cv, x, y):
    for k in range(3):
        frame(cv, x + 2 * (2 - k), y + 2 * k)


def macbook(cv, x, y):
    w, h = 54, 32
    cv.box(x, y, x + w - 1, y + h - 1, "K")
    cv.box(x + 2, y + 2, x + w - 3, y + h - 3, "S")
    for r in range(y + 3, y + h - 2, 2):
        cv.box(x + 2, r, x + w - 3, r, "s")
    # console: two camera tiles and joint bars
    cv.box(x + 4, y + 4, x + 5, y + 5, "R")
    cv.box(x + 7, y + 4, x + 16, y + 5, "D")
    for tx in (x + 4, x + 28):
        cv.box(tx, y + 8, tx + 21, y + 21, "K")
        cv.box(tx + 1, y + 9, tx + 20, y + 20, "t")
        cv.box(tx + 1, y + 17, tx + 20, y + 20, "T")
    cv.box(x + 15, y + 14, x + 17, y + 16, "O")       # cube seen from the top camera
    cv.box(x + 9, y + 11, x + 12, y + 12, "Y")
    cv.box(x + 36, y + 12, x + 41, y + 17, "O")       # cube seen from the wrist camera
    cv.box(x + 30, y + 18, x + 31, y + 20, "K")
    cv.box(x + 46, y + 18, x + 47, y + 20, "K")
    for k in range(3):
        bx = x + 4 + k * 15
        cv.box(bx, y + 24, bx + 12, y + 24, "D")
        cv.box(bx + 3 + 3 * k, y + 23, bx + 4 + 3 * k, y + 25, "O")
    # deck
    cv.box(x - 4, y + h, x + w + 3, y + h, "K")
    cv.box(x - 4, y + h + 1, x + w + 3, y + h + 2, "B")
    cv.box(x - 4, y + h + 3, x + w + 3, y + h + 3, "b")
    cv.box(x - 3, y + h + 4, x + w + 2, y + h + 4, "K")
    cv.box(x + w // 2 - 5, y + h + 1, x + w // 2 + 4, y + h + 1, "b")


def dotted(cv, pts, c, head=True):
    """Orthogonal dotted path through cell points, with a pixel arrowhead at the end."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(abs(x1 - x0), abs(y1 - y0))
        sx, sy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
        for i in range(0, n + 1):
            if i % 3 != 2:
                cv.box(x0 + sx * i, y0 + sy * i, x0 + sx * i + 1, y0 + sy * i + 1, c)
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        sx, sy = (x1 > x0) - (x1 < x0), (y1 > y0) - (y1 < y0)
        for k in range(4):
            if sx:
                cv.box(x1 - sx * k, y1 - k, x1 - sx * k + 1, y1 + k + 1, c)
            else:
                cv.box(x1 - k, y1 - sy * k, x1 + k + 1, y1 - sy * k + 1, c)


# ---------------------------------------------------------------- scene

LABELS = []  # (x cell, baseline cell, text, color key, anchor)


def label(x, y, s, color="text", anchor="middle"):
    LABELS.append((x, y, s, color, anchor))


def lane(cv, x0, x1, y, c):
    """Dotted one-cell lane from x0 to x1 with a pixel arrowhead at x1."""
    sx = 1 if x1 > x0 else -1
    for x in range(x0, x1 - 4 * sx, sx):
        if (x - x0) % 2 == 0:
            cv.put(x, y, c)
    for k in range(4):
        cv.box(x1 - sx * k, y - k, x1 - sx * k, y + k, c)


def scene(cv):
    LABELS.clear()
    g = 70                                             # table / floor row
    cv.box(4, g, 260, g, "b")

    # robot cell: leader, follower over a cube, overhead camera
    arm(cv, 18, g - 1, [(2, -14), (12, -19), (18, -11)], "D", gripper=True)
    arm(cv, 46, g - 1, [(2, -22), (20, -28), (27, -5)], "O", scale=3)
    cv.box(71, g - 5, 75, g - 1, "K")
    cv.box(72, g - 4, 74, g - 2, "T")
    cv.box(87, 31, 88, g - 1, "D")
    cv.box(84, g - 2, 91, g - 1, "K")
    cv.sprite(80, 22, CAMERA)
    label(19, 78, "leader", "muted")
    label(50, 78, "follower", "muted")
    label(86, 78, "camera", "muted")

    # MacBook
    mx, my = 124, g - 37
    macbook(cv, mx, my)
    label(mx + 27, 78, "MacBook", "text")

    # GPU tower, scaled 2x
    gx, gy = 226, g - 34
    rows = GPU.strip("\n").splitlines()
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch != ".":
                cv.box(gx + 2 * dx, gy + 2 * dy, gx + 2 * dx + 1, gy + 2 * dy + 1, ch)
    label(gx + 17, 78, "GPU", "muted")

    top, bot = 44, 58
    # robot <-> MacBook
    lane(cv, 94, 118, top, "arrow_mac")
    label(106, top - 4, "collect")
    lane(cv, 118, 94, bot, "arrow_mac")
    label(106, bot + 8, "run")
    # MacBook <-> GPU, with the dataset riding on the top lane
    dataset(cv, 194, top - 7)
    label(203, top - 11, "dataset", "muted")
    lane(cv, 182, 191, top, "arrow_gpu")
    lane(cv, 213, 222, top, "arrow_gpu")
    lane(cv, 222, 182, bot, "arrow_gpu")
    label(202, bot + 8, "checkpoint")


class Pixfont:
    """Monocraft glyph outlines; one font pixel (120 units) = half an art cell."""

    def __init__(self, path):
        from fontTools.ttLib import TTFont

        self.f = TTFont(path)
        self.gs = self.f.getGlyphSet()
        self.cmap = self.f.getBestCmap()
        self.scale = PX / 2 / 120

    def path(self, s, x, y, anchor):
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.transformPen import TransformPen

        adv = [self.f["hmtx"][self.cmap[ord(ch)]][0] * self.scale for ch in s]
        width = sum(adv) - (adv[-1] - 600 * self.scale)   # drop trailing side bearing
        x0 = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
        x0 = round(x0 / (PX / 2)) * (PX / 2)               # snap to the font pixel grid
        pen = SVGPathPen(self.gs)
        cx = x0
        for ch, a in zip(s, adv):
            tp = TransformPen(pen, (self.scale, 0, 0, -self.scale, cx, y))
            self.gs[self.cmap[ord(ch)]].draw(tp)
            cx += a
        return pen.getCommands(), (x0, y - 840 * self.scale, x0 + width, y + 240 * self.scale)


def build(theme, font):
    cv = Canvas(theme)
    scene(cv)
    paths, boxes = [], []
    for x, y, s, color, anchor in LABELS:
        d, bb = font.path(s, x * PX, y * PX, anchor)
        paths.append(f'<path fill="{theme[color]}" d="{d}"/>')
        boxes.append(bb)
    xs = [x for x, _ in cv.cells] ; ys = [y for _, y in cv.cells]
    m = 6 * PX
    x0 = min(min(xs) * PX, min(b[0] for b in boxes)) - m
    y0 = min(min(ys) * PX, min(b[1] for b in boxes)) - m
    x1 = max((max(xs) + 1) * PX, max(b[2] for b in boxes)) + m
    y1 = max((max(ys) + 1) * PX, max(b[3] for b in boxes)) + m
    w, h = x1 - x0, y1 - y0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{h:g}" viewBox="{x0:g} {y0:g} {w:g} {h:g}" '
           'role="img" aria-label="macbot: teleoperate and collect on the MacBook, export a LeRobot dataset, '
           'train on a GPU, bring the checkpoint back and run it on the robot">',
           '<g shape-rendering="crispEdges">', *cv.svg_rects(), *paths, "</g>", "</svg>"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="path to Monocraft.ttf (used only to trace label outlines)")
    font = Pixfont(ap.parse_args().font)
    for name, t in THEMES.items():
        out = ROOT / "assets" / ("scene.svg" if name == "light" else "scene-dark.svg")
        out.write_text(build(t, font))
        print(out.relative_to(ROOT), f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
