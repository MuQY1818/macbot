"""Generate the macbot pixel logo (SVG + PNG) from character grids.

    python tools/brand/make_logo.py     # writes assets/*.svg, assets/*.png, macbot/web/static/logo-mark.svg
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PALETTE = {
    "K": "#1B1A2E",  # ink outline
    "S": "#2B2A4C",  # screen
    "s": "#35345E",  # screen scanline
    "O": "#FF7A1A",  # arm (brand orange)
    "Y": "#FFC93C",  # joints and gripper
    "T": "#2EC4B6",  # cube (teal)
    "B": "#C9C3B6",  # laptop deck
    "b": "#9E978A",  # deck shadow
}

# 16 x 14 mark: a laptop whose screen shows a robot arm reaching for a cube.
MARK = """
.KKKKKKKKKKKKKK.
.KSSSSSSSSSSSSK.
.KssssssssssssK.
.KSSYOOOOOOOSSK.
.KssOsssssOOssK.
.KSSOSSSSYYYYSK.
.KssOssssYssYsK.
.KSSYSSSSSTTSSK.
.KsOOOssssTTssK.
.KOOOOOSSSSSSSK.
.KKKKKKKKKKKKKK.
KBBBBBBBBBBBBBBK
KbbbbbbbbbbbbbbK
.KKKKKKKKKKKKKK.
"""

# 5 x 7 lowercase pixel glyphs for the wordmark.
GLYPHS = {
    "m": [".....", ".....", "XX.X.", "X.X.X", "X.X.X", "X.X.X", "X.X.X"],
    "a": [".....", ".....", ".XXX.", "....X", ".XXXX", "X...X", ".XXXX"],
    "c": [".....", ".....", ".XXX.", "X....", "X....", "X....", ".XXX."],
    "b": ["X....", "X....", "XXXX.", "X...X", "X...X", "X...X", "XXXX."],
    "o": [".....", ".....", ".XXX.", "X...X", "X...X", "X...X", ".XXX."],
    "t": [".X...", ".X...", "XXXX.", ".X...", ".X...", ".X...", "..XX."],
}


def grid(text):
    return [r for r in text.strip("\n").splitlines()]


def rects(rows, px, ox=0, oy=0, colors=PALETTE):
    out = []
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            c = row[x]
            if c in colors:
                run = 1
                while x + run < len(row) and row[x + run] == c:
                    run += 1
                out.append(f'<rect x="{ox + x * px}" y="{oy + y * px}" width="{run * px}" height="{px}" fill="{colors[c]}"/>')
                x += run
            else:
                x += 1
    return out


def word_rows(word, color_char="X"):
    rows = ["" for _ in range(7)]
    for i, ch in enumerate(word):
        for r in range(7):
            rows[r] += GLYPHS[ch][r] + ("." if i < len(word) - 1 else "")
    return rows


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'shape-rendering="crispEdges" role="img" aria-label="{title}"><title>{title}</title>' + "".join(body) + "</svg>\n")


def main():
    mark = grid(MARK)
    mw, mh = len(mark[0]), len(mark)
    # mark only (favicon / avatar)
    (ROOT / "assets/logo-mark.svg").write_text(svg(mw, mh, rects(mark, 1), "macbot"))
    (ROOT / "macbot/web/static/logo-mark.svg").write_text(svg(mw, mh, rects(mark, 1), "macbot"))

    # horizontal lockup: mark + wordmark ("mac" ink, "bot" orange)
    px = 8
    wm = word_rows("macbot")
    gap = 3
    W = (mw + gap + len(wm[0])) * px
    H = mh * px
    oy = (mh - 7) // 2 + 1
    for name, ink in (("logo", "#1B1A2E"), ("logo-dark", "#FFF4DC")):
        pal = dict(PALETTE, K="#4A4878") if name == "logo-dark" else PALETTE
        body = rects(mark, px, colors=pal)
        mac = [r[:17] for r in wm]
        bot = ["." * 18 + r[18:] for r in wm]
        body += rects(mac, px, (mw + gap) * px, oy * px, {"X": ink})
        body += rects(bot, px, (mw + gap) * px, oy * px, {"X": "#FF7A1A"})
        (ROOT / f"assets/{name}.svg").write_text(svg(W, H, body, "macbot"))

    # PNG renders (avatar / social preview) via Pillow, nearest-neighbour scaling
    from PIL import Image, ImageDraw
    def paint(rows, colors, scale, bg=None, pad=0):
        w, h = len(rows[0]), len(rows)
        im = Image.new("RGBA", ((w + 2 * pad) * scale, (h + 2 * pad) * scale), bg or (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c in colors:
                    d.rectangle([(x + pad) * scale, (y + pad) * scale, (x + pad + 1) * scale - 1, (y + pad + 1) * scale - 1],
                                fill=colors[c])
        return im
    paint(mark, PALETTE, 32, bg="#FFF4DC", pad=1).save(ROOT / "assets/avatar.png")
    print("wrote assets/logo.svg, logo-dark.svg, logo-mark.svg, avatar.png")


if __name__ == "__main__":
    main()
