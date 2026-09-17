#!/usr/bin/env python3
"""Generate the animated README figures for the spotify-next-track instance.

  assets/session.svg     studying a model: a question becomes a seed-grid scan and a report
  assets/harnesses.svg   Claude Code / Gemini CLI / Codex driving this one lab
  assets/loop.svg        listening history -> sessions -> runs -> model -> record
  assets/blend.svg       how the champion ranks the next track (R'+M+C' -> gate -> MMR)

Adapted from the lensing framework's branding/make_animations.py (same glyph
outlining and CSS/SMIL plumbing, because GitHub renders README SVGs through an
<img> sandbox with no web fonts and no scripts). The content is this instance's:
the session replays the real 2026-08-02c dropout seed-grid campaign, and the
palette follows the instance's signal GREEN (ui/src/styles/tokens.css) instead
of stock lensing orange. Each figure ships a prefers-reduced-motion end state.

  python3 branding/make_animations.py             # write all four
  python3 branding/make_animations.py --only blend.svg --outdir /tmp

Fonts come from the built UI (`zig build ui`): Sora for titles, Inter for
names, JetBrains Mono for every value, path and log line.
"""

import argparse
import glob
import math
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parent.parent

# --- Control Room tokens (ui/src/styles/tokens.css) --------------------------
TOK = {
    "chrome":        (0.27,  0.025, 250),
    "chrome_sunken": (0.235, 0.025, 250),
    "chrome_raised": (0.32,  0.025, 250),
    "chrome_border": (0.42,  0.025, 250),
    "hairline":      (0.36,  0.025, 250),
    "on_chrome":     (0.96,  0.008, 250),
    "on_chrome_dim": (0.78,  0.015, 250),
    "on_chrome_faint": (0.66, 0.02, 250),
    # Signal green is this instance's live-or-primary accent (--accent-on-chrome).
    "accent":        (0.80,  0.15,  150),
    "accent_deep":   (0.70,  0.165, 150),
    # Series + semantics, lifted for legibility on chrome.
    "series_b":      (0.68,  0.11,  245),
    "good":          (0.76,  0.10,  190),
    "bad":           (0.68,  0.15,  27),
}


def oklch_hex(L, C, H):
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def enc(c):
        c = max(0.0, min(1.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, c)) * 255)

    return "#%02x%02x%02x" % (enc(r), enc(g), enc(bl))


C = {k: oklch_hex(*v) for k, v in TOK.items()}


# --- fonts -------------------------------------------------------------------
FONT_GLOBS = {
    "mono":   ["ui/dist/assets/jetbrains-mono-latin-400-normal-*.woff2",
               "ui/node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2"],
    "mono_b": ["ui/dist/assets/jetbrains-mono-latin-600-normal-*.woff2",
               "ui/node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-600-normal.woff2"],
    "inter":  ["ui/dist/assets/inter-latin-wght-normal-*.woff2",
               "ui/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"],
    "sora":   ["ui/dist/assets/sora-latin-wght-normal-*.woff2",
               "ui/node_modules/@fontsource-variable/sora/files/sora-latin-wght-normal.woff2"],
}


def font_path(key):
    for pat in FONT_GLOBS[key]:
        hits = sorted(glob.glob(str(ROOT / pat)))
        if hits:
            return hits[0]
    raise SystemExit(
        f"make_animations: no font file for {key!r}. Build the UI once "
        f"(`zig build ui`, or `npm install` in ui/) so the faces are on disk."
    )


_FACE_CACHE = {}


def face(key, weight):
    ck = (key, weight)
    if ck not in _FACE_CACHE:
        f = TTFont(font_path(key))
        if "fvar" in f:
            instantiateVariableFont(f, {"wght": weight}, inplace=True)
        _FACE_CACHE[ck] = f
    return _FACE_CACHE[ck]


# Each distinct outline is written once into <defs> and referenced by <use>;
# a terminal figure repeats a few dozen glyphs hundreds of times, and inlining
# every path made the files five times larger than they needed to be.
GLYPH_IDS: dict = {}
GLYPH_DEFS: list = []


def glyph_id(key, weight, gname, glyphs):
    ck = (key, weight, gname)
    gid = GLYPH_IDS.get(ck)
    if gid is None:
        pen = SVGPathPen(glyphs)
        glyphs[gname].draw(pen)
        d = pen.getCommands()
        if not d:
            GLYPH_IDS[ck] = ""
            return ""
        gid = f"g{len(GLYPH_DEFS)}"
        GLYPH_IDS[ck] = gid
        GLYPH_DEFS.append(f'<path id="{gid}" d="{d}"/>')
    return gid


class Text:
    """An outlined run of text; glyph ids + font-unit offsets until placed."""

    def __init__(self, text, key, weight, size):
        f = face(key, weight)
        upm = f["head"].unitsPerEm
        self.scale = size / upm
        glyphs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
        self.parts, x = [], 0.0
        for ch in text:
            gname = cmap.get(ord(ch))
            if gname is None:
                raise SystemExit(f"make_animations: {key} has no glyph for {ch!r} "
                                 f"(U+{ord(ch):04X}) — draw it as a path instead")
            gid = glyph_id(key, weight, gname, glyphs)
            if gid:
                self.parts.append((x, gid))
            x += hmtx[gname][0]
        self.advance = x * self.scale

    def place(self, tx, baseline, fill, opacity=None):
        s = self.scale
        body = "".join(
            f'<use href="#{gid}" transform="translate({tx + gx * s:.2f} {baseline:.2f}) '
            f'scale({s:.5f} {-s:.5f})"/>'
            for gx, gid in self.parts
        )
        op = f' opacity="{opacity}"' if opacity is not None else ""
        return f'<g fill="{fill}"{op}>{body}</g>'


def text(s, x, y, key="mono", weight=400, size=13, fill=None):
    return Text(s, key, weight, size).place(x, y, fill or C["on_chrome"])


def width_of(s, key="mono", weight=400, size=13):
    return Text(s, key, weight, size).advance


# --- drawn glyphs the subset fonts do not carry ------------------------------
def check(x, y, color, s=1.0):
    """A tick, bottom vertex on the baseline."""
    return (f'<path d="M{x:.1f},{y - 3.6 * s:.1f} l{3.2 * s:.1f},{3.4 * s:.1f} '
            f'l{6.2 * s:.1f},{-8.4 * s:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{1.7 * s:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')


def rarrow(x, y, color, w=12.0, sw=1.3):
    """A rightward arrow, vertically centred on y."""
    return (f'<path d="M{x:.1f},{y:.1f} h{w:.1f} m{-4.2:.1f},{-3.4:.1f} '
            f'l{4.2:.1f},{3.4:.1f} l{-4.2:.1f},{3.4:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>')


CHECK_W, ARROW_W = 11.0, 14.0


def line(segments, x, baseline):
    """Lay out a mixed run: ('m'|'b'|'i'|'s', str, color, size) | ('check'|'arrow', color) | ('gap', n)."""
    out, cx = [], x
    for seg in segments:
        kind = seg[0]
        if kind == "gap":
            cx += seg[1]
        elif kind == "check":
            out.append(check(cx, baseline, seg[1]))
            cx += CHECK_W
        elif kind == "arrow":
            out.append(rarrow(cx, baseline - 4, seg[1]))
            cx += ARROW_W
        else:
            key, weight = {"m": ("mono", 400), "b": ("mono_b", 600),
                           "i": ("inter", 500), "s": ("sora", 550)}[kind]
            _, s, color = seg[0], seg[1], seg[2]
            size = seg[3] if len(seg) > 3 else 13
            t = Text(s, key, weight, size)
            out.append(t.place(cx, baseline, color))
            cx += t.advance
    return "".join(out), cx


def measure(segments, size_default=13):
    _, end = line(segments, 0, 0)
    return end


# --- animation plumbing ------------------------------------------------------
class Anim:
    """Generates one @keyframes rule per cue, all sharing the loop duration so
    the whole figure stays in lockstep and restarts together."""

    def __init__(self, duration):
        self.d = float(duration)
        self.rules = []
        self.names = []
        self.steps = {}
        self.dashed = {}
        self.n = 0

    def _name(self, prefix):
        self.n += 1
        name = f"{prefix}{self.n}"
        self.names.append(name)
        return name

    def _pct(self, t):
        return max(0.0, min(100.0, t / self.d * 100.0))

    def fade(self, t0, t1=None, rise=0.30, fall=0.45):
        """Class that fades a cue in at t0 and out at t1 (default: end of loop)."""
        t1 = self.d - 0.75 if t1 is None else t1
        n = self._name("f")
        p0, p1 = self._pct(t0), self._pct(t0 + rise)
        p2, p3 = self._pct(t1), self._pct(t1 + fall)
        self.rules.append(
            f"@keyframes {n}{{0%,{p0:.3f}%{{opacity:0}}"
            f"{p1:.3f}%,{p2:.3f}%{{opacity:1}}{p3:.3f}%,100%{{opacity:0}}}}"
        )
        return n

    def type(self, t0, dur, chars, span):
        """Class for a background-coloured cover that retreats one char at a time."""
        n = self._name("t")
        p0, p1 = self._pct(t0), self._pct(t0 + dur)
        self.rules.append(
            f"@keyframes {n}{{0%,{p0:.3f}%{{transform:translateX(0)}}"
            f"{p1:.3f}%,100%{{transform:translateX({span:.1f}px)}}}}"
        )
        self.steps[n] = chars
        return n

    def grow(self, t0, dur):
        """Class for a bar that fills left to right."""
        n = self._name("g")
        p0, p1 = self._pct(t0), self._pct(t0 + dur)
        self.rules.append(
            f"@keyframes {n}{{0%,{p0:.3f}%{{transform:scaleX(0)}}"
            f"{p1:.3f}%,100%{{transform:scaleX(1)}}}}"
        )
        return n

    def draw(self, t0, dur, length):
        """Class for a stroke that draws itself."""
        n = self._name("d")
        p0, p1 = self._pct(t0), self._pct(t0 + dur)
        self.rules.append(
            f"@keyframes {n}{{0%,{p0:.3f}%{{stroke-dashoffset:{length:.1f}}}"
            f"{p1:.3f}%,100%{{stroke-dashoffset:0}}}}"
        )
        self.dashed[n] = length
        return n

    def lit(self, t0, hold=1.1, rise=0.25):
        """Class that raises opacity to 1 briefly, then settles back to a dim rest."""
        n = self._name("l")
        p0, p1 = self._pct(t0), self._pct(t0 + rise)
        p2, p3 = self._pct(t0 + rise + hold), self._pct(t0 + rise + hold + 0.6)
        self.rules.append(
            f"@keyframes {n}{{0%,{p0:.3f}%{{opacity:0}}"
            f"{p1:.3f}%,{p2:.3f}%{{opacity:1}}{p3:.3f}%,100%{{opacity:0}}}}"
        )
        return n

    freeze = None  # debug: pin every animation at this many seconds into the loop

    def css(self, extra=""):
        body = list(self.rules)
        for n in self.names:
            if n in self.steps:
                body.append(f".{n}{{animation:{n} {self.d}s steps({self.steps[n]}) infinite}}")
            elif n in self.dashed:
                ln = self.dashed[n]
                body.append(f".{n}{{stroke-dasharray:{ln:.1f};stroke-dashoffset:{ln:.1f};"
                            f"animation:{n} {self.d}s ease-out infinite}}")
            elif n.startswith("g"):
                body.append(f".{n}{{transform-origin:0 0;animation:{n} {self.d}s "
                            f"cubic-bezier(.3,.7,.4,1) infinite}}")
            else:
                body.append(f".{n}{{opacity:0;animation:{n} {self.d}s linear infinite}}")
        if extra:
            body.append(extra)
        body.append(
            "@media (prefers-reduced-motion:reduce){"
            "*{animation:none!important}"
            "[class^=f]{opacity:1!important}"
            # the lit rings mark where the signal is *now*; with nothing moving
            # they would just paint every box orange at once.
            "[class^=l]{display:none!important}"
            "[class^=t]{transform:none!important;opacity:0!important}"
            "[class^=g]{transform:scaleX(1)!important}"
            "[class^=d]{stroke-dashoffset:0!important}"
            ".pm{display:none!important}}"
        )
        if Anim.freeze is not None:
            body.append(f"*{{animation-delay:{-Anim.freeze}s!important;"
                        f"animation-play-state:paused!important}}")
        return "<style>" + "".join(body) + "</style>"


def frame(w, h, r=16, fill=None, stroke=None):
    fill = fill or C["chrome"]
    stroke = stroke or C["chrome_border"]
    return (f'<rect x="0" y="0" width="{w}" height="{h}" rx="{r}" fill="{fill}"/>'
            f'<rect x="0.75" y="0.75" width="{w - 1.5}" height="{h - 1.5}" rx="{r - 0.75}" '
            f'fill="none" stroke="{stroke}" stroke-width="1.5"/>')


def box(x, y, w, h, r=8, fill=None, stroke=None, sw=1.2, extra=""):
    fill = fill or C["chrome_sunken"]
    stroke = stroke or C["hairline"]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')


def svg(w, h, label, title, body, style):
    defs = "".join(GLYPH_DEFS)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{label}">'
            f'<title>{title}</title>{style}'
            f'<defs>{defs}'
            f'<clipPath id="clip"><rect x="0" y="0" width="{w}" height="{h}" rx="16"/></clipPath>'
            f'</defs>'
            f'{frame(w, h)}<g clip-path="url(#clip)">{body}</g></svg>')




# ============================================================================
# 1. session.svg — studying a model turns into a scan and a report
# ============================================================================
def build_session():
    W, H, D = 880, 560, 30.0
    a = Anim(D)
    L, R = 30, W - 30
    o = []

    # --- topbar ---------------------------------------------------------
    o.append(f'<path d="M0,44 H{W}" stroke="{C["hairline"]}" stroke-width="1"/>')
    tb, _ = line([("s", "spotify-next-track", C["on_chrome"], 13),
                  ("gap", 7), ("m", "·", C["on_chrome_faint"], 12),
                  ("gap", 7), ("m", "studying a model", C["on_chrome_dim"], 12)], L, 28)
    o.append(tb)

    pill_w = 62
    lp = a.fade(13.6, 20.8)
    o.append(f'<g class="{lp} pm">'
             f'{box(R - pill_w, 15, pill_w, 19, r=9.5, fill=C["chrome_raised"], stroke=C["accent_deep"], sw=1)}'
             f'<circle cx="{R - pill_w + 13}" cy="24.5" r="3.2" fill="{C["accent"]}"/>'
             f'{text("live", R - pill_w + 22, 28.5, "mono_b", 600, 10.5, C["accent"])}</g>')

    def prompt(y, s, t0, dur):
        x = L + 15
        g = [text("›", L, y, "mono_b", 600, 14, C["accent"])]
        g.append(text(s, x, y, "mono", 400, 13.5, C["on_chrome"]))
        span = width_of(s, "mono", 400, 13.5)
        tc = a.type(t0, dur, max(1, len(s)), span)
        fc = a.fade(t0, t0 + dur + 0.35, rise=0.01, fall=0.2)
        g.append(f'<g transform="translate({x:.1f} {y})"><g class="{tc}">'
                 f'<rect x="0" y="-13" width="{R - x + 20:.1f}" height="19" fill="{C["chrome"]}"/>'
                 f'<g class="{fc}"><rect x="1" y="3.5" width="7.5" height="1.9" fill="{C["accent"]}"/></g>'
                 f'</g></g>')
        cue = a.fade(t0 - 0.1, rise=0.01)
        return f'<g class="{cue}">' + "".join(g) + "</g>"

    def reply(y, segments, t0):
        body, _ = line(segments, L + 15, y)
        return f'<g class="{a.fade(t0)}">{body}</g>'

    dim, faint, ink = C["on_chrome_dim"], C["on_chrome_faint"], C["on_chrome"]

    o.append(prompt(78, "study the deployed dual-tower GRU — would dropout help it?", 0.7, 3.0))
    o.append(reply(104, [
        ("m", "experiment-designer", faint, 12.5), ("gap", 9),
        ("m", "reads PROJECT-FACTS.md: every fusion_layers=0 arm trained with no dropout", dim, 12.5)], 4.1))
    o.append(reply(128, [
        ("m", "design", faint, 12.5), ("gap", 9),
        ("m", "dropout {0.0, 0.1} × seed {1337, 7, 42}", ink, 12.5), ("gap", 9),
        ("m", "· 6 runs · judged on the 3-seed paired mean", dim, 12.5)], 5.6))

    o.append(prompt(162, "approved — run it", 7.6, 1.4))
    o.append(reply(188, [
        ("m", "experiment-runner", faint, 12.5), ("gap", 9),
        ("m", "seq-bank · views latent+cummean · h256 · queued on the worker", dim, 12.5)], 9.4))

    # --- the runs panel -------------------------------------------------
    px, py, pw, ph = L, 204, R - L, 160
    panel = [box(px, py, pw, ph, r=10)]
    panel.append(text("runs · seq-bank latent+cummean h256", px + 18, py + 24, "mono_b", 600, 11, faint))
    panel.append(f'<path d="M{px + 18},{py + 34} H{px + pw - 18}" stroke="{C["hairline"]}" stroke-width="1"/>')

    labels = ["d0.0 · s1337", "d0.1 · s1337", "d0.0 · s7", "d0.1 · s7", "d0.0 · s42", "d0.1 · s42"]
    hits = [168, 172, 166, 162, 173, 160]
    bx, bw = px + 128, 220
    for i, (lab, h) in enumerate(zip(labels, hits)):
        ry = py + 58 + i * 18
        t0 = 13.8 + i * 0.45
        dur = 3.4 + (i % 3) * 0.55
        end = t0 + dur
        panel.append(text(lab, px + 18, ry + 3, "mono", 400, 10.5, dim))
        panel.append(f'<rect x="{bx}" y="{ry - 3}" width="{bw}" height="5" rx="2.5" fill="{C["chrome_raised"]}"/>')
        panel.append(f'<g transform="translate({bx} {ry - 3})">'
                     f'<rect x="0" y="0" width="{bw}" height="5" rx="2.5" fill="{C["accent"]}" '
                     f'class="{a.grow(t0, dur)}"/></g>')
        panel.append(f'<g class="{a.fade(end, rise=0.4)}"><rect x="{bx}" y="{ry - 3}" width="{bw}" '
                     f'height="5" rx="2.5" fill="{C["on_chrome_faint"]}"/></g>')
        panel.append(f'<g class="{a.fade(end + 0.15, rise=0.35)}">'
                     f'{text("R@10", bx + bw + 14, ry + 3, "mono", 400, 10, faint)}'
                     f'{text(f"{h / 1431:.4f}", bx + bw + 48, ry + 3, "mono", 400, 10.5, ink)}'
                     f'{text(f"{h} hits", bx + bw + 104, ry + 3, "mono", 400, 10, faint)}</g>')

    # paired deltas per seed, drawn as CI whiskers straddling zero
    cx0, cx1 = px + 590, px + pw - 24
    wx0 = cx0 + 52
    zero = wx0 + (cx1 - wx0) * 0.66
    scale = (zero - wx0) / 0.02
    panel.append(text("paired delta recall@10 per seed", cx0, py + 52, "mono", 400, 10, faint))
    panel.append(f'<path d="M{zero:.1f},{py + 62} V{py + 146}" stroke="{C["hairline"]}" stroke-width="1.2"/>')
    cis = [("s1337", 0.002795, -0.002795, 0.007704),
           ("s7", -0.002795, -0.008386, 0.002795),
           ("s42", -0.009085, -0.018868, 0.000699)]
    for i, (lab, m, lo, hi) in enumerate(cis):
        yy = py + 80 + i * 26
        t = 19.2 + i * 0.5
        g = [text(lab, cx0, yy + 3.5, "mono", 400, 10, faint),
             f'<path d="M{zero + lo * scale:.1f},{yy} H{zero + hi * scale:.1f}" stroke="{dim}" stroke-width="1.6"/>',
             f'<circle cx="{zero + m * scale:.1f}" cy="{yy}" r="3.4" fill="{ink}"/>']
        panel.append(f'<g class="{a.fade(t, rise=0.4)}">' + "".join(g) + "</g>")
    panel.append(f'<g class="{a.fade(20.8, rise=0.4)}">'
                 f'{text("0", zero - 3, py + 158 - 2, "mono", 400, 9.5, faint)}</g>')
    o.append(f'<g class="{a.fade(13.3)}">' + "".join(panel) + "</g>")

    # --- verdict and what it left behind --------------------------------
    o.append(reply(396, [
        ("check", C["good"]), ("gap", 4),
        ("m", "verdict", faint, 12.5), ("gap", 9),
        ("b", "a recall gain is EXCLUDED", ink, 12.5), ("gap", 9),
        ("m", "3-seed delta −0.0030 [−0.0070, +0.0009]", dim, 12.5)], 21.6))
    o.append(reply(420, [
        ("check", C["good"]), ("gap", 4),
        ("m", "by-product: first seed-variance figure — range 12 hits at d0.1", dim, 12.5)], 22.8))
    o.append(reply(444, [
        ("check", C["good"]), ("gap", 4),
        ("m", "experiments/2026-08-02c-dropout-seed-grid-and-the-walk-read.md", ink, 12.5), ("gap", 9),
        ("m", "written", faint, 12.5)], 24.0))
    o.append(reply(468, [
        ("check", C["good"]), ("gap", 4),
        ("m", "PROJECT-FACTS.md", ink, 12.5), ("gap", 9),
        ("m", "· leaderboard and follow-ups reconciled", dim, 12.5)], 25.1))
    foot, _ = line([("i", "One sentence of intent in; a pre-registered scan, six runs and a "
                          "written verdict out.", faint, 12)], L + 15, 510)
    o.append(f'<g class="{a.fade(26.4)}">{foot}</g>')

    return svg(W, H, "Studying a model: a plain-language question becomes a designed seed-grid "
                     "scan of six runs, a verdict and a written experiment report.",
               "studying a model, start to finish", "".join(o), a.css())


def pulse(a, path, t0, t1, D, r=3.3, color=None, glow=True):
    """A signal travelling a path once per loop, inside its own visibility window."""
    color = color or C["accent"]
    motion = (f'<animateMotion dur="{D}s" repeatCount="indefinite" calcMode="linear" '
              f'keyPoints="0;0;1;1" keyTimes="0;{t0 / D:.4f};{t1 / D:.4f};1" path="{path}"/>')
    halo = (f'<circle r="{r * 2.4:.1f}" fill="{color}" opacity="0.16">{motion}</circle>') if glow else ""
    return (f'<g class="{a.fade(t0, t1, rise=0.18, fall=0.22)} pm">{halo}'
            f'<circle r="{r}" fill="{color}">{motion}</circle></g>')


# ============================================================================
# 2. harnesses.svg — any coding agent drives this lab
# ============================================================================
def build_harnesses():
    W, H, D = 880, 268, 9.6
    a = Anim(D)
    o = []
    dim, faint, ink = C["on_chrome_dim"], C["on_chrome_faint"], C["on_chrome"]

    o.append(text("YOUR CODING AGENT", 28, 34, "mono_b", 600, 10.5, faint))
    o.append(text("SKILLS RENDERED FOR THIS DOMAIN", 304, 34, "mono_b", 600, 10.5, faint))
    o.append(text("THIS LAB", 634, 34, "mono_b", 600, 10.5, faint))

    sources = [
        ("Claude Code", ".claude/skills · agents"),
        ("Gemini CLI",  ".gemini/skills"),
        ("Codex", ".agents/skills"),
    ]
    sx, sw, sh = 28, 196, 52
    mid_x, mid_y, mid_w, mid_h = 304, 58, 250, 180
    mid_cy = mid_y + mid_h / 2

    starts = [0.5, 3.0, 5.5]
    for i, ((name, sub), t0) in enumerate(zip(sources, starts)):
        y = 58 + i * 64
        cy = y + sh / 2
        o.append(box(sx, y, sw, sh, r=9))
        o.append(text(name, sx + 16, y + 23, "inter", 600, 13.5, ink))
        o.append(text(sub, sx + 16, y + 39, "mono", 400, 10, faint))
        o.append(f'<g class="{a.lit(t0, hold=1.6)}">'
                 f'{box(sx, y, sw, sh, r=9, fill="none", stroke=C["accent_deep"], sw=1.4)}</g>')
        p = (f"M{sx + sw},{cy:.1f} C{sx + sw + 44},{cy:.1f} "
             f"{mid_x - 44},{mid_cy:.1f} {mid_x},{mid_cy:.1f}")
        o.append(f'<path d="{p}" fill="none" stroke="{C["hairline"]}" stroke-width="1.4"/>')
        o.append(pulse(a, p, t0 + 0.25, t0 + 1.55, D))

    o.append(box(mid_x, mid_y, mid_w, mid_h, r=10, fill=C["chrome_raised"], stroke=C["chrome_border"]))
    o.append(text("tracks · sessions · H@10", mid_x + 18, mid_y + 25, "inter", 600, 13.5, ink))
    o.append(f'<path d="M{mid_x + 18},{mid_y + 35} H{mid_x + mid_w - 18}" '
             f'stroke="{C["hairline"]}" stroke-width="1"/>')
    cmds = ["experiment-designer", "experiment-runner", "/dataset-design",
            "/report-curator", "/information-capture", "/showcase"]
    for i, c in enumerate(cmds):
        o.append(text(c, mid_x + 18, mid_y + 58 + i * 20, "mono", 400, 11.5, dim))

    rx, ry, rw, rh = 634, 110, 218, 76
    link = f"M{mid_x + mid_w},{mid_cy:.1f} H{rx - 9}"
    o.append(f'<path d="{link}" fill="none" stroke="{C["hairline"]}" stroke-width="1.4"/>')
    o.append(rarrow(rx - 13, mid_cy, C["hairline"], w=8, sw=1.4))
    for t0 in starts:
        o.append(pulse(a, link, t0 + 1.6, t0 + 2.3, D))
    o.append(box(rx, ry, rw, rh, r=9))
    o.append(text("lensing-server :8096", rx + 18, ry + 28, "inter", 600, 13.5, ink))
    o.append(text("API · Control Room · /lab", rx + 18, ry + 47, "mono", 400, 10.5, faint))
    o.append(text("runs · best-models · SAE", rx + 18, ry + 63, "mono", 400, 10.5, faint))

    return svg(W, H, "Claude Code, the Gemini CLI and Codex each read the skill layer rendered "
                     "for this domain, and all three drive the same lensing server.",
               "three front doors, one lab", "".join(o), a.css())


# ============================================================================
# 3. loop.svg — the empirical loop, in this instance's nouns
# ============================================================================
def build_loop():
    W, H, D = 880, 262, 12.0
    a = Anim(D)
    o = []
    dim, faint, ink = C["on_chrome_dim"], C["on_chrome_faint"], C["on_chrome"]

    tb, _ = line([("s", "the loop", ink, 13), ("gap", 8),
                  ("m", "· a campaign ends by writing itself down", faint, 12)], 28, 36)
    o.append(tb)

    nodes = [
        ("history",  "export · Qdrant"),
        ("sessions", "split · PCA-192"),
        ("runs",     "seq-* · recall · H@10"),
        ("model",    "promote · ONNX"),
        ("record",   "report · PDF"),
    ]
    nw, nh, gap, ny = 148, 62, 22, 70
    x0 = (W - (len(nodes) * nw + (len(nodes) - 1) * gap)) / 2
    cy = ny + nh / 2
    lit_at = [0.3, 1.9, 3.5, 5.1, 6.7]

    for i, ((name, sub), t) in enumerate(zip(nodes, lit_at)):
        x = x0 + i * (nw + gap)
        o.append(box(x, ny, nw, nh, r=9))
        o.append(text(name, x + 16, ny + 27, "inter", 600, 13.5, ink))
        o.append(text(sub, x + 16, ny + 45, "mono", 400, 10, faint))
        o.append(f'<g class="{a.lit(t, hold=1.2)}">'
                 f'{box(x, ny, nw, nh, r=9, fill="none", stroke=C["accent_deep"], sw=1.4)}</g>')
        if i < len(nodes) - 1:
            ax = x + nw + 3
            seg = f"M{ax},{cy:.0f} H{ax + gap - 6}"
            o.append(f'<path d="{seg}" stroke="{C["hairline"]}" stroke-width="1.4"/>')
            o.append(rarrow(ax + gap - 12, cy, C["hairline"], w=6, sw=1.4))
            o.append(pulse(a, seg, t + 1.0, t + 1.5, D, r=3.0, glow=False))

    xa = x0 + 4 * (nw + gap) + nw / 2
    xb = x0 + 2 * (nw + gap) + nw / 2
    arc = (f"M{xa:.0f},{ny + nh} C{xa:.0f},{ny + nh + 62} {xa:.0f},{ny + nh + 62} "
           f"{xa - 70:.0f},{ny + nh + 62} L{xb + 70:.0f},{ny + nh + 62} "
           f"C{xb:.0f},{ny + nh + 62} {xb:.0f},{ny + nh + 62} {xb:.0f},{ny + nh + 7}")
    o.append(f'<path d="{arc}" fill="none" stroke="{C["hairline"]}" stroke-width="1.4" '
             f'stroke-dasharray="4 4"/>')
    o.append(f'<path d="M{xb - 4:.0f},{ny + nh + 12} L{xb:.0f},{ny + nh + 5} '
             f'L{xb + 4:.0f},{ny + nh + 12}" fill="none" stroke="{C["hairline"]}" '
             f'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(pulse(a, arc, 7.7, 9.6, D))
    msg = "PROJECT-FACTS.md is what the next scan design reads first"
    _, wlbl = line([("m", msg, faint, 11.5)], 0, 0)
    lbl, _ = line([("m", msg, faint, 11.5)], (W - wlbl) / 2, ny + nh + 84)
    o.append(f'<g class="{a.fade(7.7, 11.2)}">{lbl}</g>')

    return svg(W, H, "The loop: listening history to sessions to runs to a promoted model to a "
                     "written record, which the next scan design reads first.",
               "the loop", "".join(o), a.css())


# ============================================================================
# 4. blend.svg — how the champion ranks the next track
# ============================================================================
def build_blend():
    W, H, D = 880, 330, 11.0
    a = Anim(D)
    o = []
    dim, faint, ink = C["on_chrome_dim"], C["on_chrome_faint"], C["on_chrome"]

    tb, _ = line([("s", "the champion", ink, 13), ("gap", 8),
                  ("m", "· blend-gru-markov-content-proj · how one next track gets ranked", faint, 12)],
                 28, 36)
    o.append(tb)

    # input: the session prefix as a row of track chips
    ix, iy = 28, 70
    o.append(text("session so far", ix, iy - 8, "mono", 400, 10.5, faint))
    for i in range(5):
        cx = ix + i * 30
        o.append(f'<rect x="{cx}" y="{iy}" width="24" height="24" rx="5" fill="{C["chrome_raised"]}" '
                 f'stroke="{C["hairline"]}"/>')
        o.append(f'<g class="{a.lit(0.2 + i * 0.18, hold=0.6)}"><rect x="{cx}" y="{iy}" width="24" '
                 f'height="24" rx="5" fill="none" stroke="{C["accent_deep"]}" stroke-width="1.4"/></g>')
    in_x, in_y = ix + 5 * 30 - 6, iy + 12

    legs = [
        ("R'", "session GRU", "recurrence over the prefix, retrieved", "in the learned projection"),
        ("M", "Markov bigram", "what followed the last track in train;", "abstains with no evidence (gate)"),
        ("C'", "content kNN", "nearest tracks to the prefix,", "in the learned projection"),
    ]
    lx, lw, lh = 250, 250, 64
    blend_x, blend_y, blend_w, blend_h = 560, 118, 120, 104
    bcy = blend_y + blend_h / 2
    for i, (tag, name, s1, s2) in enumerate(legs):
        y = 58 + i * 84
        cy = y + lh / 2
        t0 = 1.3 + i * 0.35
        o.append(box(lx, y, lw, lh, r=9))
        o.append(text(tag, lx + 14, y + 24, "mono_b", 600, 14, C["accent"]))
        o.append(text(name, lx + 50, y + 24, "inter", 600, 13, ink))
        o.append(text(s1, lx + 14, y + 42, "mono", 400, 9.5, faint))
        o.append(text(s2, lx + 14, y + 55, "mono", 400, 9.5, faint))
        o.append(f'<g class="{a.lit(t0 + 0.9, hold=1.2)}">'
                 f'{box(lx, y, lw, lh, r=9, fill="none", stroke=C["accent_deep"], sw=1.4)}</g>')
        pin = f"M{in_x},{in_y} C{in_x + 50},{in_y} {lx - 50},{cy:.0f} {lx},{cy:.0f}"
        o.append(f'<path d="{pin}" fill="none" stroke="{C["hairline"]}" stroke-width="1.3"/>')
        o.append(pulse(a, pin, t0, t0 + 0.9, D, r=3.0))
        pout = f"M{lx + lw},{cy:.0f} C{lx + lw + 30},{cy:.0f} {blend_x - 30},{bcy:.0f} {blend_x},{bcy:.0f}"
        o.append(f'<path d="{pout}" fill="none" stroke="{C["hairline"]}" stroke-width="1.3"/>')
        o.append(pulse(a, pout, t0 + 2.2, t0 + 3.0, D, r=3.0))

    o.append(box(blend_x, blend_y, blend_w, blend_h, r=10, fill=C["chrome_raised"], stroke=C["chrome_border"]))
    o.append(text("z-blend", blend_x + 16, blend_y + 30, "inter", 600, 13.5, ink))
    o.append(text("z-score each", blend_x + 16, blend_y + 52, "mono", 400, 10, faint))
    o.append(text("leg, average", blend_x + 16, blend_y + 66, "mono", 400, 10, faint))
    o.append(text("the thirds", blend_x + 16, blend_y + 80, "mono", 400, 10, faint))
    o.append(f'<g class="{a.lit(4.6, hold=1.0)}">'
             f'{box(blend_x, blend_y, blend_w, blend_h, r=10, fill="none", stroke=C["accent_deep"], sw=1.4)}</g>')

    # output: top-10 list, drawn row by row
    ox, oy = 716, 62
    link = f"M{blend_x + blend_w},{bcy:.0f} H{ox - 8}"
    o.append(f'<path d="{link}" fill="none" stroke="{C["hairline"]}" stroke-width="1.3"/>')
    o.append(pulse(a, link, 5.4, 5.9, D, r=3.0))
    o.append(text("top 10", ox, oy - 4, "mono", 400, 10.5, faint))
    for i in range(10):
        ry = oy + 6 + i * 21
        w = 112 - (i * 6)
        o.append(f'<g class="{a.fade(5.9 + i * 0.16, rise=0.2)}">'
                 f'{text(f"{i + 1:02d}", ox, ry + 11, "mono", 400, 9.5, faint)}'
                 f'<rect x="{ox + 22}" y="{ry + 3}" width="{w}" height="10" rx="3" '
                 f'fill="{C["accent"] if i == 0 else C["on_chrome_faint"]}" '
                 f'opacity="{1 if i == 0 else 0.55}"/></g>')

    foot, _ = line([("m", "Recall@10 0.212 (0.232 with the gate) vs 0.107 for Markov alone", dim, 11.5)],
                   250, 318)
    o.append(f'<g class="{a.fade(7.8)}">{foot}</g>')

    return svg(W, H, "The champion ranks the next track by z-blending three legs: a session GRU and "
                     "a content kNN retrieved in a learned projection, and a gated Markov bigram.",
               "how the champion ranks", "".join(o), a.css())


FIGURES = {
    "session.svg": build_session,
    "harnesses.svg": build_harnesses,
    "loop.svg": build_loop,
    "blend.svg": build_blend,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "assets"))
    ap.add_argument("--only", choices=sorted(FIGURES), help="write just one figure")
    ap.add_argument("--freeze", type=float,
                    help="debug: pin every animation N seconds into its loop")
    args = ap.parse_args()
    Anim.freeze = args.freeze
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        if args.only and name != args.only:
            continue
        GLYPH_IDS.clear()
        GLYPH_DEFS.clear()
        p = out / name
        p.write_text(build() + "\n")
        print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
