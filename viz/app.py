"""TraceLens — spot AI-generated images, and see how far they survive resharing.

    source ~/techjam/venv/bin/activate
    streamlit run viz/app.py --server.port 8508

No model is loaded here, so no GPU and no CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.transforms import apply_transform  # noqa: E402 — needs sys.path first

# Transport cap. Only shrinks the copy handed to the browser; adjustments and
# scoring always run on the native-resolution image.
PREVIEW_MAX_SIDE = 900

# On-screen box for each of the two pictures, in CSS pixels. The width handed to
# st.image is computed from this, because CSS cannot reach the <img>: Streamlit
# wraps every element in its own container, so a bare <div> from st.markdown is
# auto-closed and ends up a sibling of the image rather than its parent.
FIGURE_BOX = (340, 270)

_ACCEPTED = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

# Pinned rather than left to CUDA_VISIBLE_DEVICES, so the running server does not
# have to be restarted with an env var — and so we never land on GPU 0, where vLLM
# is holding 39 GB.
DETECTOR_DEVICE = "cuda:1"

ABLATION_HTML = _ROOT / "ablation" / "ablation.html"


# ======================================================================
# Adjustments
# ======================================================================

@dataclass(frozen=True)
class Adjustment:
    """One official degradation. ``values`` runs mild → severe, left to right."""

    key: str                       # a data.transforms TransformName
    label: str
    values: tuple[float, ...]      # officially permitted values, in severity order
    fmt: tuple[str, ...]           # caption per value
    note: str


# Applied in the order an image actually meets them out in the world: framed,
# rescaled by the platform, softened, grainy, color-shifted, then encoded.
ADJUSTMENTS: tuple[Adjustment, ...] = (
    Adjustment("center_crop", "Crop", (0.8,), ("80%",),
               "Keeps the middle 80%, then scales back up"),
    Adjustment("resize", "Resample", (0.5, 0.25), ("½", "¼"),
               "Shrinks and stretches back, the way a platform would"),
    Adjustment("gaussian_blur", "Blur", (0.5, 1.0, 2.0), ("0.5", "1.0", "2.0"),
               "Gaussian radius, in pixels"),
    Adjustment("gaussian_noise", "Noise", (0.02, 0.05, 0.10), ("2%", "5%", "10%"),
               "Sensor-style grain, added per channel"),
    Adjustment("color_jitter", "Color", (0.2,), ("20%",),
               "Nudges brightness, contrast and saturation"),
    Adjustment("jpeg_compression", "JPEG", (90, 70, 50, 30), ("90", "70", "50", "30"),
               "Encoder quality — lower means heavier blocking"),
)

_BY_KEY = {a.key: a for a in ADJUSTMENTS}

# Leftmost segment of every control. A string sentinel rather than None, because
# an unselected segmented_control already returns None — both mean "not applied".
_OFF = "Off"


def _noise_fast(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    """Vectorized Gaussian noise. Same semantics as ``data.transforms``, ~40× faster.

    Upstream builds the image byte by byte in Python::

        bytes(_clip(c + rng.gauss(0.0, pixel_sigma)) for c in source.tobytes())

    which costs 24 s on a 12 MP image — far too slow to sit behind a live control.
    Noise is per-channel i.i.d., so a different random stream is statistically
    indistinguishable (``evaluate.py`` calls ``apply_transform`` without a seed
    anyway, so it is already irreproducible). Only the implementation changes:
    same σ×255, same round-then-clip into [0, 255].
    """
    arr = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma * 255.0, arr.shape)
    out = np.clip(np.rint(arr + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@st.cache_data(max_entries=24, show_spinner=False)
def _apply_chain(_image: Image.Image, digest: str,
                 chain: tuple[tuple[str, float], ...], seed: int) -> Image.Image:
    """Apply ``chain`` in order. The ``_image`` underscore keeps it out of the cache key.

    Keyed on (digest, chain, seed), so returning to a setting already viewed is instant.
    """
    result = _image
    for key, value in chain:
        if key == "gaussian_noise":
            result = _noise_fast(result, value, seed)
        else:
            result = apply_transform(result, key, value=value, seed=seed)
    return result


def _chain_label(chain: tuple[tuple[str, float], ...]) -> str:
    parts = []
    for key, value in chain:
        adj = _BY_KEY[key]
        parts.append(f"{adj.label} {adj.fmt[adj.values.index(value)]}")
    return "  ·  ".join(parts)


# ======================================================================
# The image under examination
# ======================================================================

@dataclass(frozen=True)
class Photo:
    """``raw`` is the uploaded bytes, never modified."""

    raw: bytes
    name: str
    image: Image.Image          # EXIF-corrected RGB at native resolution
    exif_rotated: bool

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


def _decode(raw: bytes, name: str) -> Photo:
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        original = opened.copy()
    upright = ImageOps.exif_transpose(original)
    return Photo(raw=raw, name=name, image=upright.convert("RGB"),
                 exif_rotated=upright.size != original.size)


def _preview_copy(image: Image.Image, *, max_side: int = PREVIEW_MAX_SIDE) -> Image.Image:
    """Shrink a copy for the browser. Never feed this to adjustment or scoring."""
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _display_width(image: Image.Image, box: tuple[int, int]) -> int:
    """Width in CSS pixels that fits ``image`` inside ``box`` at its own aspect ratio."""
    box_w, box_h = box
    scale = min(box_w / image.width, box_h / image.height, 1.0)
    return max(1, round(image.width * scale))


# ======================================================================
# Detector
# ======================================================================

@st.cache_resource(show_spinner=False)
def _gpu_ready() -> bool:
    """Scoring needs CUDA. CPU is skipped — ViT-H is not a demo-time CPU model."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _detector_device() -> str:
    """Prefer the pinned card; fall back to cuda:0 if that index is missing."""
    import torch
    try:
        idx = int(DETECTOR_DEVICE.rsplit(":", 1)[1])
    except (ValueError, IndexError):
        idx = 0
    if torch.cuda.device_count() > idx:
        return DETECTOR_DEVICE
    return "cuda:0"


@st.cache_resource(show_spinner=False)
def _detector():
    """Built once per server, then reused. ~2.4 GB of weights land on GPU.

    Imported lazily: pulling in torch and open_clip at module scope would cost
    every page load, and this page is useful without ever scoring anything.
    """
    from serve.spatial_backend import DEFAULT_CKPT, SpatialDetector
    return SpatialDetector(ckpt=_ROOT / DEFAULT_CKPT, device=_detector_device())


def _score(photo: Photo, chain: tuple[tuple[str, float], ...],
           seed: int) -> list[dict]:
    """Score the original and, when adjusted, the adjusted copy alongside it.

    Both go in at native resolution: the model does its own Resize(224) +
    CenterCrop(224), and shrinking early would change the resampling traces it sees.
    """
    images = [photo.image]
    names = [photo.name]
    if chain:
        images.append(_apply_chain(photo.image, photo.digest, chain, seed))
        names.append(f"{photo.name} · adjusted")
    return _detector().score_pils(images, names=names)


def _verdict_html(record: dict | None, width: int) -> str:
    """Result card, same width as the picture above it."""
    if record is None:
        return (f'<div class="verdict-wrap"><div class="verdict empty" '
                f'style="width:{width}px">Not scored yet</div></div>')
    pred = record["pred"]
    fake = record["label"] == "fake"
    call = "Likely AI-generated" if fake else "Likely a real photo"
    tone = "fake" if fake else "real"
    confidence = pred if fake else 1.0 - pred
    return (
        f'<div class="verdict-wrap"><div class="verdict {tone}" style="width:{width}px">'
        f'<span class="dot"></span><span class="call">{call}</span>'
        f'<span class="pct">{confidence * 100:.0f}%</span>'
        f'<div class="meter"><div class="fill" style="width:{max(2.0, pred * 100):.1f}%">'
        "</div></div>"
        f'<div class="detail">P(AI) {pred:.3f} · logit {record["logit"]:+.2f}</div>'
        "</div></div>"
    )


# ======================================================================
# Formatting
# ======================================================================

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _fmt_ratio(width: int, height: int) -> str:
    from math import gcd
    g = gcd(width, height) or 1
    w, h = width // g, height // g
    return f"{w}:{h}" if max(w, h) <= 40 else f"{width / height:.2f}:1"


def _facts(photo: Photo) -> list[tuple[str, str]]:
    with Image.open(io.BytesIO(photo.raw)) as probe:
        fmt, mode = probe.format or "Unknown", probe.mode
        width, height = probe.size
        n_frames = getattr(probe, "n_frames", 1)
        icc = "Embedded" if probe.info.get("icc_profile") else "None"
        exif = probe.getexif()

    rows = [
        ("Format", fmt),
        ("Dimensions", f"{width} × {height}"),
        ("Resolution", f"{width * height / 1e6:.1f} MP"),
        ("Aspect", _fmt_ratio(width, height)),
        ("Size", _fmt_bytes(len(photo.raw))),
        ("Color", mode),
        ("Color profile", icc),
        ("Metadata", f"{len(exif)} EXIF tags" if exif else "None"),
    ]
    if n_frames > 1:
        rows.insert(2, ("Frames", f"{n_frames}, using the first"))
    return rows


# ======================================================================
# Appearance
#
# Shared with ablation/ablation.html: system typeface, page #f5f5f7, white
# 12px cards with a 7% hairline (no drop shadow), accent #0071e3, and the
# same success / warning / danger inks.
# ======================================================================

_CSS = """
<style>
:root {
    --bg:      #f5f5f7;
    --card:    #ffffff;
    --text:    #1d1d1f;
    --muted:   #6e6e73;
    --line:    rgba(0, 0, 0, 0.07);
    --fill:    rgba(120, 120, 128, 0.12);
    --accent:  #0071e3;
    --success: #1f8a4c;
    --warning: #b25000;
    --danger:  #c41e3a;
    --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font: 15px/1.5 system-ui, -apple-system, "SF Pro Text", "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stHeader"] { background: transparent; }
/* Hide the empty chrome while the sidebar is open. When it is collapsed,
   Streamlit puts the reopen control in the header — height: 0 would clip it. */
[data-testid="stHeader"]:not(:has([data-testid="stExpandSidebarButton"])) {
    height: 0;
    min-height: 0;
    overflow: hidden;
}
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    z-index: 1000000 !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 40px 24px 80px;
    max-width: 1080px;
}
[data-testid="stVerticalBlock"] { gap: 0.75rem; }
[data-testid="stElementContainer"] { margin: 0; }

/* ---- title ---------------------------------------------------- */
.title {
    font-size: 28px;
    font-weight: 650;
    letter-spacing: -0.03em;
    margin: 0 0 8px;
}
.subtitle {
    color: var(--muted);
    margin: 0 0 16px;
    max-width: 72ch;
}

/* ---- cards ---------------------------------------------------- */
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 16px 18px;
}
.card-title {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.02em;
    padding-bottom: 10px;
    margin-bottom: 2px;
    border-bottom: 1px solid var(--line);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.card .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    padding: 9px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
}
.card .row:last-of-type { border-bottom: 0; }
.card .k { color: var(--muted); }
.card .v { font-variant-numeric: tabular-nums; text-align: right; }
.card-foot {
    margin-top: 8px;
    padding-top: 10px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

[data-testid="stHorizontalBlock"] { align-items: flex-start; }
[data-testid="column"],
[data-testid="stColumn"] { min-width: 0 !important; }

/* ---- the two pictures ----------------------------------------- */
.pane-label { min-height: 2.7rem; margin-bottom: 0.5rem; }
.pane-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: -0.02em;
}
.pane-state {
    display: block;
    font-size: 12px;
    font-weight: 400;
    color: var(--muted);
    margin-top: 0.12rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stImage"] { display: flex; justify-content: flex-start; width: 100%; }
[data-testid="stImage"] img {
    border-radius: var(--radius);
    border: 1px solid var(--line);
    box-shadow: none;
    max-width: 100% !important;
    height: auto !important;
}
.slot-wrap { display: flex; justify-content: flex-start; width: 100%; }
.slot {
    border-radius: var(--radius);
    background: var(--card);
    border: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 13px;
    text-align: center;
    box-sizing: border-box;
}

/* ---- file row ------------------------------------------------- */
.filerow {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 16px;
}
.filerow .name { color: var(--text); font-weight: 600; }
.filerow .dot { color: var(--muted); }
.filerow .meta { font-variant-numeric: tabular-nums; }

/* ---- sidebar: the adjustment panel ---------------------------- */
[data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebarUserContent"] {
    min-width: 16rem;
    max-width: 21rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.15rem; }
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"],
[data-testid="stSidebarUserContent"] { padding-top: 1.6rem; }
.panel-title {
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.panel-sub {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 16px;
}

[data-testid="stSegmentedControl"] label p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
[data-testid="stSegmentedControl"] [data-testid="stWidgetLabel"] {
    margin-bottom: 0.25rem;
}
[data-testid="stSegmentedControl"] [role="radiogroup"],
[data-testid="stSegmentedControl"] [role="group"] {
    background: var(--fill);
    border-radius: 9px;
    padding: 2px;
    gap: 2px;
}
[data-testid="stSegmentedControl"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    min-height: 1.7rem !important;
    padding: 0.1rem 0.45rem !important;
    box-shadow: none !important;
    transition: background .18s ease;
}
[data-testid="stSegmentedControl"] button:hover {
    background: rgba(255,255,255,.55) !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
    background: var(--card) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    border: 1px solid var(--line) !important;
}
.adjust-row { margin-bottom: 0.55rem; }

/* ---- buttons and fields --------------------------------------- */
[data-testid="stBaseButton-tertiary"] {
    color: var(--accent) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0 !important;
}
[data-testid="stBaseButton-secondary"] {
    border-radius: var(--radius);
    border: 1px solid var(--line);
    font-size: 13px;
    background: var(--card);
    box-shadow: none;
}
[data-testid="stNumberInput"] label p {
    font-size: 13px !important;
    color: var(--muted) !important;
}
[data-testid="stNumberInput"] input {
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    border-radius: var(--radius);
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: none;
}
[data-testid="stExpander"] summary {
    font-size: 13px;
    color: var(--muted);
    font-weight: 600;
}

/* ---- upload --------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 1px dashed rgba(0, 0, 0, 0.14);
    border-radius: var(--radius);
    padding: 1.5rem 1.2rem;
    box-shadow: none;
    transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
    background: #f4f8ff;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: var(--radius) !important;
    font-size: 13px !important;
}

/* ---- verdict --------------------------------------------------- */
.verdict-wrap { display: flex; justify-content: flex-start; width: 100%; }
.verdict {
    margin-top: 12px;
    padding: 14px 16px;
    box-sizing: border-box;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    font-size: 13px;
    max-width: 100%;
}
.verdict.empty {
    background: transparent;
    border-color: transparent;
    color: var(--muted);
    font-size: 13px;
    text-align: left;
}
.pane-label .changed {
    color: var(--accent);
    font-weight: 500;
    margin-left: 0.4rem;
}
.verdict .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: 1px;
}
.verdict.real .dot { background: var(--success); }
.verdict.fake .dot { background: var(--danger); }
.verdict .call { font-weight: 600; letter-spacing: -0.02em; }
.verdict .pct {
    float: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
}
.verdict .meter {
    height: 4px;
    border-radius: 99px;
    background: var(--fill);
    margin: 10px 0 6px;
    overflow: hidden;
}
.verdict .fill { height: 100%; border-radius: 99px; }
.verdict.real .fill { background: var(--success); }
.verdict.fake .fill { background: var(--danger); }
.verdict .detail {
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
}
.verdict.real { background: #f3faf6; border-color: rgba(31, 138, 76, 0.18); }
.verdict.fake { background: #fff5f6; border-color: rgba(196, 30, 58, 0.18); }

.callout {
    background: #fff6eb;
    border: 1px solid rgba(178, 80, 0, 0.18);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin: 8px 0;
    color: var(--text);
    font-size: 13px;
}
.callout strong { display: block; margin-bottom: 4px; color: var(--warning); }

/* ---- detect action --------------------------------------------- */
.action-note { font-size: 13px; color: var(--muted); margin-top: 8px; }
[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    min-height: 2.4rem !important;
    box-shadow: none !important;
}
[data-testid="stBaseButton-primary"]:hover { background: #0077ed !important; }

.hint { font-size: 13px; color: var(--muted); }
.empty {
    color: var(--muted);
    font-size: 15px;
    text-align: left;
    max-width: 72ch;
    padding: 24px 0 8px;
}
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
}
footer { visibility: hidden; }
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] { display: none !important; }

/* Detect / Ablation — viewport top-right. Not in the document flow, so
   sidebar width and Ablation's full-bleed layout cannot move it. */
[data-testid="stMainBlockContainer"] [data-testid="stSegmentedControl"] {
    position: fixed;
    top: 12px;
    right: 20px;
    z-index: 1000001;
    width: auto !important;
    background: var(--bg);
    border-radius: 11px;
    padding: 2px;
}
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has([data-testid="stSegmentedControl"]) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

a.ablation-link {
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
}
a.ablation-link:hover { text-decoration: underline; }
</style>
"""


# ======================================================================
# Screen
# ======================================================================

def _card(photo: Photo) -> str:
    """A grouped list of facts, the way Settings shows them."""
    rows = "\n".join(
        f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in _facts(photo)
    )
    return (
        '<div class="card">'
        f'<div class="card-title">{photo.name}</div>'
        f"{rows}"
        f'<div class="card-foot">SHA-256 · {photo.digest[:12]}</div>'
        "</div>"
    )


def _read_chain() -> tuple[tuple[str, float], ...]:
    """Current selection, in ADJUSTMENTS order."""
    chain = []
    for adj in ADJUSTMENTS:
        value = st.session_state.get(f"knob_{adj.key}")
        if value in adj.values:          # both None and _OFF fall outside
            chain.append((adj.key, value))
    return tuple(chain)


def _reset() -> None:
    """Back to Off. An on_click callback, because writing session_state inside an
    `if st.button()` branch happens after the widgets exist and is rejected."""
    for adj in ADJUSTMENTS:
        st.session_state[f"knob_{adj.key}"] = _OFF
    st.session_state.pop("verdict", None)


def _panel() -> None:
    """The adjustment panel. Rendered before the pictures, so the chain read out
    of session_state is this run's widget state rather than the previous run's."""
    with st.sidebar:
        head, action = st.columns([1, 0.42], gap="small")
        with head:
            st.markdown('<div class="panel-title">Adjustments</div>',
                        unsafe_allow_html=True)
        with action:
            st.button("Reset", type="tertiary", on_click=_reset,
                      disabled=not _read_chain())
        st.markdown('<div class="panel-sub">Edits apply as you change them.</div>',
                    unsafe_allow_html=True)

        # Seed before the widgets exist: writing session_state *before* a widget is
        # instantiated is legal, and lets us skip default= without tripping
        # Streamlit's default-vs-session-state warning.
        for adj in ADJUSTMENTS:
            st.session_state.setdefault(f"knob_{adj.key}", _OFF)
        st.session_state.setdefault("seed", 0)

        for adj in ADJUSTMENTS:
            st.segmented_control(
                adj.label,
                options=[_OFF, *adj.values],
                selection_mode="single",
                format_func=lambda v, a=adj: (
                    _OFF if v not in a.values else a.fmt[a.values.index(v)]
                ),
                key=f"knob_{adj.key}",
                width="stretch",
                help=adj.note,
            )
            st.markdown('<div class="adjust-row"></div>', unsafe_allow_html=True)

        st.number_input(
            "Random seed",
            min_value=0, max_value=9999, step=1, key="seed",
            help="Noise and color are random. Same seed gives the same result.",
        )
        st.markdown(
            '<div class="hint">Applied in order: crop, resample, blur, noise, '
            "color, JPEG.</div>",
            unsafe_allow_html=True,
        )


def _intake() -> Photo | None:
    """Upload or drop. Collapses out of the way once a photo is loaded."""
    photo = st.session_state.get("photo")

    with st.expander("Choose a different photo" if photo else "Choose a photo",
                     expanded=photo is None):
        uploaded = st.file_uploader(
            "Drag a photo here", type=_ACCEPTED, accept_multiple_files=False,
            label_visibility="collapsed", key="uploader",
        )

    # Streamlit reruns the whole script on every interaction. Latching on file_id
    # decodes only when the file actually changes, and clearing the widget never
    # leaves the previous photo's results on screen.
    if uploaded is not None:
        token = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("token") != token:
            try:
                photo = _decode(uploaded.getvalue(), uploaded.name)
            except Exception as exc:  # noqa: BLE001 — say so plainly, do not crash
                st.error(f"That file could not be opened. {type(exc).__name__}: {exc}")
                st.session_state.pop("photo", None)
                st.session_state.pop("token", None)
                return None
            st.session_state["photo"] = photo
            st.session_state["token"] = token
            st.session_state.pop("verdict", None)
    elif st.session_state.get("token") is not None:
        st.session_state.pop("photo", None)
        st.session_state.pop("token", None)
        st.session_state.pop("verdict", None)

    return st.session_state.get("photo")


def _pictures(photo: Photo) -> None:
    """Original on the left, adjusted on the right, facts alongside.

    The grid is always three columns wide — with an empty slot of exactly the
    picture's footprint when nothing is applied — so a click never reflows the
    page and the original never moves.
    """
    chain = _read_chain()
    seed = int(st.session_state.get("seed", 0))

    st.markdown(
        f'<div class="filerow"><span class="name">{photo.name}</span>'
        f'<span class="dot">·</span><span class="meta">{photo.image.width} × '
        f'{photo.image.height}</span><span class="dot">·</span>'
        f'<span class="meta">{_fmt_bytes(len(photo.raw))}</span>'
        + ('<span class="dot">·</span><span class="meta">rotated by EXIF</span>'
           if photo.exif_rotated else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    original = _preview_copy(photo.image)
    width = _display_width(original, FIGURE_BOX)
    height = max(1, round(original.height * width / original.width))

    left, right, aside = st.columns([1, 1, 0.8], gap="large")

    key = (photo.digest, chain, seed)
    stored = st.session_state.get("verdict")
    records = stored["records"] if stored and stored["key"] == key else None

    with left:
        st.markdown(
            '<div class="pane-label"><div class="pane-title">Original</div>'
            '<div class="pane-state">unedited</div></div>',
            unsafe_allow_html=True,
        )
        st.image(original, width=width)
        st.markdown(_verdict_html(records[0] if records else None, width),
                    unsafe_allow_html=True)

    with right:
        state = _chain_label(chain) if chain else "nothing applied"
        st.markdown(
            f'<div class="pane-label"><div class="pane-title">Adjusted</div>'
            f'<div class="pane-state">{state}</div></div>',
            unsafe_allow_html=True,
        )

        if chain:
            st.image(
                _preview_copy(_apply_chain(photo.image, photo.digest, chain, seed)),
                width=width,
            )
            st.markdown(
                _verdict_html(records[1] if records and len(records) > 1 else None,
                              width),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="slot-wrap"><div class="slot" '
                f'style="width:{width}px;height:{height}px">'
                "Pick an adjustment</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(_verdict_html(None, width), unsafe_allow_html=True)

    with aside:
        st.markdown(
            '<div class="pane-label"><div class="pane-title">Details</div>'
            '<div class="pane-state">&nbsp;</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_card(photo), unsafe_allow_html=True)

    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    gpu_ok = _gpu_ready()
    action, note = st.columns([0.6, 2.2], gap="medium")
    with action:
        clicked = st.button(
            "Detect", type="primary", width="stretch", disabled=not gpu_ok,
        )
    with note:
        if gpu_ok:
            hint = (
                "Scores the original"
                + (" and the adjusted copy" if chain else "")
                + ". The first run loads the model, which takes a moment."
            )
        else:
            hint = (
                "No CUDA GPU on this machine — scoring is skipped. "
                "Upload and adjustments still work."
            )
        st.markdown(f'<div class="action-note">{hint}</div>', unsafe_allow_html=True)

    if clicked and gpu_ok:
        try:
            with st.spinner("Scoring…"):
                st.session_state["verdict"] = {
                    "key": key,
                    "records": _score(photo, chain, seed),
                }
        except FileNotFoundError as exc:
            st.error(f"Checkpoint missing. {exc}")
        except Exception as exc:  # noqa: BLE001 — surface it instead of a blank page
            st.error(f"Scoring failed. {type(exc).__name__}: {exc}")
        else:
            st.rerun()


# ======================================================================
# Ablation tab
# ======================================================================

def _ablation_page() -> None:
    """Show ablation/ablation.html as a full document (iframe, not st.html).

    ``st.html`` sanitizes the snippet and Streamlit's own CSS flattens
    ``<table>``. An iframe keeps the file's tables, charts, and scripts intact.
    """
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] { display: none !important; }
        [data-testid="stMainBlockContainer"] {
            padding: 0.5rem 0 0 !important;
            max-width: 100% !important;
        }
        [data-testid="stVerticalBlock"] { gap: 0.35rem; }
        [data-testid="stIFrame"] iframe,
        iframe[data-testid="stIFrame"] {
            border: none !important;
            border-radius: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.iframe(ABLATION_HTML, width="stretch", height="content")


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    st.set_page_config(page_title="TraceLens", page_icon="◉",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.session_state.setdefault("nav_page", "Detect")
    page = st.segmented_control(
        "Page",
        options=["Detect", "Ablation"],
        key="nav_page",
        label_visibility="collapsed",
        width="content",
    )

    if page == "Ablation":
        _ablation_page()
        return

    st.markdown('<div class="title">TraceLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Tell real photos from AI-generated ones, '
                "even after resharing has chewed them up.</div>",
                unsafe_allow_html=True)

    _panel()
    photo = _intake()
    if photo is None:
        st.markdown('<div class="empty">Choose a photo to get started.</div>',
                    unsafe_allow_html=True)
        return

    _pictures(photo)


if __name__ == "__main__":
    main()
