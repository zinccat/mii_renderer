import gradio as gr
import os, glob, re, tempfile, json, struct, sys, trimesh
import numpy as np
from PIL import Image, ImageFilter
from trimesh.scene.scene import append_scenes
from trimesh.visual.material import PBRMaterial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mii_metadata as mm

_ASSETS      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
GLB_DIR      = os.path.join(_ASSETS, "glb")
MIIPARTS_DIR = os.path.join(_ASSETS, "miiparts_png")

# ── Color tables ───────────────────────────────────────────────────────────────
# ── Gamma helpers (sRGB ↔ linear) ────────────────────────────────────────────
# The game stores PBR colours as LINEAR floats and renders through an sRGB
# framebuffer. To match its output we must:
#   * feed glTF `baseColorFactor` linear values (per glTF 2.0 §3.9.3.1);
#   * do 2D-tinting math in linear space, then encode to sRGB before handing
#     to PIL (which is sRGB-encoded).
# Multiplying sRGB-encoded values directly (the previous approach) is wrong
# for partial-coverage / partial-alpha pixels — it produces darker, less
# saturated results than the game on anti-aliased edges and gradient masks.
def _srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(x):
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def _rgb_table(rgb_lin_list):
    """Build {label: [r, g, b, 1.0]} from (label, linear-RGB) tuples — fed
    straight into glTF baseColorFactor (which is linear per glTF 2.0)."""
    return {label: [float(r), float(g), float(b), 1.0]
            for label, (r, g, b) in rgb_lin_list}

# Skin tones — `nn::mii::detail::FacelineColorTable` @ sdk.nso 0x97dff0
# (10 entries × 32 B, sRGB row at +12). Display order is the
# `FacelineColorOrderIndexToColorIndex` permutation @ 0x97e130:
# [0, 7, 1, 4, 5, 6, 3, 2, 8, 9]. Labels are UI-side text only.
#
# NB: The Switch 2 editor's `SkinColorOrder` bgyml lists 14 indices
# (0x64..0x71 = 100..113) — i.e., 4 more skin slots than this 10-entry
# SDK table provides. Static analysis can't locate the extended palette:
# (a) main.nso doesn't contain the SDK's faceline floats, so there's no
#     extension copy embedded there; (b) `GetSkinColor(int, GammaType)`
#     resolves via dynsym to a 4-byte stub at sdk VA 0x3d5a10 that
#     PLT-jumps through BSS slot 0xb73ac8 (lazy-bound at runtime);
#     (c) the bgyml permutation isn't stored as a u8/u32 byte sequence
#     in any segment we searched; (d) no `cmp wN,#114` + `sub wN,#100`
#     bounds-check signature found in sdk text. Resolving the missing 4
#     slots would require dynamic analysis (RAM dump) or an unexamined
#     romfs file. We ship the 10-entry SDK palette as the source of truth.
# ── Color tables (NSO-sourced) ───────────────────────────────────────────────
# Tuples are LINEAR floats — the row at +0 of each entry, NOT the +12 sRGB row.
# Pulled directly from sdk.nso:
#   nn::mii::detail::CommonColorTable   @ VA 0x97c9ac (100 entries × 32B)
#   nn::mii::detail::UpperLipColorTable @ VA 0x97d690 (100 entries × 24B)
#   nn::mii::detail::FacelineColorTable @ VA 0x97dff0 ( 10 entries × 32B)
#   nn::mii::detail::{Mole,Noseline,Wrinkle,EyeWhite,Teeth}Color (single 24B)
# Each entry is `Color3f linear; Color3f srgb;`. The game's shaders take the
# linear row, multiply in linear space, and encode to sRGB via the
# framebuffer. We must do the same — using the sRGB row directly was the bug
# that made our colours look wrong vs. the game (especially anti-aliased
# edges where partial coverage exposes the gamma mismatch). Indexing comes
# from `Editor/MiiEditorColorOrder/System.editor__mii__ColorOrder.bgyml`.
# Labels are UI-side picker text only (not from the binary).
_NSO_SKIN_RGB_ORDERED = [
    ("Ivory",     (1.000000, 0.651406, 0.417885)),  # FacelineColorTable[0]
    ("Peach",     (1.000000, 0.558340, 0.274677)),  # FacelineColorTable[7]
    ("Tan",       (1.000000, 0.467784, 0.147027)),  # FacelineColorTable[1]
    ("Brown",     (0.417885, 0.082283, 0.022174)),  # FacelineColorTable[4]
    ("Dark Brown",(0.124772, 0.025187, 0.009134)),  # FacelineColorTable[5]
    ("Pale",      (1.000000, 0.514918, 0.376262)),  # FacelineColorTable[6]
    ("Warm",      (1.000000, 0.401978, 0.262251)),  # FacelineColorTable[3]
    ("Bronze",    (0.730461, 0.191202, 0.054480)),  # FacelineColorTable[2]
    ("Deep",      (0.262251, 0.045186, 0.016807)),  # FacelineColorTable[8]
    ("Darkest",   (0.045186, 0.026241, 0.016807)),  # FacelineColorTable[9]
]
SKIN_TABLE = _rgb_table(_NSO_SKIN_RGB_ORDERED)   # values fed to baseColorFactor

# Glasses frame colours — `CommonColorTable` indices per GlassColorOrder.
GLASS_TABLE = _rgb_table([
    ("Black",  (0.000000, 0.000000, 0.000000)),   # CommonColorTable[0x08]
    ("Gray",   (0.187821, 0.162029, 0.138432)),   # CommonColorTable[0x12]
    ("Brown",  (0.116971, 0.039546, 0.005182)),   # CommonColorTable[0x0e]
    ("Amber",  (0.391573, 0.116971, 0.000000)),   # CommonColorTable[0x11]
    ("Red",    (0.391573, 0.005182, 0.002428)),   # CommonColorTable[0x0f]
    ("Navy",   (0.014444, 0.029557, 0.138432)),   # CommonColorTable[0x10]
])

# Full 100-entry hair / 2D-feature palette in editor display order.
# Source: `CommonColorTable` @ sdk.nso 0x97c9ac (linear row at +0) reordered
# by `CommonColorOrderIndexToColorIndex` @ 0x97d62c. The previous 8-entry
# subset (HairColorOrder permutation [8,4,5,1,2,3,6,7]) was the editor's
# legacy "quick pick" — Switch 2 surfaces the entire 100-color palette.
_NSO_HAIR_COLORS = [
    (0.107023, 0.009134, 0.003035),  # slot 0  table[ 2]
    (0.230740, 0.019382, 0.019382),  # slot 1  table[24]
    (0.132868, 0.045186, 0.025187),  # slot 2  table[10]
    (0.262251, 0.080220, 0.051269),  # slot 3  table[23]
    (0.391573, 0.005182, 0.002428),  # slot 4  table[15]
    (0.871367, 0.003677, 0.002428),  # slot 5  table[20]
    (0.913099, 0.064803, 0.064803),  # slot 6  table[21]
    (1.000000, 0.171441, 0.132868),  # slot 7  table[25]
    (1.000000, 0.381326, 0.381326),  # slot 8  table[26]
    (1.000000, 0.527115, 0.491021),  # slot 9  table[27]
    (0.171441, 0.027321, 0.043735),  # slot 10 table[28]
    (0.318547, 0.013702, 0.046665),  # slot 11 table[29]
    (0.254152, 0.008568, 0.048172),  # slot 12 table[30]
    (0.462077, 0.048172, 0.054480),  # slot 13 table[31]
    (0.571125, 0.012983, 0.093059),  # slot 14 table[32]
    (0.434154, 0.086500, 0.219526),  # slot 15 table[33]
    (0.571125, 0.088656, 0.155926),  # slot 16 table[34]
    (0.955974, 0.177888, 0.309469),  # slot 17 table[35]
    (0.973445, 0.412543, 0.584078),  # slot 18 table[36]
    (1.000000, 0.584078, 0.686686),  # slot 19 table[37]
    (0.030713, 0.011612, 0.051269),  # slot 20 table[38]
    (0.038204, 0.021219, 0.046665),  # slot 21 table[39]
    (0.072272, 0.009134, 0.074214),  # slot 22 table[40]
    (0.158961, 0.054480, 0.450786),  # slot 23 table[41]
    (0.234551, 0.107023, 0.479320),  # slot 24 table[42]
    (0.527115, 0.226966, 0.603827),  # slot 25 table[43]
    (0.391573, 0.291771, 0.584078),  # slot 26 table[44]
    (0.558340, 0.412543, 0.791298),  # slot 27 table[45]
    (0.854993, 0.514918, 0.955974),  # slot 28 table[46]
    (0.644480, 0.558340, 0.846874),  # slot 29 table[47]
    (0.009721, 0.013702, 0.051269),  # slot 30 table[48]
    (0.014444, 0.029557, 0.138432),  # slot 31 table[16]
    (0.006049, 0.049707, 0.132868),  # slot 32 table[49]
    (0.061246, 0.088656, 0.391573),  # slot 33 table[12]
    (0.023153, 0.223228, 0.658375),  # slot 34 table[50]
    (0.095307, 0.456411, 0.887923),  # slot 35 table[51]
    (0.194618, 0.558340, 0.730461),  # slot 36 table[52]
    (0.250158, 0.381326, 0.955974),  # slot 37 table[53]
    (0.230740, 0.508881, 0.955974),  # slot 38 table[54]
    (0.356400, 0.768151, 1.000000),  # slot 39 table[55]
    (0.003347, 0.027321, 0.036889),  # slot 40 table[56]
    (0.000304, 0.046665, 0.043735),  # slot 41 table[57]
    (0.004025, 0.078187, 0.099899),  # slot 42 table[58]
    (0.016807, 0.132868, 0.124772),  # slot 43 table[59]
    (0.039546, 0.162029, 0.097587),  # slot 44 table[13]
    (0.029557, 0.208637, 0.262251),  # slot 45 table[60]
    (0.078187, 0.423268, 0.434154),  # slot 46 table[61]
    (0.194618, 0.552011, 0.341915),  # slot 47 table[62]
    (0.212231, 0.658375, 0.527115),  # slot 48 table[63]
    (0.242281, 0.783538, 0.467784),  # slot 49 table[64]
    (0.003035, 0.068478, 0.035601),  # slot 50 table[65]
    (0.056128, 0.194618, 0.000000),  # slot 51 table[66]
    (0.000607, 0.177888, 0.122139),  # slot 52 table[67]
    (0.036889, 0.318547, 0.162029),  # slot 53 table[68]
    (0.070360, 0.417885, 0.010330),  # slot 54 table[69]
    (0.287441, 0.520996, 0.003035),  # slot 55 table[70]
    (0.124772, 0.571125, 0.246201),  # slot 56 table[71]
    (0.341915, 0.745404, 0.054480),  # slot 57 table[72]
    (0.304987, 0.730461, 0.208637),  # slot 58 table[73]
    (0.496933, 0.887923, 0.401978),  # slot 59 table[74]
    (0.076185, 0.048172, 0.005182),  # slot 60 table[ 5]
    (0.116971, 0.111933, 0.029557),  # slot 61 table[11]
    (0.318547, 0.291771, 0.024158),  # slot 62 table[75]
    (0.381326, 0.300544, 0.124772),  # slot 63 table[76]
    (0.603827, 0.527115, 0.040915),  # slot 64 table[77]
    (0.603827, 0.485150, 0.242281),  # slot 65 table[78]
    (0.693872, 0.603827, 0.223228),  # slot 66 table[79]
    (0.665388, 0.693872, 0.158961),  # slot 67 table[80]
    (0.665388, 0.791298, 0.226966),  # slot 68 table[81]
    (0.686686, 0.955974, 0.337164),  # slot 69 table[82]
    (0.116971, 0.039546, 0.005182),  # slot 70 table[14]
    (0.205079, 0.059511, 0.000000),  # slot 71 table[83]
    (0.246201, 0.097587, 0.009134),  # slot 72 table[ 6]
    (0.391573, 0.116971, 0.000000),  # slot 73 table[17]
    (0.630757, 0.351533, 0.068478),  # slot 74 table[ 7]
    (0.791298, 0.496933, 0.194618),  # slot 75 table[84]
    (0.991102, 0.760525, 0.068478),  # slot 76 table[85]
    (0.955974, 0.730461, 0.223228),  # slot 77 table[86]
    (0.930111, 0.822786, 0.332452),  # slot 78 table[87]
    (0.955974, 0.938686, 0.327778),  # slot 79 table[88]
    (0.051269, 0.014444, 0.005182),  # slot 80 table[ 1]
    (0.201556, 0.042311, 0.006995),  # slot 81 table[ 3]
    (0.381326, 0.074214, 0.012983),  # slot 82 table[89]
    (0.686686, 0.084376, 0.002428),  # slot 83 table[19]
    (1.000000, 0.304987, 0.004025),  # slot 84 table[90]
    (0.637597, 0.327778, 0.141263),  # slot 85 table[91]
    (0.871367, 0.323143, 0.174647),  # slot 86 table[22]
    (1.000000, 0.445201, 0.132868),  # slot 87 table[92]
    (1.000000, 0.539480, 0.262251),  # slot 88 table[93]
    (0.783538, 0.623961, 0.439657),  # slot 89 table[94]
    (0.000000, 0.000000, 0.000000),  # slot 90 table[ 8]   black
    (0.026241, 0.021219, 0.021219),  # slot 91 table[ 0]
    (0.052861, 0.052861, 0.052861),  # slot 92 table[95]
    (0.149960, 0.162029, 0.162029),  # slot 93 table[ 9]
    (0.187821, 0.162029, 0.138432),  # slot 94 table[18]
    (0.187821, 0.187821, 0.215861),  # slot 95 table[ 4]   gray
    (0.327778, 0.327778, 0.327778),  # slot 96 table[96]
    (0.514918, 0.514918, 0.514918),  # slot 97 table[97]
    (0.715694, 0.679543, 0.610496),  # slot 98 table[98]
    (1.000000, 1.000000, 1.000000),  # slot 99 table[99]   white
]
_NSO_HAIR_LABELS = [f"{i:02d}" for i in range(len(_NSO_HAIR_COLORS))]

# Eye iris body color (B channel of the eye sprite); EyeColorOrder = [8,9,10,11,13,12]
_NSO_EYE_B_COLORS = [
    (0.000000, 0.000000, 0.000000),  # CommonColorTable[8]   — Black
    (0.149960, 0.162029, 0.162029),  # CommonColorTable[9]   — Gray
    (0.132868, 0.045186, 0.025187),  # CommonColorTable[10]  — Brown
    (0.116971, 0.111933, 0.029557),  # CommonColorTable[11]  — Olive
    (0.039546, 0.162029, 0.097587),  # CommonColorTable[13]  — Green
    (0.061246, 0.088656, 0.391573),  # CommonColorTable[12]  — Blue
]
_NSO_EYE_B_LABELS = ["Black","Gray","Brown","Olive","Green","Blue"]
_NSO_EYE_G_COLOR  = (1.0, 1.0, 1.0)            # EyeWhiteColor @ 0x97e2c4

# Mouth lower-lip (R channel); MouthColorOrder = [0x13..0x17]
_NSO_MOUTH_R_COLORS = [
    (0.686686, 0.084376, 0.002428),  # CommonColorTable[0x13] — Orange-Red
    (0.871367, 0.003677, 0.002428),  # CommonColorTable[0x14] — Bright Red
    (0.913099, 0.064803, 0.064803),  # CommonColorTable[0x15] — Pink
    (0.871367, 0.323143, 0.174647),  # CommonColorTable[0x16] — Salmon
    (0.262251, 0.080220, 0.051269),  # CommonColorTable[0x17] — Dark Red
]
# Mouth upper-lip (G channel) — separate UpperLipColorTable, same 5 indices
_NSO_MOUTH_G_COLORS = [
    (0.223228, 0.029557, 0.009134),  # UpperLipColorTable[0x13]
    (0.187821, 0.003677, 0.003677),  # UpperLipColorTable[0x14]
    (0.246201, 0.014444, 0.021219),  # UpperLipColorTable[0x15]
    (0.715694, 0.187821, 0.080220),  # UpperLipColorTable[0x16]
    (0.061246, 0.012983, 0.003035),  # UpperLipColorTable[0x17]
]
_NSO_MOUTH_B_COLOR = (1.0, 1.0, 1.0)             # TeethColor @ 0x97e2f4
_NSO_MOUTH_LABELS  = ["Orange-Red","Bright Red","Pink","Salmon","Dark Red"]

_NSO_NOSELINE_COLOR = (0.000000, 0.000000, 0.000000)  # NoselineColor @ 0x97e294
_NSO_MOLE_COLOR     = (0.006049, 0.004777, 0.004777)  # MoleColor     @ 0x97e27c
_NSO_WRINKLE_COLOR  = (0.000000, 0.000000, 0.000000)  # WrinkleColor  @ 0x97e2ac

# Demo prefix → MiiParts.bntx category name. "Beard" (demo) → "BeardShort" (asset).
_MIIPARTS_CAT_MAP = {
    "Eye": "Eye", "Eyebrow": "Eyebrow", "Mouth": "Mouth", "Mole": "Mole",
    "Mustache": "Mustache", "Beard": "BeardShort",
    "EyelashUpper": "EyelashUpper", "EyelashLower": "EyelashLower",
    "EyelidUpper":  "EyelidUpper",  "EyelidLower":  "EyelidLower",
    "Highlight":    "Highlight",
    "MakeUpper":    "MakeUpper",    "MakeLower":    "MakeLower",
    "WrinkleUpper": "WrinkleUpper", "WrinkleLower": "WrinkleLower",
}

def _bgyml_offset_rot(name):
    """Per-shape baseline tilt (raw int from bgyml `OffsetRotate`).
    Callers convert to degrees via `* 11.25°` per EMPIRICAL_AUDIT.md §D.
    Used for both Eye and Eyebrow shapes."""
    if not name or name == "(none)":
        return 0
    v = (mm.parts().get(name) or {}).get("OffsetRotate")
    return int(v) if isinstance(v, int) else 0

def _mustache_rotate_deg(name):
    """Per-mustache baseline rotation from `RotateAxis.Y` (bgyml radians,
    ~ -0.2 .. +0.07). Returns degrees so it can be added to the existing
    `rot` parameter passed to _rotate_sprite."""
    if not name or name == "(none)":
        return 0.0
    v = (mm.parts().get(name) or {}).get("RotateAxis")
    if not isinstance(v, dict):
        return 0.0
    try:
        import math
        return math.degrees(float(v.get("Y", 0.0)))
    except (TypeError, ValueError):
        return 0.0

def _eye_size_for_expr(name):
    """Per-shape eye size scalar (`SizeForExpression`, ~0.6-1.0).
    Used to shrink certain eye shapes that are authored larger in the
    sprite sheet than their on-screen size."""
    if not name or name == "(none)":
        return 1.0
    v = (mm.parts().get(name) or {}).get("SizeForExpression")
    try:
        return float(v) if v is not None else 1.0
    except (TypeError, ValueError):
        return 1.0

def _parse_feat_index(name):
    """Extract numeric index from name like 'Eye054' → 54.

    Some bgyml entries declare a `TextureName` that differs from their
    `FileName` (notably Mole: FileName=Mole00, TextureName=Mole01); the
    extracted PNG is named after the texture, so prefer that when the
    metadata is loaded — otherwise the index parsed from FileName won't
    line up with the PNG registry."""
    entry = mm.parts().get(name) if name else None
    src = (entry or {}).get("TextureName") or name or ""
    m = re.search(r'(\d+)$', src)
    return int(m.group(1)) if m else 0

# ── MiiParts.bntx PNG loader (extracted via extract_all_miiparts.py) ──────────
# Naming: Eye{idx:03d}.png, Eyebrow{idx:02d}.png, BeardShort{idx:02d}.png, ...
# Sprites are returned at the source's NATIVE pixel size (no LANCZOS downscale).
_MIIPARTS_REG = {}  # cat -> {idx: (filename, abs_path)}
def _miiparts_reg(cat):
    """Return {idx: (basename_no_ext, path)} for one category, scanning once."""
    if cat in _MIIPARTS_REG:
        return _MIIPARTS_REG[cat]
    out = {}
    pat = re.compile(rf'^{re.escape(cat)}(\d+)\.png$')
    if os.path.isdir(MIIPARTS_DIR):
        for f in sorted(os.listdir(MIIPARTS_DIR)):
            m = pat.match(f)
            if m:
                out[int(m.group(1))] = (f[:-4], os.path.join(MIIPARTS_DIR, f))
    _MIIPARTS_REG[cat] = out
    return out

def _miiparts_path(cat, idx):
    entry = _miiparts_reg(cat).get(idx)
    return entry[1] if entry else None

# H-flip in loader for these categories. EyelashUpper is excluded — its texture
# is already authored in the right orientation per user empirical feedback.
_H_FLIP_CATS = {"EyelashLower", "EyelidUpper", "EyelidLower", "Highlight",
                "MakeLower", "WrinkleUpper", "WrinkleLower"}
_V_FLIP_CATS = {"MakeLower", "WrinkleLower"}

def _mouth_default_no_lip(name):
    """Per-mouth default for the no-lipstick switch.

    Mouths that ship a `Mouth0XX_NoLip.png` companion default to WITH
    lipstick (toggle off → regular file's lip colour shows). Mouths
    that DON'T ship one default to NO lipstick (toggle off → R/G are
    stripped so we don't apply a lip colour onto a mouth designed
    without one). Selecting a different mouth resets the toggle to
    the new mouth's default below."""
    if not name or name == "(none)":
        return False
    idx = _parse_feat_index(name)
    p = _miiparts_path("Mouth", idx)
    return bool(p and not os.path.exists(p[:-4] + "_NoLip.png"))


def _eye_colored_icon(name, eye_color):
    """Generate a gallery icon for an eye part: composite the colour layer
    (`<name>color_Uit.png`, white iris area) with the outline overlay
    (`<name>_Uit.png`, black ring + pupil) such that the white pixels of
    the colour layer become the user's selected eye colour, and the
    outline overlays on top opaque. Cached per (name, eye_color)."""
    if not name or name == "(none)":
        return None
    base = "MiiEditor_Face_" + name
    color_path = os.path.join(_EDITOR_ICONS, f"MiiEditorIcon_{base}color_Uit.png")
    outline_path = os.path.join(_EDITOR_ICONS, f"MiiEditorIcon_{base}_Uit.png")
    if not (os.path.exists(color_path) and os.path.exists(outline_path)):
        return None
    cache_dir = os.path.join(tempfile.gettempdir(), "mii_face_renderer_eye_icons")
    os.makedirs(cache_dir, exist_ok=True)
    if isinstance(eye_color, (int, np.integer)):
        lin = _NSO_EYE_B_COLORS[int(eye_color)]
        tag = f"i{int(eye_color)}"
    else:
        lin = tuple(eye_color)
        srgb_b = tuple(int(round(max(0.0, min(1.0,
            (12.92*c) if c <= 0.0031308 else (1.055 * c**(1/2.4) - 0.055)
        )) * 255)) for c in lin[:3])
        tag = f"c{srgb_b[0]:02x}{srgb_b[1]:02x}{srgb_b[2]:02x}"
    out_path = os.path.join(cache_dir, f"{name}_{tag}_v4.png")
    if os.path.exists(out_path):
        return out_path
    color_img = np.asarray(Image.open(color_path).convert("RGBA"), dtype=np.float32) / 255.0
    outline_img = np.asarray(Image.open(outline_path).convert("RGBA"), dtype=np.float32) / 255.0
    h, w = color_img.shape[:2]
    if outline_img.shape[:2] != (h, w):
        outline_img = np.asarray(Image.open(outline_path).convert("RGBA").resize((w, h)), dtype=np.float32) / 255.0
    # The colour layer encodes the iris-fill mask: WHITE pixels mark
    # the iris area (replaced with the selected eye colour) and BLACK
    # pixels mark the surrounding sclera/background (rendered as plain
    # white — the structural pupil ring + lash arc come from the
    # outline overlay below, so the colour layer's black isn't load-
    # bearing on its own and looks wrong if left dark). Mix in linear
    # space so antialiased edges between the two regions blend cleanly.
    cr_lin    = np.array(lin[:3], dtype=np.float32)
    white_lin = np.ones(3, dtype=np.float32)
    rgb_layer_lin = np.array([_srgb_to_linear(color_img[..., k])
                              for k in range(3)]).transpose(1, 2, 0)
    luma = rgb_layer_lin.mean(axis=-1, keepdims=True)
    coloured_lin  = luma * cr_lin + (1.0 - luma) * white_lin
    coloured_srgb = _linear_to_srgb(coloured_lin)
    base_rgba = np.concatenate([coloured_srgb, color_img[..., 3:4]], axis=-1)
    # Overlay the outline image on top using its alpha.
    over_alpha = outline_img[..., 3:4]
    out_rgb = (1.0 - over_alpha) * base_rgba[..., :3] + over_alpha * outline_img[..., :3]
    out_a = np.maximum(base_rgba[..., 3:4], over_alpha)
    out = np.concatenate([out_rgb, out_a], axis=-1)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)
    return out_path


def _mouth_colored_icon(name, mouth_color):
    """Generate a gallery icon from the Mouth<NNN>.png texture so the
    thumbnail matches the rendered face: with-lip mouths get the lip
    colour applied, no-lip mouths (no `_NoLip` companion) drop R/G
    just like the renderer's default for them. Cached per
    (name, mouth_color)."""
    if not name or name == "(none)":
        return None
    idx = _parse_feat_index(name)
    src = _miiparts_path("Mouth", idx)
    if not src or not os.path.exists(src):
        return None
    has_nolip = os.path.exists(src[:-4] + "_NoLip.png")
    cache_dir = os.path.join(tempfile.gettempdir(), "mii_face_renderer_mouth_icons")
    os.makedirs(cache_dir, exist_ok=True)
    # `mouth_color` may be a legacy int index OR a linear (R,G,B) tuple
    # from the swatch picker. Normalise to a linear tuple here and use a
    # short hash for the cache filename so tuples don't blow up the path.
    if isinstance(mouth_color, (int, np.integer)):
        _lower = _NSO_MOUTH_R_COLORS[int(mouth_color)]
        _upper = _NSO_MOUTH_G_COLORS[int(mouth_color)]
        suffix = f"c{int(mouth_color)}" if has_nolip else "nolip"
    elif (len(mouth_color) == 2
          and hasattr(mouth_color[0], "__len__") and len(mouth_color[0]) == 3):
        # (lower, upper) pair from the favourites click
        _lower = tuple(mouth_color[0])
        _upper = tuple(mouth_color[1])
        tag = (f"{int(_lower[0]*255):02x}{int(_lower[1]*255):02x}{int(_lower[2]*255):02x}"
               f"_{int(_upper[0]*255):02x}{int(_upper[1]*255):02x}{int(_upper[2]*255):02x}")
        suffix = f"c{tag}" if has_nolip else "nolip"
    else:
        _lower = _upper = tuple(mouth_color)
        tag = f"{int(_lower[0]*255):02x}{int(_lower[1]*255):02x}{int(_lower[2]*255):02x}"
        suffix = f"c{tag}" if has_nolip else "nolip"
    out_path = os.path.join(cache_dir, f"{name}_{suffix}_v3.png")
    if os.path.exists(out_path):
        return out_path
    arr = np.asarray(Image.open(src).convert("RGBA"), dtype=np.float32) / 255.0
    r, g, b, a = arr[..., 0:1], arr[..., 1:2], arr[..., 2:3], arr[..., 3:4]
    # Coverage masks treated as linear (the artist's intent for masks);
    # tints are linear from the binary; result encoded to sRGB at the end.
    cB = np.array(_NSO_MOUTH_B_COLOR, dtype=np.float32)
    if has_nolip:
        cR = np.array(_lower, dtype=np.float32)
        cG = np.array(_upper, dtype=np.float32)
        rgb_lin = r * cR + g * cG + b * cB
        alpha = a
    else:
        # Default render strips R/G — match here. Lipstick-only pixels
        # become transparent; black outline (R=G=B=0, A>0) survives.
        drop = ((r > 0) | (g > 0)) & (b == 0)
        rgb_lin = b * cB
        alpha = a * (1.0 - drop.astype(np.float32))
    rgb = _linear_to_srgb(rgb_lin)
    out = np.concatenate([rgb, alpha], axis=-1)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)
    return out_path


def _load_miiparts_sprite(cat, idx, eye_color=4, mouth_color=0, hair_color=0, scale=1.0,
                          mouth_drop_lips=False, eye_use_red=False,
                          eyeline_color=(1.0, 1.0, 1.0),
                          eyebrow_color=None, beard2d_color=None, mustache_color=None):
    """Load a MiiParts texture and apply per-category color modulation.

    Eye rule: R→transparent (skipped), G→white sclera, B→user-selected iris.
    Mouth rule: RGB-layered (R=lower lip, G=upper lip, B=teeth).
    Hair-like (Eyebrow/BeardShort/Mustache): BC4 mask in R, fill with hair color.
    Mole: BC4 mask in R, fill with dark mole color.
    Eyelash/Eyelid/Highlight/Make/Wrinkle: BC4 mask in R, fill black.

    V-flip note: BNTX uses GL-convention (v=0 at bottom) but the extractor
    treats lash/lid/highlight textures as PIL-convention (row 0 at top).
    The engine compositor assumes GL convention, so these specific textures
    need V-flipping at load time to align with the engine's expected layout
    — verified empirically (V-flipped result matches typical Mii lash arc
    above the iris). Other categories (eye/eyebrow/mouth/etc.) render
    correctly without V-flip, so the extractor likely handled them with a
    different convention.
    """
    path = _miiparts_path(cat, idx)
    if path is None or not os.path.exists(path):
        return None
    # Mouth lipstick handling. The `mouth_drop_lips` flag is the direct
    # render request: True → no lipstick, False → with lipstick.
    # 8 mouths ship a `Mouth0XX_NoLip.png` companion (pre-baked teeth-
    # only); use that when no-lip is requested. Otherwise strip R/G in
    # the colour-mix path below.
    if cat == "Mouth" and mouth_drop_lips:
        nolip_path = path[:-4] + "_NoLip.png"
        if os.path.exists(nolip_path):
            path = nolip_path
            mouth_drop_lips = False     # texture is already lip-free
    arr = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    if cat in _H_FLIP_CATS:
        arr = arr[:, ::-1, :].copy()
    if cat in _V_FLIP_CATS:
        arr = arr[::-1, :, :].copy()
    h, w = arr.shape[:2]
    r_ch = arr[:, :, 0:1]
    g_ch = arr[:, :, 1:2]
    b_ch = arr[:, :, 2:3]
    a_ch = arr[:, :, 3:4]

    # Tint math is in linear space (the binary stores linear floats and the
    # game's shaders multiply in linear). The result is encoded to sRGB at
    # the bottom of the branch so PIL gets sRGB-encoded bytes.
    if cat == "Eye":
        # Channels are intensities (0..1): R = highlight, G = sclera, B = iris.
        # Toggle OFF → R contributes nothing to rgb AND R-only pixels fade
        # to transparent in proportion to R intensity, so high-R highlight
        # pixels disappear while pure-black outline pixels (R=G=B=0, A>0)
        # stay fully opaque. Toggle ON → R adds a white highlight scaled by
        # intensity, alpha unchanged.
        cG = np.array(_NSO_EYE_G_COLOR,             dtype=np.float32)
        # eye_color is now a linear (R,G,B) tuple from the swatch picker;
        # legacy int index still accepted as a fallback (tests / API).
        if isinstance(eye_color, (int, np.integer)):
            _ec = _NSO_EYE_B_COLORS[int(eye_color)]
        else:
            _ec = eye_color
        cB = np.array(_ec, dtype=np.float32)
        rgb_lin = g_ch * cG + b_ch * cB
        if eye_use_red:
            cE = np.array(eyeline_color, dtype=np.float32)
            rgb_lin = rgb_lin + r_ch * cE     # tinted highlight
            alpha   = a_ch
        else:
            # Only R-only pixels get faded — keep mixed pixels (G or B > 0)
            # at full alpha so the iris/sclera aren't accidentally cut.
            r_only_mask = ((g_ch == 0) & (b_ch == 0)).astype(np.float32)
            alpha = a_ch * (1.0 - r_ch * r_only_mask)
    elif cat == "Mouth":
        cB = np.array(_NSO_MOUTH_B_COLOR, dtype=np.float32)
        if mouth_drop_lips:
            # Strip the R/G channels (lower + upper lip colour). Pixel
            # categories in the source PNG:
            #   R>0 or G>0  → lipstick fill   — make TRANSPARENT
            #   B>0          → teeth/iris fill — keep, tinted by cB
            #   R=G=B=0, A>0 → black outline   — keep, opaque black
            drop    = ((r_ch > 0) | (g_ch > 0)) & (b_ch == 0)
            rgb_lin = b_ch * cB
            alpha   = a_ch * (1.0 - drop.astype(np.float32))
        else:
            # mouth_color is now a linear (R,G,B) tuple from the swatch
            # picker. The legacy lower-vs-upper-lip distinction (R from
            # CommonColorTable, G from UpperLipColorTable) is dropped:
            # the same picked colour applies to both. Legacy int index
            # still accepted.
            if isinstance(mouth_color, (int, np.integer)):
                _lower = _NSO_MOUTH_R_COLORS[int(mouth_color)]
                _upper = _NSO_MOUTH_G_COLORS[int(mouth_color)]
            elif (len(mouth_color) == 2
                  and hasattr(mouth_color[0], "__len__") and len(mouth_color[0]) == 3):
                _lower, _upper = mouth_color[0], mouth_color[1]
            else:
                _lower = _upper = mouth_color
            cR = np.array(_lower, dtype=np.float32)
            cG = np.array(_upper, dtype=np.float32)
            rgb_lin = r_ch * cR + g_ch * cG + b_ch * cB
            alpha   = a_ch
    else:
        # Hair-tinted 2D features can each pick their own colour; fall
        # back to the legacy single hair_color when a per-cat one isn't
        # passed.
        per_cat = {"Eyebrow": eyebrow_color, "BeardShort": beard2d_color,
                   "Mustache": mustache_color}
        if cat in per_cat:
            ci   = per_cat[cat] if per_cat[cat] is not None else hair_color
            fill = np.array(_NSO_HAIR_COLORS[ci], dtype=np.float32)
        elif cat == "Mole":
            fill = np.array(_NSO_MOLE_COLOR, dtype=np.float32)
        elif cat == "Highlight":
            fill = np.array((1.0, 1.0, 1.0), dtype=np.float32)
        elif cat in ("WrinkleUpper", "WrinkleLower"):
            fill = np.array(_NSO_WRINKLE_COLOR, dtype=np.float32)
        else:
            fill = np.array(_NSO_NOSELINE_COLOR, dtype=np.float32)
        rgb_lin = np.ones((h, w, 3), dtype=np.float32) * fill
        # Most BeardShort textures (02, 05, 10, …) carry their coverage in
        # the G channel — only 00/01/04/13 use R. Mustache textures all
        # use R. Take the max of R and G so the same code path handles
        # both encodings; per-pixel taken-from-max preserves intensity.
        if cat == "BeardShort":
            alpha = np.maximum(r_ch, g_ch)
        else:
            alpha = r_ch
    rgb = _linear_to_srgb(rgb_lin)
    out = np.concatenate([rgb, alpha], axis=2)
    sprite = (out * 255).clip(0, 255).astype(np.uint8)
    sx, sy = (scale, scale) if isinstance(scale, (int, float)) else scale
    if sx != 1.0 or sy != 1.0:
        nw, nh = max(1, int(w * sx)), max(1, int(h * sy))
        sprite = np.array(Image.fromarray(sprite, "RGBA").resize((nw, nh), Image.LANCZOS))
    if cat == "Highlight":
        # Highlight gets a soft halo. NSO float-pool literal at 0x2C085D2
        # (= 11.25, also referenced from 3 other call-sites) is the
        # `Highlight.BlurWidth` field's default. The exact engine units
        # aren't pinned down (it's also `360°/32 = 11.25°` so the symbol
        # is overloaded), so we treat it as a sigma in *native* sprite
        # pixels and rescale by the sprite's apparent canvas size, then
        # apply through PIL's GaussianBlur (radius ≈ sigma).
        blur_native_px = _HIGHLIGHT_BLUR_WIDTH * _HIGHLIGHT_BLUR_FACTOR
        blur_canvas_px = blur_native_px * (sx + sy) * 0.5
        if blur_canvas_px > 0.5:
            sprite = np.array(
                Image.fromarray(sprite, "RGBA").filter(
                    ImageFilter.GaussianBlur(radius=blur_canvas_px)))
    return np.asarray(sprite, dtype=float)

def _miiparts_indices(cat):
    """Available indices for a MiiParts category, sorted."""
    return sorted(_miiparts_reg(cat))

def _miiparts_name(cat, idx):
    """Filename-without-extension for a MiiParts entry, e.g. 'Eye070' or 'Eyebrow03'."""
    entry = _miiparts_reg(cat).get(idx)
    return entry[0] if entry else None

# ── Face canvas geometry ───────────────────────────────────────────────────────
# Icon PNGs are 300×300; we composite at _PX_SCALE× so MiiParts textures
# (e.g. Eye=152px native) fit without LANCZOS downscale softening their lines.
ICON_RES     = 300                  # source-icon resolution (Face_*.png)
_PX_SCALE    = 2                    # canvas-pixel ÷ icon-pixel multiplier
FACE_RES     = ICON_RES * _PX_SCALE
FACE_CX      = 150 * _PX_SCALE
FACE_OVL_TOP = 18  * _PX_SCALE
FACE_OVL_BOT = 280 * _PX_SCALE
FACE_CY      = (FACE_OVL_TOP + FACE_OVL_BOT) // 2

# ── Engine positioning constants (Switch 2 NSO main.nso) ──────────────────────
# All values below were located in the decompressed NSO via the float-pool audit
# at tools/nso_re/FORMULAS.md §5. They match the FFL (Wii-era) formulas only
# coincidentally — the names use the `_NSO_` prefix and cite the binary VA so
# the source is unambiguous. The legacy FFL `_POS_X_ADD` (3.5323312) is NOT
# present in the binary and is omitted (it only mattered for a self-cancelling
# mole-X term).
_NSO_POS_Y_ADD    = 0.46293801   # NSO float pool (FFL stored 4.629278; Switch 2 = old/10)
_NSO_SPACING_MUL  = 0.88961464   # spacing param → engine units
_NSO_POS_X_MUL    = 1.7792293    # x-position param → engine units
_NSO_POS_Y_MUL    = 1.0760943    # y-position param → engine units

# Per-feature Y anchor floats located in the NSO float pool (one per feature
# category). Addresses verified in EMPIRICAL_AUDIT.md §B.
_NSO_EYE_Y_BASE   = 18.4515   # NSO @ 0x29eb2d0
_NSO_BROW_Y_BASE  = 16.5498   # NSO @ 0x29eaca4
_NSO_MOUTH_Y_BASE = 29.2589   # NSO @ 0x29eb31c
_NSO_MSTCH_Y_BASE = 31.7635   # NSO @ 0x29eab80
_NSO_MOLE_X_BASE  = 17.7662   # NSO @ 0x29eb214
_NSO_MOLE_Y_BASE  = 17.9599   # NSO @ 0x29eaf24

# Authoritative head-mesh face-paint UV transform (source: MiiSystem/System.mii__SystemParam.bgyml).
# FacePaintSize {0.75, 0.75}, FacePaintOffset {0, 0.03}: the Mask mesh samples
# the central 75%×75% of the painted texture with a +0.03 shift in GL UV (which
# is upward in the painted PIL image since GL v=1 ≡ PIL row 0). Applied as a
# UV transform on the Mask mesh in _apply_face_texture_to_scene.
_FACE_PAINT_SCALE    = 0.75
_FACE_PAINT_OFFSET_Y = 0.03

# Per-feature sprite shrink. The engine's FacePaintSize (0.75) applies
# uniformly to BOTH position AND size — the engine paint formula
# `pixel = (raw-16)/32 * tex_dim * 0.75 + 0.03 * tex_dim` puts a feature
# of native size N at canvas size N * 0.75 within the central 75% region.
# Face-spanning categories (Faceline / Make / Wrinkle) are NOT shrunk;
# they're authored to fill the whole paint region.
#
# `AccessoryScaleCoef = 0.7` from System.mii__SystemParam.bgyml is also
# loaded by the engine (string referenced from main.nso .rodata @
# 0x28A6E86), but its specific application site is NOT disassembled. The
# name suggests "stick-on accessories" (e.g. mole, paired with the
# `AccessoryBagScaleCoef = 0.85` for "wearable bag accessories" like
# hats/glasses). Earlier code applied 0.7 to ALL features as a per-feature
# shrink — that was overcompensating for a separate bug in the mesh UV
# transform (since fixed). Until the engine site for AccessoryScaleCoef is
# pinned down, we apply only FacePaintSize to face features.
_NSO_FEATURE_SCALE = _FACE_PAINT_SCALE   # 0.75 — engine's FacePaintSize for size

# Highlight blur — NSO @ 0x2C085D2 holds `Highlight.BlurWidth` = 11.25, but the
# unit is ambiguous (the same literal also encodes `360°/32`, a rotation step).
# Treating it as a Gaussian sigma in native sprite pixels and applying any
# meaningful factor (≥0.1) inflates the white sparkle until it dominates the
# iris, making the eye look much larger than the engine's. Disabled by default;
# re-enable once the BlurWidth semantic is pinned via engine comparison.
_HIGHLIGHT_BLUR_WIDTH  = 11.25
_HIGHLIGHT_BLUR_FACTOR = 0.0

# ── Canvas Y calibration (engine-derived from FORMULAS.md §6) ─────────────────
# Engine paint formula (verified against the float pool — 1/32 @ 0x29EB949,
# 0.75 = FacePaintSize, 0.03 = FacePaintOffset.Y):
#   pixel_y = engine_y * (1/32) * tex_h * 0.75 + 0.03 * tex_h
# For tex_h = FACE_RES = 600:
#   PY_SCALE = 600 * 0.75 / 32 = 14.0625
#   PY_OFFS  = 600 * 0.03      = 18.0
# Earlier comment claimed engine-faithful OFFS was 57 (=tex_h*((1-0.75)/2 -
# 0.03), an inset-model derivation). That derivation was wrong: the engine
# paints features into the FULL canvas with the formula above, then the Mask
# mesh UV transform applies the 0.75 inset. With the corrected OFFS=18 the
# analytic positions are eye→mesh.v 0.538, brow 0.582, mouth 0.284,
# mustache 0.226, mole 0.549 — matches the prior empirical fit (eye 0.530,
# brow 0.560, mole 0.538) within 1-2% for upper-face features. Mouth/
# mustache fall ~7% lower than the empirical fit; the empirical fit was
# overcompensating for the wrong-OFFS engine prediction.
_CANVAS_PY_SCALE = FACE_RES * _FACE_PAINT_SCALE / 32   # 14.0625 for FACE_RES=600
# Engine-derived: FACE_RES * FacePaintOffset.Y (from MiiSystem/System.mii__SystemParam.bgyml).
# NOTE: the eye compositor (FUN_71001737d0 @ 0x173968) outputs `eye.pos.y =
# eyePosY * 1.0760943 + 18.4515` in *engine coordinates*, not raw bgyml int.
# Chaining `engine_y * 14.0625 + 18` treats that engine output as a raw field
# and may be a category error — see /tmp/nso_re/LASH_REPORT.md §"Eye-position
# investigation" for the unresolved engine→canvas mapping.
_CANVAS_PY_OFFS  = FACE_RES * _FACE_PAINT_OFFSET_Y     # = 18.0 (engine-derived)

# ── Canvas X calibration ──────────────────────────────────────────────────────
# Engine Pos.x uses the same `(raw-16)/32 * tex_w * 0.75` formula as Pos.y
# (per FORMULAS.md §6), giving 14.0625 px per raw unit. _engine_spacing_x
# below uses _NSO_SPACING_MUL (0.88961, in the pool @ 0x29EB08C) which
# converts spacing param → engine units; the per-engine-unit pixel rate
# would then be PY_SCALE/SPACING_MUL ≈ 15.81. The hand-anchored
# `_PX_SCALE * 48 / (2 * SPACING_MUL)` formula was kept because the engine's
# spacing semantics — whether `spacing` even uses the Pos formula, or maps
# through a separate per-feature default — has not been confirmed by
# disassembly. Task #19 leaves this anchored to icon-pixel convention until
# the spacing site in main.nso is decoded.
_CANVAS_PX_SCALE = _PX_SCALE * 48.0 / (2 * _NSO_SPACING_MUL)

def _engine_y(base_engine, param=0):
    """Convert engine y-coordinate (NSO units) to canvas pixel Y."""
    return (base_engine + param * _NSO_POS_Y_MUL) * _CANVAS_PY_SCALE + _CANVAS_PY_OFFS

def _engine_spacing_x(spacing_param):
    """Convert engine spacing parameter to canvas pixel distance from centre."""
    return max(10, spacing_param * _NSO_SPACING_MUL * _CANVAS_PX_SCALE)

def _ffl_scale(scale_param, scale_y_param=0):
    """FFL-style feature base scale from Mii scale parameters (0=default).

    FFL-SOURCED FORMULA (tasks #27, #28): `0.4*raw + 1.0` is the FFL
    convention, NOT the engine's. The Switch 2 NSO int→float Scale
    conversion sits behind a vtable dispatch and has not been decoded.
    Renaming kept as `_ffl_*` deliberately as a flag — when the engine
    formula lands this becomes `_engine_scale`.
    """
    return 0.4 * scale_param + 1.0

# Derived canvas defaults (param=0, spacing=2)
_DEF_EYE_Y   = int(_engine_y(_NSO_EYE_Y_BASE))       # ~150
_DEF_EYE_X   = int(_engine_spacing_x(2))              # ~48
_DEF_BROW_Y  = int(_engine_y(_NSO_BROW_Y_BASE))       # ~132
_DEF_BROW_X  = int(_engine_spacing_x(2))              # ~48
_DEF_MOUTH_Y = int(_engine_y(_NSO_MOUTH_Y_BASE))      # ~252

# ── Gallery helpers ────────────────────────────────────────────────────────────
def _swatch_paths(prefix, linear_colors, size=30):
    """Generate small PNG colour swatches for an arbitrary palette and
    return paths in slot order. Cached per `prefix` so the skin/eye/
    mouth/glass pickers don't collide. Each tile is `size`×`size` with
    a 1-px contrast border."""
    cache_dir = os.path.join(tempfile.gettempdir(),
                             f"mii_face_renderer_{prefix}_swatches")
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for slot, lin in enumerate(linear_colors):
        srgb = tuple(
            int(round(max(0.0, min(1.0,
                (12.92*c) if c <= 0.0031308 else (1.055 * c**(1/2.4) - 0.055)
            )) * 255))
            for c in lin[:3]
        )
        p = os.path.join(cache_dir, f"{prefix}_{slot:02d}.png")
        if not os.path.exists(p):
            arr = np.full((size, size, 3), srgb, dtype=np.uint8)
            border = (96, 96, 96) if (sum(srgb) > 384) else (200, 200, 200)
            arr[0, :]  = border; arr[-1, :] = border
            arr[:, 0]  = border; arr[:, -1] = border
            Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _hair_swatch_paths():
    """Generate 100 small PNG colour swatches (one per slot in
    `_NSO_HAIR_COLORS`) and return a list of paths in slot order. Cached
    in tempdir; regenerated whenever the cache file is missing.

    Each swatch is 24×24 — kept tiny so the 10×10 gallery fits in a
    compact ~260 px column without scrollbars. 1-px border for visibility.
    """
    cache_dir = os.path.join(tempfile.gettempdir(), "mii_face_renderer_hair_swatches")
    os.makedirs(cache_dir, exist_ok=True)
    SIZE = 24
    paths = []
    for slot, lin in enumerate(_NSO_HAIR_COLORS):
        # Encode linear → sRGB for display (PIL pixels are sRGB-encoded).
        srgb = tuple(
            int(round(max(0.0, min(1.0,
                (12.92*c) if c <= 0.0031308 else (1.055 * c**(1/2.4) - 0.055)
            )) * 255))
            for c in lin
        )
        p = os.path.join(cache_dir, f"hair_{slot:02d}.png")
        if not os.path.exists(p):
            arr = np.full((SIZE, SIZE, 3), srgb, dtype=np.uint8)
            border = (96, 96, 96) if (sum(srgb) > 384) else (200, 200, 200)
            arr[0, :]  = border; arr[-1, :] = border
            arr[:, 0]  = border; arr[:, -1] = border
            Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _empty_tile_path():
    """Lazily create a single transparent placeholder PNG used as the first
    'no selection' tile in every face-feature gallery. Cached path."""
    if not hasattr(_empty_tile_path, "_path"):
        path = os.path.join(tempfile.gettempdir(), "mii_face_renderer_empty_tile.png")
        if not os.path.exists(path):
            img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            img.save(path)
        _empty_tile_path._path = path
    return _empty_tile_path._path


def _gallery_items(prefix):
    """Return [(image_path, name), ...] for one face-category, ordered by the
    in-game PartsOrder bgyml and filtered to IsVisibleInEditor=true (when the
    metadata is available; falls back to raw filename order otherwise).

    The first entry is always a transparent '(none)' tile so the user can
    deselect via the gallery itself (the in-game `*Nothing` parts have no
    extracted PNG, so without this the first tile would be a real choice
    that duplicates one already in the grid)."""
    mp_cat = _MIIPARTS_CAT_MAP.get(prefix)
    if not mp_cat:
        return []
    items = [(_none_icon_path(prefix), "(none)")]
    ordered = mm.visible_names_in_order(mp_cat)
    if ordered:
        for name in ordered:
            if name and name.endswith("Nothing"):
                continue  # represented by the leading transparent tile
            icon = _face_icon_path(prefix, name)
            if icon:
                items.append((icon, name))
    else:
        for idx in _miiparts_indices(mp_cat):
            name = _miiparts_name(mp_cat, idx)
            if name:
                items.append((_face_icon_path(prefix, name), name))
    return items

def _rotate_sprite(sprite, deg):
    """Rotate a float RGBA sprite by `deg` degrees, padding with transparent."""
    if sprite is None or deg == 0:
        return sprite
    img = Image.fromarray(sprite.astype(np.uint8), "RGBA")
    img = img.rotate(-deg, resample=Image.BICUBIC, expand=True)
    return np.array(img, dtype=float)


def _rotate_around_pivot(sprite, deg, pivot_xy):
    """Rotate a float RGBA sprite by `deg` degrees around `pivot_xy`
    (in native-image pixel coords). Returns `(rotated_sprite,
    pivot_xy_in_rotated)` — the new image is expanded so all of the
    rotated content fits, and the second tuple element is where the
    original pivot ends up inside that expanded canvas. Callers paste
    so the new_pivot lands at the desired anchor in canvas px."""
    h, w = sprite.shape[:2]
    if deg == 0:
        return sprite, pivot_xy
    img = Image.fromarray(sprite.astype(np.uint8), "RGBA")
    img = img.rotate(-deg, resample=Image.BICUBIC, expand=True,
                     center=pivot_xy)
    # Compute where the original `pivot_xy` ends up in the rotated
    # canvas by mirroring PIL's expand-bbox math.
    import math
    t = math.radians(-deg)
    c, s = math.cos(t), math.sin(t)
    px, py = pivot_xy
    # Original image corners rotated around pivot:
    corners = [(0 - px, 0 - py), (w - px, 0 - py),
               (w - px, h - py), (0 - px, h - py)]
    rotated = [(x * c - y * s + px, x * s + y * c + py) for x, y in corners]
    min_x = min(r[0] for r in rotated)
    min_y = min(r[1] for r in rotated)
    new_pivot = (px - min_x, py - min_y)
    return np.array(img, dtype=float), new_pivot


# ── Native MiiParts texture sizes (from BNTX header) ──────────────────────────
# Eye-overlay categories (lashes/lids/highlight) are authored at these smaller
# sizes but UV-mapped onto the same eye region as the Eye texture (152×128).
# So when compositing in PNG-pixel space, they must be scaled UP to the eye
# size — otherwise lash strokes appear as tiny marks instead of spanning the eye.
_MIIPARTS_NATIVE = {
    "Eye": (152, 128), "EyelashUpper": (96, 80), "EyelashLower": (96, 80),
    "EyelidUpper": (96, 80), "EyelidLower": (96, 80), "Highlight": (64, 64),
    "Eyebrow": (144, 128),
    "Mouth": (176, 128),
    # Make/Wrinkle PNGs are 256×256 half-face masks (shape lives in one
    # quadrant), authored to be painted onto the full face-paint region and
    # then mirrored to cover the opposite side — see `_PART_DEFAULT_PLACEMENT_CATS`
    # branch in compose_face_texture.
    "MakeUpper": (256, 256), "MakeLower": (256, 256),
    "WrinkleUpper": (256, 256), "WrinkleLower": (256, 256),
}
_EYE_OVERLAY_CATS = {"EyelashUpper", "EyelashLower",
                     "EyelidUpper",  "EyelidLower",
                     "Highlight"}
# Make/Wrinkle parts carry their own DefaultTransX/Y/Scale in the parts bgyml
# (no per-eye accessory row), so they're placed via _part_default_placement.
_PART_DEFAULT_PLACEMENT_CATS = {"MakeUpper", "MakeLower",
                                "WrinkleUpper", "WrinkleLower"}

# Per-eye overlay placement comes from MiiEyeAccessoryParam.byml.yaml — every
# Eye part references a row that gives, for each overlay (EyelashUpper/Lower,
# EyelidUpper/Lower, Highlight), Pos {X,Y}, Rotate, Scale, Aspect.
# Field semantics are decoded in _overlay_placement per FORMULAS.md (NSO RE).

_FACE_PAINT_REGION_PX = FACE_RES * _FACE_PAINT_SCALE  # 450 px on the 600 canvas

def _part_default_placement(part_name, native_w, native_h):
    """Resolve canvas-space placement for a Make/Wrinkle part using
    DefaultTransX/Y/Scale/Aspect from the part's bgyml.

    Make/Wrinkle assets are 256×256 half-face masks authored to be painted
    onto the full face-paint region (FACE_RES × _FACE_PAINT_SCALE) and then
    mirrored to cover the opposite side. The result is added once at the
    face centre — there is no per-eye pairing.

    Returns (cx, cy, sx, sy) in canvas pixels. Y still uses the 32-unit
    grid centred at (Y=16), scaled by the face-paint region height; X is
    left at 0 (TransX semantics differ from the accessory-Pos 32-grid and
    are bounded by per-part `MaxTransX`).
    """
    p = mm.parts().get(part_name, {})
    tx  = p.get("DefaultTransX", 0)
    ty  = p.get("DefaultTransY", 16)
    scl = p.get("DefaultScale",  8)
    asp = p.get("DefaultAspect", None)
    # Y offset on the face-paint canvas (paint region = 0.75 × face).
    # The bgyml MinTransY/MaxTransY ranges (e.g. MakeUpper00: 7..24,
    # default 12) are absolute positions in the engine's 32-unit grid,
    # so we use the same `(raw - 16) * step` mapping as Pos.Y.
    offy = (ty - 16) * (_FACE_PAINT_REGION_PX / 32.0)
    # X offset: the engine paints one half-face mask, then mirrors it.
    # TransX is the *outward* shift from face centre in the same 32-unit
    # grid (TX=0 → no shift, asset mirrored at the centre line). MaxTransX
    # in the bgyml caps the user-facing slider; default values cluster
    # around 1-7 across MakeUpper/MakeLower/Wrinkle parts.
    offx = tx * (_FACE_PAINT_REGION_PX / 32.0)
    # Scale: Switch 2 lash/lid/highlight compositors use the linear form
    # `scale_factor = scl * 0.0345479 + 0.1554656` (binary-derived, NSO
    # constants @ 0x2601e10 / 0x25fa23c — see _NSO_LASH_SCALE_SLOPE).
    # Make/Wrinkle compositor isn't statically decoded yet (one of the
    # 0x175630 / 0x176010 / 0x2ff640 / 0x2ffb20 wrappers around the eye
    # compositor), but the surrounding scale-formula constants in main.nso
    # all follow this `slope * raw + offset` shape. Treat the resulting
    # factor as the canvas-pixel-per-native ratio (no extra paint-region
    # scale-up — assets are already authored at the engine's intended
    # rendered size).
    scale_factor = scl * _NSO_LASH_SCALE_SLOPE + _NSO_LASH_SCALE_OFFSET
    sx = scale_factor
    # DefaultAspect (when present) follows the same `aspect*0.12 + 0.64`
    # form as the mustache/mole compositors (NSO constants @ 0x29eae74 /
    # 0x29ead0c, see FORMULAS.md). Treat the result as the multiplier on
    # sy relative to sx (so aspect=8 → sy=1.6×sx, aspect=0 → 0.64×sx).
    sy = sx * (asp * 0.12 + 0.64) if isinstance(asp, (int, float)) else sx
    return offx, offy, sx, sy


# Inverses of the FFL-pipeline fixed-aspect scale constants. Verified in
# main.nso .rodata: -1.1240822 @ 0x29eb090 (= -1/SPACING_MUL),
# 1.0779724 @ 0x29eaf80 (= 1/Y_FIXED_SCALE). Used by the lash compositor
# to undo the matrix-builder's fixed-aspect scaling.
_NSO_INV_SPACING_MUL_NEG = -1.1240822
_NSO_INV_Y_FIXED_SCALE   =  1.0779724

# Lash/lid/highlight scale formula (Switch 2 NEW — does NOT match FFL's
# `0.4 * raw + 1.0`). Decoded from FUN_71001733a0 in main.nso. Constants
# verified in float pool: 0.0345479 @ 0x2601e10 (slope), 0.1554656 @
# 0x25fa23c (offset). Aspect Y formula `0.12 * raw + 0.64` is the FFL
# eyebrow/eye Y-aspect form (constants present at 0x2610a2c, 0x29ead0c).
_NSO_LASH_SCALE_SLOPE  = 0.0345479
_NSO_LASH_SCALE_OFFSET = 0.1554656
_NSO_ASPECT_Y_SLOPE    = 0.12
_NSO_ASPECT_Y_OFFSET   = 0.64

def _eye_iris_bounds_native(idx):
    """Iris content row range within the native eye texture (152×128).
    Computed from BNTX texture data (binary-derived, NOT empirical).

    The engine's eye.scale.y matrix-input rectangle corresponds to the
    visible iris area, not the texture frame. Our extracted PNG textures
    have transparent margins above/below the iris; using the alpha-weighted
    G+B-channel bbox gives the engine-equivalent eye rectangle.

    Returns (top_row, bot_row, cx_native) — last is the iris X centre,
    needed because eye PNGs are authored asymmetrically (iris pushed toward
    one frame edge; CLAUDE.md principle 2).

    Cached per Eye part index.
    """
    if not hasattr(_eye_iris_bounds_native, "_cache"):
        _eye_iris_bounds_native._cache = {}
    cache = _eye_iris_bounds_native._cache
    if idx in cache:
        return cache[idx]
    eye_w, eye_h = _MIIPARTS_NATIVE["Eye"]
    path = _miiparts_path("Eye", idx)
    if path is None or not os.path.exists(path):
        cache[idx] = (0.0, float(eye_h), eye_w / 2.0)
        return cache[idx]
    arr = np.array(Image.open(path).convert("RGBA"))
    mask = (arr[:, :, 1].astype(int) + arr[:, :, 2].astype(int)) > 16
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        cache[idx] = (0.0, float(eye_h), eye_w / 2.0)
    else:
        # Alpha-weighted X centroid of the iris content (G+B channels).
        weights = (arr[:, :, 1].astype(float) + arr[:, :, 2].astype(float))
        cols = np.arange(arr.shape[1], dtype=float)
        cx = float((weights * cols[None, :]).sum() / weights.sum())
        cache[idx] = (float(rows[0]), float(rows[-1] + 1), cx)
    return cache[idx]


def _overlay_placement(cat, acc, eye_pos_x, eye_pos_y, rendered_eye_w, rendered_eye_h, eye_idx, mirror_x=False):
    """Switch 2 lash/lid/highlight compositor (decoded from main.nso
    `FUN_71001733a0` @ 0x1733a0 — type 0xd overlay compositor).

    Each overlay's position is RELATIVE to the eye sprite's position and
    scale. The engine first runs the eye compositor (`FUN_71001737d0`),
    then this function adjusts the descriptor for the overlay, then the
    matrix builder (`FUN_7100171910` = CalcMVMatrix equivalent) consumes
    the descriptor.

    The matrix builder applies a final fixed scale of (0.88961464,
    0.9276675) to width/height. That cancels nicely with the
    `1/-1.1240822` and `1/1.0779724` divisors in the engine source: the
    rendered overlay's CANVAS position offset is
        x_offset = -(Pos.X/32 - 0.5) * rendered_eye_w
        y_offset = +(Pos.Y/32 - 0.5) * rendered_eye_h
    and rendered overlay size is
        width  = (Scale * 0.0345479 + 0.1554656) * rendered_eye_w
        height = width * (Aspect * 0.12 + 0.64) * (0.9276675 / 0.88961464)

    Returns (pos_x_canvas, pos_y_canvas, rot_deg, sprite_w_canvas,
    sprite_h_canvas) — caller should paste sprite scaled to (w, h) px
    centered at (pos_x, pos_y).
    """
    pos = acc.get(cat + "Pos", {"X": 16, "Y": 16})
    scl = acc.get(cat + "Scale", 8)
    asp = acc.get(cat + "Aspect", 3)
    rot = float(int(acc.get(cat + "Rotate", 0)) & 0x1f) * 11.25
    # Position. Per user empirical feedback the lash Y position depends on
    # the eye EDGE (not center): upper lash should sit just above the eye top,
    # lower lash just below the eye bottom. Pos.Y < 16 anchors to eye_top;
    # Pos.Y > 16 anchors to eye_bot. The (Pos.Y - 16) delta is interpreted as
    # a small pixel offset from that edge.
    # IRIS bounds (per CLAUDE.md principle 2 — derived from BNTX texture
    # content, not the sprite frame). Convert native rows to canvas Y by
    # mapping native rows [0, eye_h_native] to canvas [eye_top_sprite,
    # eye_bot_sprite].
    eye_w_native, eye_h_native = _MIIPARTS_NATIVE["Eye"]
    iris_top_native, iris_bot_native, iris_cx_native = _eye_iris_bounds_native(eye_idx)
    sx_per_native = rendered_eye_w / eye_w_native
    # Iris-X anchor only applies to features whose own texture is centred
    # in its frame (e.g. Highlight). Lash/lid textures are authored
    # asymmetrically — thick base toward inner corner, thin tip extending
    # outward past the eye edge — so they're already anchored to land
    # correctly when pasted at the eye-FRAME centre. Anchoring those at
    # the iris centre would pull them further inward.
    if cat == "Highlight":
        iris_dx_native = iris_cx_native - eye_w_native / 2.0
        if mirror_x:
            iris_dx_native = -iris_dx_native
        anchor_x = eye_pos_x + iris_dx_native * sx_per_native
    else:
        anchor_x = eye_pos_x
    iris_cx_canvas = anchor_x  # for caller's mirror around the right anchor
    # NSO-confirmed lash compositor X formula (FUN_71001733a0 @ 0x1733a0):
    #     lash.pos.x = eye.pos.x + (Pos.X/32 - 0.5) * eye.scale.x / -1.1240822
    # In canvas units: eye.scale.x_canvas = rendered_eye_w / 2 (since
    # 5.34375 engine units = 152/2 native = rendered_eye_w/2 canvas;
    # the engine_unit→canvas factor is dimensionless once we use
    # consistent half-extents). Per-eye scale is captured in
    # rendered_eye_w already. Sign: -1.1240822 inverts X direction so
    # Pos.X < 16 → positive offset = INWARD on the R-eye, OUTWARD on the
    # L-eye. We add the offset (not subtract) and let the caller mirror.
    pos_x = anchor_x + (pos["X"] / 32.0 - 0.5) * rendered_eye_w / 2.0 / -1.1240822
    sprite_top = eye_pos_y - rendered_eye_h / 2.0
    sy_per_native = rendered_eye_h / eye_h_native
    iris_top_canvas = sprite_top + iris_top_native * sy_per_native
    iris_bot_canvas = sprite_top + iris_bot_native * sy_per_native
    # Engine formula: pos.y = eye.pos.y + (Pos.Y/32 - 0.5) * eye.scale.y / 1.078
    # In our renderer, eye.scale.y_canvas should equal the IRIS height
    # (not sprite height) because the engine's eye.scale.y matrix-input
    # rectangle is the visible iris. Use iris height as the multiplier and
    # iris center as the anchor.
    iris_h_canvas = iris_bot_canvas - iris_top_canvas
    iris_cy_canvas = (iris_top_canvas + iris_bot_canvas) / 2.0
    # NSO-confirmed lash compositor (FUN_71001733a0 @ 0x1733a0):
    #     lash.pos.y = eye.pos.y + (Pos.Y/32 - 0.5) * eye.scale.y / 1.0779724
    # Engine eye.scale.y_engine for neutral eye = 4.5 (half-extent of
    # unit quad in engine units). In CANVAS pixels, the engine→canvas
    # factor must be derived consistently with sprite sizing:
    #     1 engine unit = (rendered_eye_h / 9) canvas px
    # because the eye full height is 9 engine units (= 4.5 × 2) and
    # rendered_eye_h is the full canvas height. So
    #     eye.scale.y_canvas = 4.5 × (rendered_eye_h / 9) = rendered_eye_h / 2
    # (per-eye eye_scale is captured in rendered_eye_h already, the
    # 4.5 / 9 ratio is dimensionless and constant).
    # Earlier the code chained `_CANVAS_PY_SCALE` with `_NSO_FEATURE_SCALE`
    # and produced a multiplier ~32% too large vs the engine. Anchor:
    # eye.pos.y_canvas (the engine reads eye.pos.y for all overlay
    # variants — there's no per-cat iris-edge anchoring in the binary).
    _LASH_Y_DIVISOR = 1.0779724  # Y aspect undo, RO @ 0x29eaf80
    pos_y = eye_pos_y + (pos["Y"] / 32.0 - 0.5) * rendered_eye_h / 2.0 / _LASH_Y_DIVISOR
    # Scale (NEW Switch 2 formula, NOT FFL):
    scale_factor = scl * _NSO_LASH_SCALE_SLOPE + _NSO_LASH_SCALE_OFFSET
    aspect_y     = asp * _NSO_ASPECT_Y_SLOPE  + _NSO_ASPECT_Y_OFFSET
    sprite_w = scale_factor * rendered_eye_w
    # Y rendered = scale.x_matrix * aspect_y * 0.9276675; equivalently w * aspect_y * (0.9276675/0.88961464):
    sprite_h = sprite_w * aspect_y * (0.9276675 / 0.88961464)
    return pos_x, pos_y, rot, sprite_w, sprite_h, iris_cx_canvas


# ── Expression resolver ───────────────────────────────────────────────────────
# An expression bgyml may override eye/eyebrow/mouth parts and may reference
# a *Location* file that adds Rotate / PositionY / Scale deltas. The deltas
# below are converted into compositor-friendly units:
#   Rotate    : degrees (treated 1:1 — small game-side ints)
#   PositionY : icon-pixel delta (×_PX_SCALE for canvas pixels)
#   Scale     : multiplier on existing scale (treated as +N → ×(1+0.2N))

def _resolve_expression(expr_name):
    """Return dict of overrides for the active expression. Empty if Normal/None.

    Keys produced (any subset):
      eye_l, eye_r, eyebrow_l, eyebrow_r, mouth     — part names (str)
      eye_rot, eyebrow_rot, mouth_rot               — degrees
      eye_dy, eyebrow_dy, mouth_dy                  — icon-pixel Y offset
      eye_scale_mul, eyebrow_scale_mul, mouth_scale_mul  — float multiplier
    """
    if not expr_name or expr_name == "Normal":
        return {}
    expr = mm.expressions().get(expr_name)
    if not expr:
        return {}
    out = {}
    el, er = expr.get("EyePartsLName"), expr.get("EyePartsRName")
    if el: out["eye_l"] = el
    if er: out["eye_r"] = er
    bl, br = expr.get("EyebrowPartsLName"), expr.get("EyebrowPartsRName")
    if bl: out["eyebrow_l"] = bl
    if br: out["eyebrow_r"] = br
    m = expr.get("MouthPartsName") or expr.get("LipSyncMouthPartsName")
    if m: out["mouth"] = m
    locs = mm.parts_locations()
    for src_key, prefix in (("EyeLocationName",     "eye_"),
                            ("EyebrowLocationName", "eyebrow_"),
                            ("MouthLocationName",   "mouth_")):
        loc_name = expr.get(src_key)
        if not loc_name:
            continue
        loc = locs.get(loc_name, {})
        if "Rotate" in loc:    out[prefix + "rot"]       = float(loc["Rotate"])
        if "PositionY" in loc: out[prefix + "dy"]        = float(loc["PositionY"])
        if "Scale" in loc:     out[prefix + "scale_mul"] = 1.0 + 0.2 * float(loc["Scale"])
    return out


def _paste_sprite(canvas, sprite, cx, cy, scale=1.0):
    """Alpha-composite sprite centred at (cx, cy) onto canvas. Sprite paints
    at native PNG size when scale=1.0; FacePaintSize is applied via the
    mesh UV transform in `_apply_face_texture_to_scene` (NOT by pre-shrinking
    the sprite, which would double-compensate)."""
    if sprite is None or sprite.size == 0:
        return
    if scale != 1.0:
        h, w = sprite.shape[:2]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        img = Image.fromarray(sprite.astype(np.uint8))
        sprite = np.array(img.resize((nw, nh), Image.LANCZOS), dtype=float)
    h, w = sprite.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    cx0 = max(0, x0);  cy0 = max(0, y0)
    cx1 = min(canvas.shape[1], x0 + w);  cy1 = min(canvas.shape[0], y0 + h)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    sa = sprite[sy0:sy1, sx0:sx1, 3:4] / 255.0
    canvas[cy0:cy1, cx0:cx1, :3] = (
        sa * sprite[sy0:sy1, sx0:sx1, :3] +
        (1 - sa) * canvas[cy0:cy1, cx0:cx1, :3]
    )
    canvas[cy0:cy1, cx0:cx1, 3:4] = np.maximum(
        canvas[cy0:cy1, cx0:cx1, 3:4],
        sprite[sy0:sy1, sx0:sx1, 3:4],
    )


# ── Face texture compositor ────────────────────────────────────────────────────
FACE_PARTS = [
    ("Eye",          "Eyes"),
    ("Eyebrow",      "Eyebrows"),
    ("EyelashUpper", "Eyelash Upper"),
    ("EyelashLower", "Eyelash Lower"),
    ("EyelidUpper",  "Eyelid Upper"),
    ("EyelidLower",  "Eyelid Lower"),
    ("Mouth",        "Mouth"),
    ("MakeUpper",    "Makeup Upper"),
    ("MakeLower",    "Makeup Lower"),
    ("Highlight",    "Highlight"),
    ("Mole",         "Mole"),
    ("Mustache",     "Mustache"),
    ("Beard",        "Beard 2D"),
    ("WrinkleUpper", "Wrinkle Upper"),
    ("WrinkleLower", "Wrinkle Lower"),
]

# Eye-area iteration order, back-to-front matching the engine compositing order:
# face make/wrinkle paint first (under everything), then the colored Eye,
# then Eyelid (lid colour), Eyelash (dark line on top of eye), Highlight last
# (the white sparkle on the iris). Painting Eyelash *after* Eye is what makes
# the lash visible at all — otherwise the opaque Eye sprite covers it.
_EYE_STACK = [
    ("MakeUpper",      0, False),
    ("MakeLower",      0, False),
    ("WrinkleUpper",   0, False),
    ("WrinkleLower",   0, False),
    ("Eye",            0, True),   # True → apply eye scale
    ("EyelidLower",    0, False),
    ("EyelidUpper",    0, False),
    ("EyelashLower",   0, False),
    ("EyelashUpper",   0, False),
    ("Highlight",      0, False),
]


def compose_face_texture(face_sels,
                          eye_pos_y=0, eye_spacing_x=2, eye_scale=0,
                          brow_pos_y=0, brow_spacing_x=2, brow_scale=0,
                          mouth_pos_y=0, mouth_scale=0,
                          eye_color=4, mouth_color=0, hair_color=0,
                          expression="Normal",
                          mouth_drop_lips=False,
                          eye_use_red=False,
                          eyeline_color=(1.0, 1.0, 1.0),
                          eyebrow_color=None, beard2d_color=None, mustache_color=None):
    """
    Build a 300×300 RGBA face texture using the engine positioning formulas
    (Switch 2 NSO float-pool constants, see tools/nso_re/FORMULAS.md).
    eye_pos_y / brow_pos_y / mouth_pos_y are Mii parameter values
    (range ≈ 0–18); eye_spacing_x / brow_spacing_x ≈ 0–12.

    `expression` (e.g. 'Smile', 'Anger') overrides eye/brow/mouth selections
    and applies any PartsLocation rotate/offset/scale deltas.
    """
    sel = {prefix: name for (prefix, _), name in zip(FACE_PARTS, face_sels)}
    # Two canvases: one for FLAT face-decal features (eyes/brows/mouth/
    # mustache/mole — applied to the Mask mesh which is a wide flat overlay),
    # and one for SKIN-WRAPPING features (BeardShort, Wrinkle — applied to
    # the Head mesh which has the actual 3D-curved chin/cheek geometry).
    # Putting beard+wrinkle on the Mask mesh makes them appear flat-projected
    # (not following the chin's 3D curve); putting them on the Head mesh
    # gives proper 3D wrap.
    canvas       = np.zeros((FACE_RES, FACE_RES, 4), dtype=float)  # → Mask
    head_canvas  = np.zeros((FACE_RES, FACE_RES, 4), dtype=float)  # → Head
    any_drawn = False
    any_head_drawn = False

    expr_ovr = _resolve_expression(expression)

    # NSO-derived canvas positions, plus expression offsets in icon-pixel space
    eye_dy_px  = expr_ovr.get("eye_dy",     0.0) * _PX_SCALE
    brow_dy_px = expr_ovr.get("eyebrow_dy", 0.0) * _PX_SCALE
    mth_dy_px  = expr_ovr.get("mouth_dy",   0.0) * _PX_SCALE
    eye_rot    = expr_ovr.get("eye_rot",     0.0)
    brow_rot   = expr_ovr.get("eyebrow_rot", 0.0)
    mth_rot    = expr_ovr.get("mouth_rot",   0.0)
    eye_smul   = expr_ovr.get("eye_scale_mul",     1.0)
    brow_smul  = expr_ovr.get("eyebrow_scale_mul", 1.0)
    mth_smul   = expr_ovr.get("mouth_scale_mul",   1.0)

    eye_cy = _engine_y(_NSO_EYE_Y_BASE, eye_pos_y) + eye_dy_px
    eye_cx = _engine_spacing_x(eye_spacing_x)
    esc    = _ffl_scale(eye_scale) * eye_smul * _NSO_FEATURE_SCALE

    # The Eye selection drives per-overlay placement via MiiEyeAccessoryParam.
    eye_name_for_acc = expr_ovr.get("eye_l") or sel.get("Eye", "(none)")
    acc = mm.eye_accessory(eye_name_for_acc) if eye_name_for_acc and eye_name_for_acc != "(none)" else {}

    # ── Eye-area features (paired left / right) ────────────────────────────
    for cat, dy, use_eye_sc in _EYE_STACK:
        name = sel.get(cat, "(none)")
        # Expression eye-name override applies only to the Eye layer (not lashes/etc)
        if cat == "Eye":
            override_l = expr_ovr.get("eye_l")
            override_r = expr_ovr.get("eye_r")
        else:
            override_l = override_r = None
        eff_l = override_l or name
        eff_r = override_r or name
        if not eff_l or eff_l == "(none)":
            continue
        mp_cat = _MIIPARTS_CAT_MAP.get(cat)
        if not mp_cat:
            continue

        idx_l = _parse_feat_index(eff_l)
        idx_r = _parse_feat_index(eff_r)
        if cat == "Eye":
            # Eye renders unscaled (apart from engine eye_scale param) at eye_cy.
            spr_l = _load_miiparts_sprite(mp_cat, idx_l,
                                          eye_color=eye_color, hair_color=hair_color,
                                          eye_use_red=eye_use_red,
                                          eyeline_color=eyeline_color)
            spr_r = _load_miiparts_sprite(mp_cat, idx_r,
                                          eye_color=eye_color, hair_color=hair_color,
                                          eye_use_red=eye_use_red,
                                          eyeline_color=eyeline_color)
            if spr_r is None or spr_l is None:
                continue
            # Per-eye baseline rotation from the bgyml (`OffsetRotate`,
            # raw int converted to degrees via `raw * 11.25°` per
            # EMPIRICAL_AUDIT.md §D). Adds onto the expression-driven
            # eye_rot; right-eye negation matches the existing pattern.
            ofs_r = _bgyml_offset_rot(eff_r) * 11.25
            ofs_l = _bgyml_offset_rot(eff_l) * 11.25
            r_eye = _rotate_sprite(spr_r,  eye_rot + ofs_r)
            l_eye = _rotate_sprite(spr_l[:, ::-1, :].copy(), -(eye_rot + ofs_l))
            # Per-eye `SizeForExpression` (bgyml scalar, range ~0.6-1.0).
            # Applied as a multiplicative scale on top of the engine's
            # eye_scale param + expression eye_smul. Most eyes have it
            # unset (treated as 1.0).
            size_r = _eye_size_for_expr(eff_r)
            size_l = _eye_size_for_expr(eff_l)
            scl_r  = esc * size_r
            scl_l  = esc * size_l
            # Eye paste centres the sprite TEXTURE FRAME on
            # `(FACE_CX ± eye_cx, eye_cy)` — same anchor the lash
            # compositor reads as `eye.pos.{x,y}`. Don't compensate
            # for iris asymmetry here; the engine doesn't.
            _paste_sprite(canvas, r_eye, FACE_CX - eye_cx, eye_cy, scl_r)
            _paste_sprite(canvas, l_eye, FACE_CX + eye_cx, eye_cy, scl_l)
            any_drawn = True
            continue

        # Eye-overlay (lash/lid/highlight): engine-decoded compositor.
        # Lash position is RELATIVE to the eye sprite the engine just drew.
        if mp_cat in _EYE_OVERLAY_CATS and acc:
            eye_w_native, eye_h_native = _MIIPARTS_NATIVE["Eye"]
            rendered_eye_w = eye_w_native * esc
            rendered_eye_h = eye_h_native * esc
            # Right side of face (R-eye)
            # Use the EYE part index for iris-bounds derivation
            eye_part_idx = _parse_feat_index(eye_name_for_acc)
            # R-side eye sprite (canvas-left) is drawn ORIGINAL (line ~747);
            # L-side eye sprite (canvas-right) is H-flipped (line ~748).
            # The iris-X offset within the PNG flips with that mirror.
            pos_xR, pos_yR, ovl_rot, lash_w, lash_h, _iris_cxR = _overlay_placement(
                mp_cat, acc, FACE_CX - eye_cx, eye_cy, rendered_eye_w, rendered_eye_h, eye_part_idx, mirror_x=False)
            pos_xL, pos_yL, _, _, _, iris_cxL = _overlay_placement(
                mp_cat, acc, FACE_CX + eye_cx, eye_cy, rendered_eye_w, rendered_eye_h, eye_part_idx, mirror_x=True)
            # Convert rendered px to sprite-scale factor relative to native.
            ov_w_native, ov_h_native = _MIIPARTS_NATIVE[mp_cat]
            # Per-overlay scale multiplier — empirical. Eyelids ship at a
            # tiny native size relative to the eye sprite, so scale them
            # up 1.5× to match the in-game proportions.
            _OVL_SCALE = {"EyelidUpper": 1.5, "EyelidLower": 1.5}
            ovl_scale = _OVL_SCALE.get(mp_cat, 1.0)
            sx = lash_w / ov_w_native * ovl_scale
            sy = lash_h / ov_h_native * ovl_scale
            spr_r = _load_miiparts_sprite(mp_cat, idx_r,
                                          eye_color=eye_color, hair_color=hair_color,
                                          scale=(sx, sy))
            spr_l = _load_miiparts_sprite(mp_cat, idx_l,
                                          eye_color=eye_color, hair_color=hair_color,
                                          scale=(sx, sy))
            if spr_r is None or spr_l is None:
                continue
            # H-flip applied to the L-eye (the source texture is authored for
            # the right eye, so the left side is the mirrored copy). Verified
            # visually: lash thick base sits at the INNER corner with the thin
            # tip extending OUTWARD past the eye edge — matches typical Mii.
            # Lash/lid/highlight overlays sit on top of the eye sprite,
            # so they need to track the eye's baseline tilt
            # (`OffsetRotate` from the eye's bgyml). Without this a
            # tilted eye renders straight lashes intersecting the
            # rotated iris.
            eye_ofs_r = _bgyml_offset_rot(eff_r) * 11.25
            eye_ofs_l = _bgyml_offset_rot(eff_l) * 11.25
            r_spr = _rotate_sprite(spr_r,                     ovl_rot + eye_ofs_r)
            l_spr = _rotate_sprite(spr_l[:, ::-1, :].copy(), -(ovl_rot + eye_ofs_l))
            # Mirror the X-offset for the L-eye (since we flip the sprite,
            # the offset needs to flip too to land on the same side of the
            # mirrored eye). Mirror around the iris-content centre, NOT the
            # frame centre, since the iris sits asymmetrically in the frame.
            pos_xL_mirrored = 2 * iris_cxL - pos_xL
            # Compensate for the lash/lid PNG's frame-vs-ink offset.
            # The engine quad is centred at (lash.pos.x, lash.pos.y)
            # per the matrix math, but our extracted PNGs have the ink
            # in the bottom half of the frame (e.g. EyelashUpper ink
            # centroid at native (26, 65) on a 96×80 frame). Without
            # compensation the visible lash lands ~16 canvas px BELOW
            # the engine's intended position — putting it inside the
            # iris instead of at the eye edge. Pre-shift the paste so
            # the ink centroid lands at the formula position. The
            # L-side sprite is X-flipped, so its ink_dx flips sign.
            ink_dx_r, ink_dy_r = _overlay_ink_offset_native(mp_cat, idx_r)
            ink_dx_l, ink_dy_l = _overlay_ink_offset_native(mp_cat, idx_l)
            ink_dy_canvas_r = ink_dy_r * sy
            ink_dx_canvas_r = ink_dx_r * sx
            ink_dy_canvas_l = ink_dy_l * sy
            ink_dx_canvas_l = ink_dx_l * sx
            _paste_sprite(canvas, r_spr,
                          pos_xR - ink_dx_canvas_r, pos_yR - ink_dy_canvas_r)
            _paste_sprite(canvas, l_spr,
                          pos_xL_mirrored + ink_dx_canvas_l, pos_yL - ink_dy_canvas_l)
            any_drawn = True
        elif mp_cat in _PART_DEFAULT_PLACEMENT_CATS:
            # Whole-face mask: load once, paint at face centre, then mirror
            # for the opposite side. Wrinkle goes on head_canvas (skin
            # 3D-wrap surface); Make stays on the flat Mask canvas.
            nat_w, nat_h = _MIIPARTS_NATIVE.get(mp_cat, (256, 256))
            offx, offy, sx, sy = _part_default_placement(eff_l, nat_w, nat_h)
            spr = _load_miiparts_sprite(mp_cat, idx_l,
                                        eye_color=eye_color, hair_color=hair_color,
                                        scale=(sx, sy))
            if spr is None:
                continue
            spr_mirror = spr[:, ::-1, :].copy()
            cy = FACE_CY + offy
            # Wrinkle wraps the curved skin → head_canvas (Head mesh).
            # Make stays on flat-decal Mask canvas.
            target_canvas = head_canvas if mp_cat.startswith("Wrinkle") else canvas
            _paste_sprite(target_canvas, spr,        FACE_CX - offx, cy)
            _paste_sprite(target_canvas, spr_mirror, FACE_CX + offx, cy)
            if mp_cat.startswith("Wrinkle"):
                any_head_drawn = True
            else:
                any_drawn = True
        else:
            spr_l = _load_miiparts_sprite(mp_cat, idx_l,
                                          eye_color=eye_color, hair_color=hair_color)
            spr_r = _load_miiparts_sprite(mp_cat, idx_r,
                                          eye_color=eye_color, hair_color=hair_color)
            if spr_r is None or spr_l is None:
                continue
            l_src = spr_l[:, ::-1, :].copy()
            _paste_sprite(canvas, spr_r, FACE_CX - eye_cx, eye_cy + dy)
            _paste_sprite(canvas, l_src, FACE_CX + eye_cx, eye_cy + dy)
            any_drawn = True

    # ── Eyebrows (paired) ─────────────────────────────────────────────────
    brow_name = expr_ovr.get("eyebrow_l") or sel.get("Eyebrow", "(none)")
    brow_name_r = expr_ovr.get("eyebrow_r") or brow_name
    if brow_name and brow_name != "(none)":
        bcy = _engine_y(_NSO_BROW_Y_BASE, brow_pos_y) + brow_dy_px
        bcx = _engine_spacing_x(brow_spacing_x)
        idx_l = _parse_feat_index(brow_name)
        idx_r = _parse_feat_index(brow_name_r)
        spr_l = _load_miiparts_sprite("Eyebrow", idx_l, eyebrow_color=eyebrow_color)
        spr_r = _load_miiparts_sprite("Eyebrow", idx_r, eyebrow_color=eyebrow_color)
        if spr_r is not None and spr_l is not None:
            ofs_r = _bgyml_offset_rot(brow_name_r) * 11.25
            ofs_l = _bgyml_offset_rot(brow_name)   * 11.25
            # Brow rotation: the brow's own OffsetRotate plus the
            # expression-driven brow_rot. Rotation pivot is the
            # sprite frame centre (which is where _paste_sprite
            # anchors) — equivalent to "translate to (FACE_CX ± bcx,
            # bcy) THEN rotate in place" since rotating about the
            # paste centre commutes with the translation. Per the
            # user, this is the correct order for eye-rotated cases:
            # translation to brow position is NOT followed by an
            # eye-pivot rotation that moves the brow.
            r_brow = _rotate_sprite(spr_r, brow_rot + ofs_r)
            l_brow = _rotate_sprite(spr_l[:, ::-1, :].copy(),
                                    -(brow_rot + ofs_l))
            user_brow_scl = 1.0 + 0.15 * float(brow_scale)
            _paste_sprite(canvas, r_brow, FACE_CX - bcx, bcy,
                          brow_smul * _NSO_FEATURE_SCALE * user_brow_scl)
            _paste_sprite(canvas, l_brow, FACE_CX + bcx, bcy,
                          brow_smul * _NSO_FEATURE_SCALE * user_brow_scl)
            any_drawn = True

    # ── Centred features ──────────────────────────────────────────────────
    mouth_cy = _engine_y(_NSO_MOUTH_Y_BASE, mouth_pos_y) + mth_dy_px
    mstch_cy = _engine_y(_NSO_MSTCH_Y_BASE)
    # FacePaintSize per-feature shrink — same as Eye/Eyebrow above. Faceline
    # is face-spanning and stays at scale 1.0.
    A = _NSO_FEATURE_SCALE
    # Mustache transform — bit-exact formula extracted from
    # main.nso 0x170a80..0x170c14 (the mustache rendering function,
    # located via the EXACT load of MSTCH_Y_BASE @ 0x29eab80):
    #   mustache_scale  = scale_param * 0.4 + 1.0     (@0x29eb444, immed 1.0)
    #   mustache_aspect = aspect_param * 0.12 + 0.64  (@0x29eae74, @0x29ead0c)
    #   width_engine    = mustache_scale * 4.5        (immed 4.5)
    #   height_engine   = mustache_scale * 9.0 * mustache_aspect  (immed 9.0)
    #   y_engine        = y_param * 1.076094 + 31.763554
    # The earlier guess that 0.628/0.979 right after MSTCH_Y_BASE were
    # the scale/aspect offsets was wrong — those floats are used by
    # other unrelated functions in the same float-pool page. Mole and
    # mustache share the same SCALE_SLOPE (0.4), SCALE_OFFSET (1.0),
    # ASPECT_SLOPE (0.12), ASPECT_OFFSET (0.64) — only the dimensional
    # scale factors differ (mustache 4.5×9.0, mole has its own).
    _MUSTACHE_W_BASE        = 4.5    # engine units, NSO immed
    _MUSTACHE_H_BASE        = 9.0    # engine units, NSO immed
    _MUSTACHE_SCALE_SLOPE   = 0.4    # NSO 0x29eb444
    _MUSTACHE_SCALE_OFFSET  = 1.0    # NSO immed
    _MUSTACHE_ASPECT_SLOPE  = 0.12   # NSO 0x29eae74
    _MUSTACHE_ASPECT_OFFSET = 0.64   # NSO 0x29ead0c
    _MUSTACHE_OVERLAP       = 1
    # Beard (the 2D one is BeardShort) — case 13 of dispatcher
    # FUN_710016facc, calls the eye-relative compositor FUN_710176010.
    # Bit-exact math is documented in tools/nso_re/COMPOSITOR_NOTES.md
    # "2D Beard" section. The binary-default quad is TINY (~48 × 26
    # canvas px, max ~132 × 71); the runtime bgyml params that produce
    # the in-game placement haven't been located yet.
    # Painted on the head_canvas (Head mesh). After UV shift +0.5, the
    # Head's front-chin samples canvas y[403..468] (chin-tip 437..468,
    # lower jaw 339..392). The texture's ink centroid (sprite y=226 at
    # scale 0.55) needs to land in canvas y~435 (mid front-chin):
    #   paste_cy = 435 - (sprite_h/2 - ink_centroid_y_in_sprite)
    #            = 435 - (141 - 226) = 435 + 85 - wait it's:
    #   paste_cy + 85 = 435   →   paste_cy = 350
    # mstch_cy = 465, so DY = 350 - 465 = -115.
    _BEARD_DY           = -115      # ink centroid lands at front-chin canvas y
    _BEARD_DX           = -60       # outward mirror-pair spread
    _BEARD_SCALE        = 0.55      # uniform sprite scale

    # Default user params (CharInfo) — all 0 unless overridden.
    _mstch_scale_param  = 0
    _mstch_aspect_param = 0
    mstch_scale  = _mstch_scale_param  * _MUSTACHE_SCALE_SLOPE  + _MUSTACHE_SCALE_OFFSET
    mstch_aspect = _mstch_aspect_param * _MUSTACHE_ASPECT_SLOPE + _MUSTACHE_ASPECT_OFFSET
    # Engine width/height in canvas pixels. 1 engine unit = FACE_RES *
    # 0.75 / 32. Empirical 1.5× scale on the formula values — full-extent
    # interpretation (1×) is too small, half-extent (2×) overshoots the
    # mouth. The exact semantic of the values stored at sp+0x38 isn't
    # pinned without disassembling the downstream render call (the
    # consumer of those values lives behind a vtable dispatch).
    _engine_to_px = FACE_RES * _FACE_PAINT_SCALE / 32.0   # = 14.0625 at FACE_RES=600
    _MUSTACHE_SIZE_FUDGE = 2.2
    mstch_w_px = _MUSTACHE_SIZE_FUDGE * mstch_scale * _MUSTACHE_W_BASE * _engine_to_px
    mstch_h_px = _MUSTACHE_SIZE_FUDGE * mstch_scale * _MUSTACHE_H_BASE * mstch_aspect * _engine_to_px
    # MSTCH_Y_BASE places the BOTTOM of the mustache (not the centre) at
    # ~mouth-row + small offset. To anchor by bottom, paste with the sprite
    # centre shifted up by half the rendered height.
    _MUSTACHE_Y_ANCHOR_DY = -mstch_h_px / 2.0
    # Pre-scale for _load_miiparts_sprite: target px / native texture px.
    # Mustache PNGs are 256 wide × 512 tall.
    _MUSTACHE_SX = mstch_w_px / 256.0
    _MUSTACHE_SY = mstch_h_px / 512.0
    centred = [
        ("Mouth",    expr_ovr.get("mouth") or sel.get("Mouth", "(none)"),
                     mouth_cy, _ffl_scale(mouth_scale) * mth_smul * A, mth_rot),
        # Mustache scale comes from the bit-exact engine formula above
        # (non-uniform sx ≠ sy). _paste_sprite takes a single scale, so
        # we pre-scale the sprite asymmetrically via _load_miiparts_sprite
        # (which accepts a (sx, sy) tuple) and paste at scale=1.0. Y from
        # the engine formula `y_param * 1.076094 + 31.763554` lives in
        # mstch_cy already (computed at the top of compose_face_texture).
        # Rotation: per-shape `RotateAxis.Y` (bgyml radians, eg 0.05 ≈
        # 2.9°). Mirror copy gets the opposite tilt automatically (the
        # mirror is taken after the rotation), giving the outward-tilt
        # symmetric look.
        ("Mustache", sel.get("Mustache", "(none)"),
                     mstch_cy + _MUSTACHE_Y_ANCHOR_DY, 1.0,
                     _mustache_rotate_deg(sel.get("Mustache", "(none)"))),
        ("Beard",    sel.get("Beard",    "(none)"),
                     mstch_cy + _BEARD_DY, _BEARD_SCALE, 0.0),
        # Mole compositor (`FUN_71001764f0`, located via the Y_BASE
        # load @ 0x17662c). Bit-exact engine formulas:
        #   mole.pos.x  = posX  * 1.7792293 + 17.766164  (@0x29eabdc / 0x29eb214)
        #   mole.pos.y  = posY  * 1.0760943 + 17.959862  (@0x29eb018 / 0x29eaf24)
        #   mole.scale  = scale * 0.4       + 1.0        (@0x29eb444  — FFL scale)
        #   mole.aspect = aspect* 0.12      + 0.64       (@0x29eae74 / 0x29ead0c)
        # The engine→canvas X factor for these outputs isn't statically
        # decodable (same projection-matrix blocker as the eye Y per
        # LASH_REPORT). Engine-derived `_engine_y(17.96) ≈ 270` would
        # put the mole at the same Y as the eye centre and overlap;
        # shift it onto the lower cheek empirically until the X factor
        # is pinned. Scale 0.10 lands the mole at ~13 px on the canvas.
        ("Mole",     sel.get("Mole",     "(none)"),
                     _engine_y(_NSO_EYE_Y_BASE) + 50,     # just under the eye
                     0.10, 0.0),
        ("Faceline", sel.get("Faceline", "(none)"), FACE_CY,  1.0, 0.0),
    ]
    # Mustache is a TRUE half-mask: all R-channel content sits in the
    # right half of the 256-wide texture (left half empty), with the
    # texture's RIGHT edge meant to land on the face's vertical centre
    # line. Pasting with sprite-CENTER at FACE_CX (the previous behaviour)
    # leaves a gap because the content's right edge is at texture x=255,
    # not x=128 — so the original's content lands far right of centre and
    # the mirror's content lands far left, with empty space between.
    # Correct placement: original right-edge at FACE_CX (sprite centre at
    # FACE_CX − sprite_w/2), mirror left-edge at FACE_CX.
    #
    # 2D Beard textures, despite the comment elsewhere about half-masks,
    # actually have inked content in BOTH halves (left=30k, right=21k px
    # for BeardShort00) — the beard is full-width and roughly symmetric
    # already, so mirroring it just over-paints the same content. Render
    # once at FACE_CX with no mirror.
    _MUSTACHE_HALFMASK = {"Mustache"}
    for cat, name, cy, scale, rot in centred:
        if not name or name == "(none)":
            continue
        mp_cat = _MIIPARTS_CAT_MAP.get(cat)
        if not mp_cat:
            continue
        idx = _parse_feat_index(name)
        # Mustache uses NSO-derived non-uniform scale (engine width/height
        # 4.5 × 9.0*aspect, in 32-unit grid → canvas px ÷ texture native).
        load_scale = 1.0
        if cat == "Mustache":
            load_scale = (_MUSTACHE_SX, _MUSTACHE_SY)
        spr = _load_miiparts_sprite(mp_cat, idx,
                                    mouth_color=mouth_color, hair_color=hair_color,
                                    mouth_drop_lips=mouth_drop_lips,
                                    beard2d_color=beard2d_color,
                                    mustache_color=mustache_color,
                                    scale=load_scale)
        if spr is None:
            continue
        spr = _rotate_sprite(spr, rot)
        if cat == "Mole":
            # Beauty mole on the LEFT cheek (viewer's left = canvas
            # left = FACE_CX - eye spacing). The engine X formula
            # (mole.pos.x = posX*1.7792 + 17.766) is documented above
            # but the canvas-pixel mapping isn't pinned, so the
            # offset stays empirical for now.
            cx = FACE_CX - _engine_spacing_x(2)
        else:
            cx = FACE_CX
        if cat in _MUSTACHE_HALFMASK:
            # Original right-edge at FACE_CX, mirror left-edge at FACE_CX
            # (texture's right edge = face vertical centre line).
            # _MUSTACHE_OVERLAP shifts each half 1 px past the centre to
            # cover the rasterisation seam between the two pastes.
            half_w = (spr.shape[1] * scale) / 2.0 - _MUSTACHE_OVERLAP
            spr_mirror = spr[:, ::-1, :].copy()
            _paste_sprite(canvas, spr,        cx - half_w, cy, scale)
            _paste_sprite(canvas, spr_mirror, cx + half_w, cy, scale)
        elif cat == "Beard":
            # BeardShort wraps the curved 3D chin → paint on head_canvas
            # (Head mesh) instead of the flat-decal canvas (Mask mesh).
            # Texture is biased to one side; mirror-pair gives symmetric
            # coverage of both cheeks/jaw.
            spr_mirror = spr[:, ::-1, :].copy()
            _paste_sprite(head_canvas, spr,        cx + _BEARD_DX, cy, scale)
            _paste_sprite(head_canvas, spr_mirror, cx - _BEARD_DX, cy, scale)
            any_head_drawn = True
            continue   # don't bump any_drawn — beard isn't on the Mask
        else:
            _paste_sprite(canvas, spr, cx, cy, scale)
        any_drawn = True

    face_img = Image.fromarray(canvas.astype(np.uint8)) if any_drawn else None
    head_paint_img = Image.fromarray(head_canvas.astype(np.uint8)) if any_head_drawn else None
    if face_img is None and head_paint_img is None:
        return None
    # Backward-compat: if no head paint, return single image as before.
    if head_paint_img is None:
        return face_img
    return (face_img, head_paint_img)


# ── UV-mapped texture application ─────────────────────────────────────────────
def _apply_face_texture_to_scene(scene, face_img, fallback_skin_rgba):
    """
    Texture the head meshes:
      - Mask mesh ← `face_img` (eyes/brows/mouth/mustache/mole/Make — flat
        decal features). Mask is a wide planar overlay in front of the
        face that doesn't follow the 3D head curvature, so it's the right
        target for features rendered as flat sprites.
      - Head mesh ← `head_paint_img` (BeardShort, Wrinkle — features that
        wrap the curved 3D chin/cheeks). The Head mesh has the actual 3D
        skin geometry; its UVs (u[0..2] tile, v[0..1]) sample the texture
        across the head surface, so painting beard ink at the canvas
        position the Head's chin samples puts it correctly on the curved
        chin in 3D.

    `face_img` may be a single PIL Image (legacy) OR a tuple
    `(face_img, head_paint_img)` — the new compose_face_texture returns
    the tuple form whenever any head-paint feature is selected.
    """
    # Unpack tuple form
    if isinstance(face_img, tuple):
        face_img, head_paint_img = face_img
    else:
        head_paint_img = None

    # Same fix as _colorize: explicitly set metallic=0/roughness=1 so the
    # glTF 2.0 default metallic=1.0 doesn't render skin as a chrome ball.
    skin_mat = PBRMaterial(baseColorFactor=np.array(fallback_skin_rgba, dtype=float),
                           metallicFactor=0.0, roughnessFactor=1.0)
    mask_geoms = []
    head_geoms = []
    for name, geom in list(scene.geometry.items()):
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if name.startswith("Mask"):
            mask_geoms.append((name, geom))
        elif name.startswith("Head"):
            head_geoms.append((name, geom))
            geom.visual.material = skin_mat
        else:
            geom.visual.material = skin_mat

    out = trimesh.scene.Scene()
    for name, geom in scene.geometry.items():
        if not name.startswith("Mask"):
            out.add_geometry(geom, geom_name=name)

    if face_img is None and head_paint_img is None:
        return scene
    # Mask mesh ← face_img (flat-decal features)
    if face_img is not None:
        for name, geom in mask_geoms:
            uv = getattr(geom.visual, "uv", None)
            if uv is None or len(uv) == 0:
                continue
            feat = geom.copy()
            feat.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                material=PBRMaterial(baseColorTexture=face_img,
                                     alphaMode="BLEND", doubleSided=True),
            )
            out.add_geometry(feat, geom_name=name + "_feat")
    # Head mesh ← head_paint_img (skin-wrapping features: BeardShort, Wrinkle).
    # Head's native UVs put the u=1.0 SEAM at the front-center of the face,
    # so the front-chin samples u[0.59..1.41] — two disjoint canvas regions
    # at x[354..600] and x[0..246] (split by the seam).
    #
    # We shift u by +0.5 so the seam moves to the BACK of the head; the
    # front-chin then samples a contiguous canvas region x[54..546]
    # centered on the canvas. We paint the beard at canvas center, and
    # this remap puts it on the front-chin in 3D.
    if head_paint_img is not None:
        for name, geom in head_geoms:
            uv = getattr(geom.visual, "uv", None)
            if uv is None or len(uv) == 0:
                continue
            uv_shifted = uv.copy()
            uv_shifted[:, 0] = (uv_shifted[:, 0] + 0.5) % 1.0
            feat = geom.copy()
            feat.visual = trimesh.visual.TextureVisuals(
                uv=uv_shifted,
                material=PBRMaterial(baseColorTexture=head_paint_img,
                                     alphaMode="BLEND", doubleSided=True),
            )
            out.add_geometry(feat, geom_name=name + "_paint")
    return out


# ── GLB catalogue ─────────────────────────────────────────────────────────────
def _load_cats():
    cats = {}
    for f in sorted(glob.glob(f"{GLB_DIR}/Mii*.glb")):
        name = os.path.basename(f).replace(".glb", "")
        m = re.match(r"Mii([A-Za-z]+?)(\d+)$", name)
        if m:
            cats.setdefault(m.group(1), []).append(f)
    return cats

CATS = _load_cats()

PARTS = [
    ("Head",          "Face Shape",    True,  0),
    # Nose=01, Ear=none, HairAllLegacy=Hair012, HairAll=none. The legacy
    # and "new" hair categories must not be used together (they overlap),
    # so only HairAllLegacy gets a default — and a UI handler below clears
    # the other when either is selected.
    ("Nose",          "Nose",          False, 4),     # opts[5] = '01'
    ("Ear",           "Ears",          False, None),
    ("HairAll",       "Hair (new)",    False, None),
    ("HairAllLegacy", "All",           False, None),  # default forced to "L:012" in _build_3d
    ("HairFront",     "Hair Front",    False, None),
    ("HairBack",      "Hair Back",     False, None),
    ("HairParts",     "Hair Parts",    False, None),
    ("Beard",         "Beard",         False, None),
    ("Glass",         "Glasses",       False, None),
]

# In-game editor icon pack: `MiiEditorIcon_MiiEditor_Face_<Cat><N>_Uit.png`
# (some entries also have a `…<N>color_Uit.png` tint-aware variant). Game UI
# uses these for the part picker, so reusing them gives the most faithful look.
# Bundled inside the project under `assets/editor_icons/` so the renderer
# stays self-contained — no external `extract_dir` reference at runtime.
_EDITOR_ICONS = os.path.join(_ASSETS, "editor_icons")
# Hair surface-detail textures: grayscale luminance maps preprocessed
# from the in-game `MiiHair*_Mim.png` (vertical-stripe diffuse map that
# gives the hair its strand pattern). Applied as `baseColorTexture` on
# unlit hair materials so the rendered pixel becomes
# `sRGB(baseColorFactor) * grayscale_strand`. Filenames mirror the
# matching GLB basename (`MiiHairAllLegacy012.glb` →
# `MiiHairAllLegacy012.png`).
_HAIR_TEX_DIR = os.path.join(_ASSETS, "hair_textures")
# Editor icon names diverge from internal category names in a few places.
_EDITOR_NAME_REWRITES = {
    "EyelashUpper":  "Eyelash",  # editor merges Upper into base "Eyelash"
    "Highlight":     "EyeHighlight",
    "Head":          "Faceline", # 3D head shape uses the Faceline editor icon
    "HairAllLegacy": "Hair",     # legacy hair uses the plain "Hair" editor icons
}
# Per-category in-game "no selection" icons (where the editor pack ships one).
# Categories not listed fall back to the transparent placeholder tile.
_EDITOR_NONE_ICONS = {
    "Ear":          "Ear99",
    "Glass":        "GlassNothing",
    "Beard":        "BeardNothing",
    "EyelashLower": "EyelashLowerNothing",
    "Highlight":    "EyeHighlightNothing",
}

# NOTE: an earlier revision wired in-game category-tab icons
# (`__Combined_IconCat*^s.png` from the editor layout blarc) onto the
# accordion titles; that path was dropped when the picker switched to a
# nested-Tabs layout. The corresponding asset directory is no longer
# referenced and is intentionally not bundled with the project.

def _editor_icon_path(name):
    """Return the editor-pack PNG for a part name (e.g. 'Eye041',
    'Nose04', 'Hair073'), trying the plain and `color` variants. None if
    not present in the pack."""
    for cand in (name, name + "color"):
        p = os.path.join(_EDITOR_ICONS, f"MiiEditorIcon_MiiEditor_Face_{cand}_Uit.png")
        if os.path.exists(p):
            return p
    return None

def _bgyml_editor_icon(part_name):
    """Resolve a part's EditorIconName from its bgyml entry to a PNG path.
    Several parts (e.g. Mole) have an EditorIconName that diverges from
    their FileName — Mole00 uses Mole01_Uit, MoleNothing uses Mole00_Uit."""
    entry = mm.parts().get(part_name, {})
    icon_name = entry.get("EditorIconName")
    if not icon_name:
        return None
    # EditorIconName is e.g. 'MiiEditor_Face_Mole01_Uit' — strip the
    # 'MiiEditor_Face_' prefix and the '_Uit' suffix to feed _editor_icon_path.
    base = icon_name
    if base.startswith("MiiEditor_Face_"):
        base = base[len("MiiEditor_Face_"):]
    if base.endswith("_Uit"):
        base = base[:-len("_Uit")]
    return _editor_icon_path(base)

_FORCE_EMPTY_NONE_ICON = {"Eyebrow"}

def _none_icon_path(cat):
    """Per-category 'no selection' icon, falling back to the transparent tile.

    Manual override map wins (some categories' bgyml `<Cat>Nothing` entries
    point at OTHER categories' empty icons — e.g. GlassNothing's
    EditorIconName is BeardNothing — and the explicit override picks the
    right one). Falls back to the bgyml lookup, then the placeholder.
    Categories in `_FORCE_EMPTY_NONE_ICON` skip the bgyml lookup so their
    "(none)" tile is unambiguously empty."""
    if cat in _FORCE_EMPTY_NONE_ICON:
        return _empty_tile_path()
    name = _EDITOR_NONE_ICONS.get(cat)
    if name:
        p = _editor_icon_path(name)
        if p: return p
    p = _bgyml_editor_icon(f"{cat}Nothing")
    if p: return p
    return _empty_tile_path()

def _glb_icon_path(cat, label):
    """Best icon PNG for a 3D part: prefer the editor-pack icon, fall back
    to per-category mesh-texture variants, then the transparent placeholder.
    For Glass labels with `@<level>` suffix, returns a cached badge variant."""
    if label == "(none)" or not label:
        return _none_icon_path(cat)
    if cat == "Glass" and "@" in label:
        return _glass_variant_icon(label)
    if cat == "HairBack" and ":" in label:
        # Compound `<HHH>:<PPP>@<X>` → reuse the HairParts compound icon.
        return _glb_icon_path("HairParts", label)
    if cat == "HairAllLegacy" and label[:2] in ("L:", "N:"):
        # Unified Hair label: route to the right category's icon. Legacy
        # uses the "Hair<NNN>" editor pack name (per
        # _EDITOR_NAME_REWRITES), new uses "HairAll<NNN>".
        kind, sub = label.split(":", 1)
        return _glb_icon_path("HairAllLegacy" if kind == "L" else "HairAll", sub)
    if cat == "HairParts":
        # Label format: `<HHH>:<PPP>@<X>` (preferred — gallery emits
        # this) or any of the legacy forms (`<PPP>`, `<PPP>@<X>`,
        # `<HHH>:<PPP>`). The icon is `HairBack<HHH>Parts<PPP>_<X>_Uit`
        # with HHH defaulting to '000' and X defaulting to the part's
        # bgyml suffix when omitted.
        rest = label
        hb, pp = "000", rest
        if ":" in rest:
            hb, rest = rest.split(":", 1)
        pp, _, suf = rest.partition("@")
        if suf:
            p = _editor_icon_path(f"HairBack{hb}Parts{pp}_{suf}")
            if p: return p
        # Fall back to the part's bgyml-canonical icon, then sweep
        # C/U/D for the HB.
        p = _bgyml_editor_icon(f"HairParts{pp}")
        if p: return p
        for s in ("C", "U", "D"):
            p = _editor_icon_path(f"HairBack{hb}Parts{pp}_{s}")
            if p: return p
        label = pp
    edit_cat = _EDITOR_NAME_REWRITES.get(cat, cat)
    p = _editor_icon_path(f"{edit_cat}{label}")
    if p: return p
    return _empty_tile_path()


def _glass_variant_icon(label):
    """Composite a level badge onto the base glass editor icon and cache
    to /tmp. Three variants per glass mesh (clear / tinted / shaded) so
    the gallery shows three options per row of frames."""
    mesh_label, level_id = _glass_split_label(label)
    cache_dir = os.path.join(tempfile.gettempdir(), "mii_face_renderer_glass_icons")
    os.makedirs(cache_dir, exist_ok=True)
    # Bump the version suffix any time the composition recipe changes —
    # otherwise Gradio re-uses the already-cached file path and browsers
    # serve a stale image.
    out_path = os.path.join(cache_dir, f"Glass{mesh_label}@{level_id}_v4.png")
    if os.path.exists(out_path):
        return out_path
    base = _editor_icon_path(f"Glass{mesh_label}")
    if not base:
        return _empty_tile_path()
    name, alpha, badge = _GLASS_LEVEL_BY_ID.get(level_id, ("?", 0.0, "?"))
    glass = Image.open(base).convert("RGBA")
    arr = np.array(glass)
    # Modulate the lens-fill region (where R > 60 — bright pixel,
    # i.e. inside-the-frame fill — while alpha > 0). The frame outline
    # itself (dark pixels) stays untouched on every variant.
    R = arr[..., 0]
    A = arr[..., 3]
    fill_mask = (R > 60) & (A > 0)
    if level_id == 0:        # clear: drop the lens fill alpha
        arr[fill_mask, 3] = 0
    elif level_id == 1:      # tinted: black with 30% alpha
        arr[fill_mask, :3] = 0
        arr[fill_mask,  3] = 76
    else:                    # shaded: solid black
        arr[fill_mask, :3] = 0
        arr[fill_mask,  3] = 255
    glass = Image.fromarray(arr)
    # GlassNothing is already a filled face silhouette (alpha=255 inside,
    # not just an outline) — composite the glass on top.
    backdrop_path = _none_icon_path("Glass")
    if backdrop_path and os.path.exists(backdrop_path):
        backdrop = Image.open(backdrop_path).convert("RGBA")
        if backdrop.size != glass.size:
            backdrop = backdrop.resize(glass.size, Image.LANCZOS)
        composed = Image.alpha_composite(backdrop, glass)
    else:
        composed = glass
    # Source icons are 150×150 — the gallery upscales them and the hard
    # alpha thresholds applied above (R>60 fill mask) leave jagged edges.
    # Up-sample 2× with LANCZOS and softly blur the alpha channel so the
    # rendered tiles read as anti-aliased.
    target = (composed.width * 2, composed.height * 2)
    composed = composed.resize(target, Image.LANCZOS)
    arr2 = np.array(composed)
    a_smooth = np.array(Image.fromarray(arr2[..., 3]).filter(ImageFilter.GaussianBlur(0.6)))
    arr2[..., 3] = a_smooth
    Image.fromarray(arr2).save(out_path)
    return out_path

def _face_icon_path(prefix, name, *, mouth_color=0, eye_color=None):
    """Best icon PNG for a 2D face-feature part. For Mouth, regenerate
    from the texture with the user-selected mouth_color. For Eye,
    composite the editor's `<name>color_Uit.png` colour layer (iris
    white-fill) with the `<name>_Uit.png` outline at the selected iris
    colour. `eye_color` defaults to slot 0 in _NSO_EYE_B_COLORS (Black)
    so the initial gallery render matches the picker's default."""
    if eye_color is None:
        eye_color = _NSO_EYE_B_COLORS[0]
    if prefix == "Mouth":
        p = _mouth_colored_icon(name, mouth_color)
        if p: return p
    if prefix == "Eye":
        p = _eye_colored_icon(name, eye_color)
        if p: return p
    p = _bgyml_editor_icon(name)
    if p: return p
    rewrite = _EDITOR_NAME_REWRITES.get(prefix)
    if rewrite and name.startswith(prefix):
        candidate = rewrite + name[len(prefix):]
        p = _editor_icon_path(candidate)
        if p: return p
    p = _editor_icon_path(name)
    if p: return p
    # Mesh-texture extracted PNG (current behaviour).
    mp_cat = _MIIPARTS_CAT_MAP.get(prefix)
    if mp_cat:
        idx = _parse_feat_index(name)
        path = _miiparts_path(mp_cat, idx)
        if path and os.path.exists(path):
            return path
    return _empty_tile_path()


def _glb_choices(cat):
    files  = CATS.get(cat, [])
    labels = [os.path.basename(f).replace(".glb","").replace(f"Mii{cat}","") for f in files]
    # Reorder by in-game PartsOrder bgyml when available. The order list
    # holds integer PartsIndex values, which often differ from the GLB
    # filename suffix (e.g. for Ear: Ear00 → PartsIndex 1, Ear01 → 2, …).
    # Resolve each label to its bgyml PartsIndex first, then rank.
    order = mm.parts_order().get(cat, [])
    if order:
        # HairBack's PartsOrder file packs the PartsIndex into the high
        # 16 bits of each entry (`(PartsIndex << 16) | 0x0000`) rather
        # than storing it directly the way every other category does.
        # Without this the ranking dict's keys never match the
        # bgyml-derived PartsIndex (e.g. HairBack000 PartsIndex 206 vs.
        # the order-file value 0x00ce0000) and the gallery falls back
        # to filename order, which is wrong. Detect the encoding by
        # checking that the low 16 bits are always zero.
        if all(isinstance(v, int) and (v & 0xFFFF) == 0 and v >= 1 << 16
               for v in order):
            order = [v >> 16 for v in order]
        rank = {idx: i for i, idx in enumerate(order)}
        parts = mm.parts()
        def _parts_index(lbl):
            entry = parts.get(f"{cat}{lbl}", {})
            pi = entry.get("PartsIndex")
            if pi is not None:
                return pi
            try: return int(lbl)
            except ValueError: return None
        def _key(lbl):
            pi = _parts_index(lbl)
            if pi is None or pi not in rank:
                return (1, lbl)
            return (0, rank[pi])
        labels = sorted(labels, key=_key)
    if cat == "Glass":
        # Triple every glass mesh into clear / tinted / shaded variants
        # (level encoded in the label as `@N` suffix). Group by level
        # first so the gallery shows all 19 clear frames, then all 19
        # tinted, then all 19 shaded.
        labels = [f"{lbl}@{lid}" for lid, *_ in _GLASS_LEVELS for lbl in labels]
    elif cat == "HairAllLegacy":
        # Unified gallery for HairAll (new Switch 2) + HairAllLegacy
        # (legacy 8-color hair). Both share Category="Hair" in the
        # bgyml metadata, and Hair.mii__PartsOrder enumerates both
        # streams interleaved by PartsIndex (legacy 0..129, new 130..340
        # ish). Labels are prefixed `L:` (legacy) or `N:` (new) so
        # _glb_path / _glb_icon_path can route to the right CATS bucket
        # without a global state lookup. HairBack entries also share
        # Category="Hair" but live in their own subtab — filter them
        # out here.
        new_files    = CATS.get("HairAll", [])
        new_labels   = [os.path.basename(f).replace(".glb","").replace("MiiHairAll","")
                        for f in new_files]
        legacy_set   = set(labels)
        new_set      = set(new_labels)
        parts        = mm.parts()
        pi_to_label  = {}
        for n, e in parts.items():
            pi = e.get("PartsIndex")
            if pi is None:
                continue
            if n.startswith("HairAllLegacy"):
                lbl = n.replace("HairAllLegacy", "")
                if lbl in legacy_set:
                    pi_to_label.setdefault(pi, ("L", lbl))
            elif n.startswith("HairAll"):
                lbl = n.replace("HairAll", "")
                if lbl in new_set:
                    pi_to_label.setdefault(pi, ("N", lbl))
        # `HairAll.mii__PartsOrder` is the unified-Hair order file (245
        # entries, first PI=237 → HairAll058, then 41 → HairAllLegacy041,
        # …). The similarly named `Hair.mii__PartsOrder` is actually the
        # HairBack subtab order — using it here put HairAllLegacy041 in
        # front of HairAll058, which is wrong.
        raw = mm.parts_order().get("HairAll", [])
        expanded, seen = [], set()
        for v in raw:
            entry = pi_to_label.get(v)
            if entry is None:
                continue
            kind, lbl = entry
            key = f"{kind}:{lbl}"
            if key in seen:
                continue
            seen.add(key)
            expanded.append(key)
        # Defensive: append anything missing from PartsOrder.
        for lbl in labels:
            if f"L:{lbl}" not in seen:
                expanded.append(f"L:{lbl}")
        for lbl in new_labels:
            if f"N:{lbl}" not in seen:
                expanded.append(f"N:{lbl}")
        labels = expanded
    elif cat == "HairBack":
        # The HairBack PartsOrder file is actually a UNIFIED order for
        # bare HairBack tiles AND HairBack+HairParts compounds. Each
        # entry is packed as
        #   ((HairBack.PartsIndex << 16) | (HairParts.PartsIndex << 8) | rowIdx)
        # where rowIdx ∈ {0=U, 1=C, 2=D} maps to Upper/Middle/Lower
        # attachment rows. Low 16 bits == 0 → bare HairBack tile;
        # otherwise compound. Surfacing this list as a single
        # `<HHH>` / `<HHH>:<PPP>@<X>` gallery is what the in-game editor
        # actually shows: HairBack and HairParts are presented together
        # rather than as separate categories.
        raw = mm.parts_order().get("HairBack", [])
        parts = mm.parts()
        hb_pi  = {e["PartsIndex"]: n.replace("HairBack", "")
                  for n, e in parts.items()
                  if n.startswith("HairBack") and "PartsIndex" in e}
        hp_pi  = {e["PartsIndex"]: n.replace("HairParts", "")
                  for n, e in parts.items()
                  if n.startswith("HairParts") and "PartsIndex" in e}
        avail  = set(labels)
        suf_by_idx = ("U", "C", "D")
        row_by_idx = ("Upper", "Middle", "Lower")
        expanded, seen = [], set()
        for v in raw:
            if not isinstance(v, int):
                continue
            hb = hb_pi.get(v >> 16)
            if hb is None or hb not in avail:
                continue
            lo = v & 0xFFFF
            if lo == 0:
                key = hb
            else:
                pp = hp_pi.get(lo >> 8)
                ri = lo & 0xFF
                if pp is None or ri >= len(suf_by_idx):
                    continue
                # Defensive: drop compounds the engine would reject
                # via the HairBack's `IsAttachableHairParts<Row>` flag.
                # The icon pack only ships valid combos today, so this
                # is a no-op against bundled assets — kept so a future
                # asset drop with extra icons can't surface entries the
                # engine refuses to build.
                hb_entry = parts.get(f"HairBack{hb}", {})
                if not hb_entry.get(f"IsAttachableHairParts{row_by_idx[ri]}"):
                    continue
                key = f"{hb}:{pp}@{suf_by_idx[ri]}"
            if key in seen:
                continue
            seen.add(key)
            expanded.append(key)
        # Append any HairBack labels not enumerated by PartsOrder
        # (defensive — keeps unknown meshes pickable).
        for hb in labels:
            if hb not in seen:
                expanded.append(hb)
        labels = expanded
    return ["(none)"] + labels

def _glb_path(cat, label):
    if label == "(none)" or not label:
        return None
    if cat == "Glass" and "@" in label:
        label, _ = _glass_split_label(label)
    if cat == "HairParts":
        # Label format: optional `<HHH>:` HairBack prefix + `<PPP>` parts
        # index + optional `@<X>` row suffix. Only the `<PPP>` part
        # determines which MiiHairParts<PPP>.glb to load.
        if ":" in label:
            label = label.split(":", 1)[1]
        if "@" in label:
            label = label.split("@", 1)[0]
    if cat == "HairBack" and ":" in label:
        # Compound HairBack label `<HHH>:<PPP>@<X>` from the merged
        # gallery — the HairBack mesh is HairBack<HHH>; the parts
        # contribution is wired through HairParts in build_face.
        label = label.split(":", 1)[0]
    if cat == "HairAllLegacy" and label[:2] in ("L:", "N:"):
        # Unified Hair gallery: `L:<NNN>` → MiiHairAllLegacy<NNN>.glb,
        # `N:<NNN>` → MiiHairAll<NNN>.glb (the new Switch 2 hair).
        kind, sub = label.split(":", 1)
        if kind == "N":
            files = CATS.get("HairAll", [])
            labels_list = [os.path.basename(f).replace(".glb","").replace("MiiHairAll","")
                           for f in files]
            return files[labels_list.index(sub)] if sub in labels_list else None
        label = sub
    files  = CATS.get(cat, [])
    labels = [os.path.basename(f).replace(".glb","").replace(f"Mii{cat}","") for f in files]
    return files[labels.index(label)] if label in labels else None

# ── Skeleton / socket helpers ──────────────────────────────────────────────────
def _read_glb_nodes(path):
    with open(path, "rb") as f:
        f.read(12)
        cl = struct.unpack("<I", f.read(4))[0]; f.read(4)
        return {n["name"]: np.array(n.get("translation", [0.,0.,0.]))
                for n in json.loads(f.read(cl)).get("nodes", [])}

def _read_glb_nodes_full(path):
    """Like _read_glb_nodes but returns {name: (translation, quaternion)}."""
    with open(path, "rb") as f:
        f.read(12)
        cl = struct.unpack("<I", f.read(4))[0]; f.read(4)
        out = {}
        for n in json.loads(f.read(cl)).get("nodes", []):
            t = np.array(n.get("translation", [0.,0.,0.]))
            r = np.array(n.get("rotation",    [0.,0.,0.,1.]))
            out[n["name"]] = (t, r)
        return out

def _quat_to_mat4(q):
    """glTF quaternion (x,y,z,w) → 4x4 rotation matrix."""
    x, y, z, w = q
    M = np.eye(4)
    M[:3, :3] = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return M

def _get_sockets(head_path):
    nodes  = _read_glb_nodes(head_path)
    head_t = nodes.get("Head", np.zeros(3))
    return {k[4:]: head_t + v for k, v in nodes.items() if k.startswith("set_")}

def _get_socket_rotations(head_path):
    """{socket_name: quaternion (x,y,z,w)} for set_* nodes in the head GLB."""
    return {k[4:]: r for k, (_, r) in _read_glb_nodes_full(head_path).items()
            if k.startswith("set_")}

def _translation_matrix(offset):
    T = np.eye(4); T[:3, 3] = offset; return T

def _apply_matrix(scene, matrix):
    out = trimesh.scene.Scene()
    for name, geom in scene.geometry.items():
        g = geom.copy(); g.apply_transform(matrix)
        out.add_geometry(g, geom_name=name)
    return out

def _translate(scene, offset):
    return scene if np.allclose(offset, 0) else _apply_matrix(scene, _translation_matrix(offset))

def _mirror_x(scene):
    M = np.eye(4); M[0, 0] = -1
    return _apply_matrix(scene, M)

def _rot_x_180(scene):
    """180° rotation about the X axis: maps (x, y, z) → (x, -y, -z)."""
    M = np.eye(4); M[1, 1] = -1; M[2, 2] = -1
    return _apply_matrix(scene, M)

def _rot_y_90(scene):
    """+90° rotation about the Y axis: maps (x, y, z) → (z, y, -x).
    Sends a vector pointing along +Z to one pointing along +X."""
    M = np.eye(4)
    M[0, 0] = 0; M[0, 2] = 1
    M[2, 0] = -1; M[2, 2] = 0
    return _apply_matrix(scene, M)

_hair_mim_cache = {}    # tex_path → np.ndarray (R, G channels)
_hair_strand_cache = {} # (tex_path, t0, t1) → PIL.Image

def _hair_strand_texture(tex_path, tint0_lin, tint1_lin):
    """Build (and memoise) the per-pixel hair texture: each pixel is
    `base × dim + R/255 × tint0 + G/255 × tint1`, where R/G come from
    the bundled Mim and tint0/tint1 are linear RGB triples.

    The `base × dim` floor (with dim ≈ 0.4 of tint0) ensures non-strand
    texels still carry the picked hair color instead of going pure black.
    Without this, the texture is `R × tint0 + G × tint1` which collapses
    to black wherever neither strand channel fires — producing a uniform
    black band on the underside / inside of the hair geometry, very
    different from the game where the entire hair surface reads as the
    picked color (just dimmer on the shadow side).

    Returns an sRGB PIL image suitable for use as `baseColorTexture`
    under KHR_materials_unlit (with `baseColorFactor=(1,1,1,1)`)."""
    t0 = tuple(round(float(c), 4) for c in tint0_lin[:3])
    t1 = tuple(round(float(c), 4) for c in tint1_lin[:3])
    cache_key = (tex_path, t0, t1)
    cached = _hair_strand_cache.get(cache_key)
    if cached is not None:
        return cached
    if tex_path not in _hair_mim_cache:
        rgb = np.asarray(Image.open(tex_path).convert("RGB"), dtype=np.float32)
        _hair_mim_cache[tex_path] = rgb[..., :2] / 255.0
    rg = _hair_mim_cache[tex_path]
    R = rg[..., 0:1]; G = rg[..., 1:2]
    # Base floor: blend the strand pattern over a baseline of tint0 so
    # non-strand areas render as the picked hair color rather than pure
    # black. 0.7 keeps strand contrast visible (~30% brightness diff
    # between strand pixel and base) while keeping the whole hair surface
    # close to the picked color — matches the game's "consistent color
    # with subtle variation" look. (Was 0.4 → produced visibly darker
    # underside than the game.)
    BASE_DIM = 0.7
    base = np.asarray(t0, dtype=np.float32) * BASE_DIM
    strand = R * np.asarray(t0, dtype=np.float32) \
           + G * np.asarray(t1, dtype=np.float32)
    # max() so strand pixels override the base where they are brighter;
    # base shows through where neither R nor G fires.
    blended_lin = np.maximum(base, strand)
    blended_srgb = (_linear_to_srgb(np.clip(blended_lin, 0, 1)) * 255.0
                    ).astype(np.uint8)
    img = Image.fromarray(blended_srgb)
    if len(_hair_strand_cache) > 16:
        _hair_strand_cache.pop(next(iter(_hair_strand_cache)))
    _hair_strand_cache[cache_key] = img
    return img

def _apply_hair_strand_texture(scene, glb_path, tint0_lin, tint1_lin,
                               use_subcolor):
    """Bake the in-game two-channel hair shader into a baseColorTexture:
    `R/255 × color0 + G/255 × color1` (per the `mii_hair_color0/1`
    uniforms in the BFRES shader). When `use_subcolor` is False, force
    color1 = color0 so the G stripe just adds extra coverage of the
    main tint; when True, color1 = the user's sub-colour, giving the
    bright per-strand highlight stripe.

    The combined RGB texture replaces the flat baseColorFactor we set
    in `_colorize` (factor reset to white so unlit pixel = texture)."""
    if not glb_path:
        return scene
    base = os.path.basename(glb_path).rsplit(".", 1)[0]
    base = re.sub(r"_(?:Upper|Middle|Lower)(?:Left|Right|LeftRight|Center)$",
                  "", base)
    base = re.sub(r"_Flip$", "", base)
    tex_path = os.path.join(_HAIR_TEX_DIR, f"{base}.png")
    if not os.path.exists(tex_path):
        return scene
    t1 = tint1_lin if use_subcolor else tint0_lin
    img = _hair_strand_texture(tex_path, tint0_lin, t1)
    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mat = getattr(geom.visual, "material", None)
        if isinstance(mat, PBRMaterial):
            mat.baseColorTexture = img
            mat.baseColorFactor = np.array([1.0, 1.0, 1.0, 1.0])
    return scene

def _colorize(scene, rgba):
    if rgba is None: return scene
    color = np.array(rgba, dtype=float)
    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        # Preserve meshes that already carry a textured visual (e.g. the
        # noseline quad, replaced upstream with the editor icon at
        # alphaMode=BLEND). Overwriting them with `baseColorFactor=skin`
        # turns them back into opaque skin-coloured planes — that was the
        # source of the gray "plane through the nose" the user kept seeing.
        vis = geom.visual
        has_texture = (isinstance(vis, trimesh.visual.TextureVisuals)
                       and getattr(vis.material, "baseColorTexture", None) is not None)
        if has_texture:
            continue
        # Force dielectric + matte: glTF 2.0 defaults `metallicFactor`
        # AND `roughnessFactor` to 1.0, which makes skin / hair render
        # fully metallic-and-rough. Skin / hair / lashes are dielectric
        # (no metallic reflections) and approximately matte in the
        # toon-shader the engine uses, so explicitly clamp both factors.
        geom.visual.material = PBRMaterial(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=1.0)
    return scene

# `GlassLensOpacity` from MiiSystem/System.mii__SystemParam.bgyml. Mesh
# names in the extracted GLBs are `Flame__mt_Body` (frame), `Opa__mt_LensOpa`
# (opaque lens), `Trs__mt_LensTrs` (transparent lens) — same as the BFRES
# material names referenced in main.nso .rodata (`mt_LensTrs` @ 0x28ad0ef,
# `mt_LensOpa` @ 0x286002f, `mt_Body` @ 0x2825462).
#
# The bgyml `GlassLensOpacity = 0.7` is the engine's BLEND factor for the
# lens *colour over the eyes behind it* — i.e. the lens contribution stays
# at 70%, but the rest is the eye colour showing through. In glTF /
# Three.js BLEND we set baseColorFactor.a to the SEE-THROUGH amount, so
# we use (1 - 0.7) = 0.3 = 30% alpha (70% pass-through).
_NSO_GLASS_LENS_OPACITY = 0.7  # System.mii__SystemParam.bgyml.yaml

# Per-mesh-index opacity LEVEL for the lens. Each glass has 3 variants:
#   level 0 → no lens mesh rendered (clear, see-through)
#   level 1 → Trs lens rendered with α = 1 - GlassLensOpacity = 0.3
#   level 2 → Opa lens rendered fully opaque (sunglasses)
# The bgyml ships a single global GlassLensOpacity (0.7); the actual
# per-MII choice of which level to use is encoded in CharInfo bits the
# engine resolves at render time. We expose all three as separate
# gallery entries so the user can try each.
_GLASS_LEVELS = [
    (0, "clear",     0.0, "○"),
    (1, "tinted",    0.7, "◐"),
    (2, "shaded",    1.0, "●"),
]
_GLASS_LEVEL_BY_ID = {lid: (name, alpha, badge) for lid, name, alpha, badge in _GLASS_LEVELS}

def _glass_split_label(label):
    """Decode a Glass gallery label like '00@1' into (mesh_label, level_id)."""
    if "@" in label:
        m, _, lv = label.partition("@")
        try: return m, int(lv)
        except ValueError: pass
    return label, 1   # default tinted
# Sentinel value: tags the lens material with alphaMode=BLEND so the
# post-export GLB patch can find it and swap in the transmission
# extension. The actual transparency comes from KHR_materials_transmission
# in _patch_glb_unlit_lens, not from this alpha.
_GLASS_LENS_ALPHA       = 0.99

def _colorize_glass(scene, rgba, glass_label=None):
    """Apply user glass colour with the binary's frame/lens convention.
    Three meshes per glass GLB:
      Flame__mt_Body   — frame, opaque, user-picker colour.
      Opa__mt_LensOpa  — opaque lens; rendered for level 1 styles.
      Trs__mt_LensTrs  — transparent lens; rendered for level 0.7,
                         dropped for level 0.

    Per Babylon.js Viewer v2 docs, factor-based `baseColorFactor[3]`
    transparency is overpowered by the auto-loaded IBL environment
    (the Viewer applies a default HDR env to any PBR material). The
    existing _apply_face_texture_to_scene uses a *textured* RGBA where
    the alpha channel survives the IBL pass — apply the same pattern
    here: a 1×1 PIL Image with the lens RGBA, sampled by the lens UVs,
    keeps the see-through behaviour intact."""
    if rgba is None: return scene
    frame_color = np.array(rgba, dtype=float)
    # 1×1 RGBA texture: light-grey tint with low alpha so the eyes
    # show through. (1 - GlassLensOpacity) → 0.3 alpha.
    _, level_id = _glass_split_label(glass_label or "")
    _, level_alpha, _ = _GLASS_LEVEL_BY_ID.get(level_id, ("?", 0.7, "?"))
    a = int(round((1.0 - _NSO_GLASS_LENS_OPACITY) * 255))
    # Lens tint is BLACK by default (matches real-glass darkening — a
    # neutral-grey tint just looked white-washed).
    lens_tex = Image.new("RGBA", (4, 4), (0, 0, 0, a))
    out = trimesh.scene.Scene()
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        is_trs = "mt_LensTrs" in name
        is_opa = "mt_LensOpa" in name
        if level_alpha == 0.0 and (is_trs or is_opa):
            continue                       # clear: no lens mesh at all
        if level_alpha == 1.0 and is_trs:
            continue                       # shaded: Opa only
        if level_alpha == 0.7 and is_opa:
            continue                       # tinted: Trs only
        geom = geom.copy()
        if is_trs:                          # tinted — translucent
            geom.visual.material = PBRMaterial(
                baseColorTexture=lens_tex,
                baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                alphaMode="BLEND", doubleSided=True,
                metallicFactor=0.0, roughnessFactor=1.0)
        elif is_opa:                        # shaded — opaque sunglasses
            # Pure-black lens regardless of frame colour; matches the
            # in-game look (sunglass lenses don't pick up the frame tint).
            geom.visual.material = PBRMaterial(
                baseColorFactor=np.array([0.0, 0.0, 0.0, 1.0]),
                doubleSided=True,
                metallicFactor=0.0, roughnessFactor=1.0)
        else:                               # frame — always opaque user colour
            geom.visual.material = PBRMaterial(
                baseColorFactor=frame_color, doubleSided=True)
        out.add_geometry(geom, geom_name=name)
    return out

SKIP = ()

def _filter(scene):
    if not SKIP:
        return scene
    keep = {k: v for k, v in scene.geometry.items()
            if not any(k == p or k.startswith(p+".") for p in SKIP)}
    if len(keep) == len(scene.geometry): return scene
    out = trimesh.scene.Scene()
    for k, v in keep.items(): out.add_geometry(v, geom_name=k)
    return out

SKIN_CATS  = {"Head", "Nose", "Ear"}
HAIR_CATS  = {"HairAll", "HairAllLegacy", "HairFront", "HairBack", "HairParts"}
BEARD3D_CATS = {"Beard"}
GLASS_CATS = {"Glass"}


# Each MiiHairParts<NNN>.glb ships up to 12 sub-meshes — one per attachment
# slot named `MiiHairParts<NNN>_<Position>` where Position ∈ {Upper, Middle,
# Lower} × {Left, Right, LeftRight, Center}. The MiiHairPartsLocator.glb
# enumerates the same 9 anchor names (no LeftRight — those are paired
# left+right meshes baked into a single sub-mesh). The engine picks ONE
# slot per HairParts choice; the editor's per-Parts EditorIconName encodes
# the canonical slot via its `_C` / `_U` / `_D` suffix:
#   _U → Upper row,  _C → Middle row,  _D → Lower row.
# Within a row we default to the Center variant (single piece) and fall
# back through LeftRight → Left → Right → other rows when that position
# isn't authored for the chosen Parts NNN. Without this filter the
# rendered head shows ALL 12 attachments at once — which is the bug.
_HAIR_PARTS_POS_PRIORITY_SIDES = ("Center", "LeftRight", "Left", "Right")
_HAIR_PARTS_POS_PRIORITY_ROWS  = ("Middle", "Upper", "Lower")
_HAIR_PARTS_ROW_BY_SUFFIX      = {"U": "Upper", "C": "Middle", "D": "Lower"}


def _overlay_ink_offset_native(cat, idx):
    """Per-sprite (ink_dx, ink_dy) in NATIVE pixels = ink centroid −
    frame centre. The engine's lash compositor anchors the QUAD CENTRE
    at lash.pos.{x,y}, but our extracted PNGs store the ink in the
    bottom half of the frame (EyelashUpper at native (26, 65) on a
    96×80 frame, so dx/dy = (−22, +25)). `_paste_sprite` centres on
    the FRAME, so without this shift the visible ink lands ~25 native
    px below where the engine intends. Compensating moves the
    sprite's ink centroid to the formula position, producing the
    "lash above iris" look the engine would render. Cached per
    (cat, idx)."""
    if not hasattr(_overlay_ink_offset_native, "_cache"):
        _overlay_ink_offset_native._cache = {}
    key = (cat, idx)
    if key in _overlay_ink_offset_native._cache:
        return _overlay_ink_offset_native._cache[key]
    p = _miiparts_path(cat, idx)
    out = (0.0, 0.0)
    if p and os.path.exists(p):
        arr = np.asarray(Image.open(p).convert("RGBA"))
        a = arr[..., 3].astype(int)
        r = arr[..., 0].astype(int)
        mask = (a > 64) & (r > 32)
        if mask.any():
            h, w = arr.shape[:2]
            ys, xs = np.where(mask)
            wts   = r[mask]
            cx    = float((xs * wts).sum() / wts.sum())
            cy    = float((ys * wts).sum() / wts.sum())
            out   = (cx - w / 2.0, cy - h / 2.0)
    _overlay_ink_offset_native._cache[key] = out
    return out


_hair_parts_pos_cache = {}


def _hair_pick_subtree(scene, want_flip):
    """Filter a HairAll / HairFront / HairBack scene to the visible subset.

    Each Hair*.glb bakes up to four sub-trees rooted at distinct nodes:
        Mii<Cat><NNN>            — base, no hat
        Mii<Cat>Hat<NNN>         — variant rendered with a hat on
        Mii<Cat><NNN>_Flip       — pre-mirrored (only on legacy entries
                                    whose bgyml ships FlippedModelUnit)
        Mii<Cat>Hat<NNN>_Flip    — flipped + hat
    Without filtering all four overlay each other in the rendered scene,
    which is why an L/R flip looks like a no-op for entries that ship a
    pre-baked Flip pair (the original and its mirror are both drawn).

    Always drop the Hat sub-tree (no hat in this renderer). When a flip
    is requested and a `_Flip` sub-tree exists, render that instead of
    the base; otherwise keep the base and tell the caller to apply a
    runtime X-mirror via `_mirror_x`.
    """
    roots = []
    for u, v, attrs in scene.graph.to_edgelist():
        if "geometry" in attrs:
            roots.append((u, attrs["geometry"]))
    has_baked_flip = any(r.endswith("_Flip") for r, _ in roots)
    keep = set()
    for r, g in roots:
        if "Hat" in r:
            continue
        is_flip = r.endswith("_Flip")
        if want_flip and has_baked_flip:
            if not is_flip: continue
        else:
            if is_flip: continue
        keep.add(g)
    out = trimesh.scene.Scene()
    for n, g in scene.geometry.items():
        if n in keep:
            out.add_geometry(g, geom_name=n)
    needs_runtime_mirror = bool(want_flip and not has_baked_flip)
    return out, needs_runtime_mirror

def _hair_parts_available_positions(label):
    """Set of slot names ('UpperLeft', 'MiddleCenter', …) authored in
    MiiHairParts<label>.glb, parsed from the scene-graph edges that carry
    a `geometry` attribute under a `MiiHairParts*_<Position>` root."""
    glb = _glb_path("HairParts", label)
    if not glb:
        return {}
    s = trimesh.load(glb, force="scene")
    out = {}
    for u, _, attrs in s.graph.to_edgelist():
        if "geometry" in attrs and u.startswith("MiiHairParts"):
            try:
                pos = u.split("_", 1)[1]
            except IndexError:
                continue
            out.setdefault(pos, attrs["geometry"])
    return out

def _hair_parts_row(label):
    """Return the attachment row ('Upper' / 'Middle' / 'Lower') for the
    HairParts label — derived from the explicit `@<X>` gallery suffix
    when present, otherwise from the part's bgyml EditorIconName."""
    rest = label.split(":", 1)[1] if ":" in label else label
    base, _, explicit = rest.partition("@")
    if explicit:
        return _HAIR_PARTS_ROW_BY_SUFFIX.get(explicit), base
    entry = mm.parts().get(f"HairParts{base}", {}) or {}
    icon  = entry.get("EditorIconName") or ""
    m = re.search(r"_([CUD])_Uit$", icon)
    return (_HAIR_PARTS_ROW_BY_SUFFIX.get(m.group(1)) if m else None), base

def _hair_parts_default_position(label, side_override=None):
    """Resolve HairParts<label> to its canonical slot name. `side_override`
    (one of 'Center' / 'Left' / 'Right' / 'LeftRight') wins over the
    Center→LeftRight→Left→Right priority when supplied. Cache only the
    no-override default — overrides get recomputed each call (cheap)."""
    if side_override is None and label in _hair_parts_pos_cache:
        return _hair_parts_pos_cache[label]
    pref_row, base = _hair_parts_row(label)
    avail = _hair_parts_available_positions(base)
    if not avail:
        if side_override is None:
            _hair_parts_pos_cache[label] = None
        return None
    sides = ([side_override] if side_override else []) + [
        s for s in _HAIR_PARTS_POS_PRIORITY_SIDES if s != side_override]
    rows  = ([pref_row] if pref_row else []) + [
        r for r in _HAIR_PARTS_POS_PRIORITY_ROWS if r != pref_row]
    chosen = None
    for r in rows:
        if r is None:
            continue
        for s in sides:
            if (r + s) in avail:
                chosen = r + s; break
        if chosen:
            break
    if chosen is None:
        chosen = next(iter(avail))
    if side_override is None:
        _hair_parts_pos_cache[label] = chosen
    return chosen

def _hair_parts_filter(scene, label, side_override=None):
    """Keep only the sub-mesh attached to the canonical slot for
    HairParts<label>; drop the other ~11 attachment variants."""
    pos = _hair_parts_default_position(label, side_override=side_override)
    if not pos:
        return scene
    keep = None
    for u, v, attrs in scene.graph.to_edgelist():
        if "geometry" in attrs and u.endswith(f"_{pos}"):
            keep = attrs["geometry"]; break
    if not keep or keep not in scene.geometry:
        return scene
    out = trimesh.scene.Scene()
    out.add_geometry(scene.geometry[keep], geom_name=keep)
    return out

# ── Main builder ──────────────────────────────────────────────────────────────
def build_face(*args):
    n3d  = len(PARTS)
    n2d  = len(FACE_PARTS)
    glb_sels  = args[:n3d]
    face_sels = args[n3d:n3d + n2d]
    base       = n3d + n2d
    # Skin / Glass / Eye / Mouth pickers store their selection as a
    # LINEAR (R, G, B) tuple straight from the swatch gallery — no
    # label/index lookup needed.
    skin_lin   = args[base + 0]
    glass_lin  = args[base + 1]
    # Face positioning sliders (Mii parameter values)
    eye_pos_y, eye_spacing_x, eye_scale  = args[base+2], args[base+3], args[base+4]
    brow_pos_y, brow_spacing_x, brow_scale = args[base+5], args[base+6], args[base+7]
    mouth_pos_y, mouth_scale             = args[base+8], args[base+9]
    nose_pos_y, nose_scale               = args[base+10], args[base+11]
    glass_pos_y, glass_scale             = args[base+12], args[base+13]
    ear_scale                            = args[base+14]
    # Eye / Mouth: linear (R,G,B) tuple. Hair / Eyebrow / Beard /
    # Mustache: int slot 0..99 into _NSO_HAIR_COLORS (the legacy state
    # type, kept for the larger 10×10 + favourites picker).
    eye_color_lin      = args[base+15]
    mouth_color_lin    = args[base+16]
    hair_color_lbl     = args[base+17]
    eyebrow_color_lbl  = args[base+18]
    beard3d_color_lbl  = args[base+19]
    beard2d_color_lbl  = args[base+20]
    mustache_color_lbl = args[base+21]
    expression_lbl     = args[base+22]
    mouth_drop_lips    = bool(args[base+23])
    eye_use_red        = bool(args[base+24])
    eyeline_color_lin  = args[base+25] if len(args) > base+25 else (1.0, 1.0, 1.0)
    hair_flip          = bool(args[base+26]) if len(args) > base+26 else False
    hair_parts_side    = args[base+27] if len(args) > base+27 else None
    hair_subcolor      = args[base+28] if len(args) > base+28 else 99   # white
    use_hair_subcolor  = bool(args[base+29]) if len(args) > base+29 else False
    if isinstance(hair_subcolor, str):
        try: hair_subcolor = int(hair_subcolor)
        except ValueError: hair_subcolor = 99
    hair_subcolor = max(0, min(99, int(hair_subcolor)))

    def _idx(val):
        if isinstance(val, int):
            return val if 0 <= val < len(_NSO_HAIR_COLORS) else 0
        if val in _NSO_HAIR_LABELS:
            return _NSO_HAIR_LABELS.index(val)
        try:
            return int(val)
        except Exception:
            return 0
    hair_color    = _idx(hair_color_lbl)
    eyebrow_color = _idx(eyebrow_color_lbl)
    beard3d_color = _idx(beard3d_color_lbl)
    beard2d_color = _idx(beard2d_color_lbl)
    mustache_color= _idx(mustache_color_lbl)

    # Toon-shader dampening only applies to HAIR / 3D-BEARD meshes (the
    # ones that go through baseColorFactor + KHR_materials_unlit). Skin
    # and glass are rendered at full albedo because they pass through
    # PBR with the head's normal lighting; dampening them made the face
    # appear ~5× too dark. Eye/Mouth are 2D-decal tints applied to the
    # face texture and aren't dampened either (the original rendering
    # path used full-strength linear values from CommonColorTable).
    # Earlier (pre-strand-texture) we dampened hair tints by 0.2 to
    # compensate for the IBL over-brightening; with KHR_materials_unlit
    # + the strand baseColorTexture (R/255 from `MiiHair*_Mim.png` —
    # mean ~0.85, so a mild attenuation of its own) the rendered pixel
    # is `sRGB(tint) * R_strand`, which already lands close to the
    # in-game brightness. Dampen × 0.2 on top dropped most fav-strip
    # colours below the per-channel floor (0.04) and they all read as
    # the same dark-grey blob. Drop the dampen, keep only a tiny floor
    # so pure-black hair still picks up the strand modulation.
    _HAIR_PURE_BLACK_FLOOR = 0.02

    def _dampen(lin):
        f = _HAIR_PURE_BLACK_FLOOR
        r, g, b = lin[:3]
        return [max(float(r), f),
                max(float(g), f),
                max(float(b), f), 1.0]
    def _passthrough_rgba(lin):
        r, g, b = lin[:3]
        return [float(r), float(g), float(b), 1.0]

    skin_rgba    = _passthrough_rgba(skin_lin)
    glass_rgba   = _passthrough_rgba(glass_lin)
    hair_rgba    = _dampen(_NSO_HAIR_COLORS[hair_color])
    hair_subcolor_rgba = _dampen(_NSO_HAIR_COLORS[hair_subcolor])
    beard3d_rgba = _dampen(_NSO_HAIR_COLORS[beard3d_color])

    face_img = compose_face_texture(
        face_sels,
        eye_pos_y=eye_pos_y, eye_spacing_x=eye_spacing_x, eye_scale=eye_scale,
        brow_pos_y=brow_pos_y, brow_spacing_x=brow_spacing_x, brow_scale=brow_scale,
        mouth_pos_y=mouth_pos_y, mouth_scale=mouth_scale,
        eye_color=eye_color_lin, mouth_color=mouth_color_lin, hair_color=hair_color,
        expression=expression_lbl,
        mouth_drop_lips=mouth_drop_lips,
        eye_use_red=eye_use_red,
        eyeline_color=eyeline_color_lin,
        eyebrow_color=eyebrow_color, beard2d_color=beard2d_color,
        mustache_color=mustache_color,
    )

    sockets = {}
    socket_rotations = {}
    scenes  = []

    # The HairBack gallery is a unified picker that surfaces both bare
    # HairBack tiles AND HairBack+HairParts compounds. When the user
    # picks a compound `<HHH>:<PPP>@<X>` tile, copy that label into the
    # HairParts slot so the HairParts iteration loads / row-filters the
    # corresponding parts mesh. _glb_path strips the prefix from the
    # HairBack sel so its own iteration still loads HairBackHHH.glb.
    glb_sels = list(glb_sels)
    _hp_idx = next((i for i, (c, *_) in enumerate(PARTS) if c == "HairParts"), None)
    _hb_idx = next((i for i, (c, *_) in enumerate(PARTS) if c == "HairBack"),  None)
    if _hp_idx is not None and _hb_idx is not None:
        hb_sel = glb_sels[_hb_idx]
        if isinstance(hb_sel, str) and ":" in hb_sel:
            glb_sels[_hp_idx] = hb_sel
        elif isinstance(hb_sel, str) and hb_sel != "(none)":
            # Bare HairBack → make sure HairParts isn't carrying a
            # stale compound from a previous render.
            if isinstance(glb_sels[_hp_idx], str) and ":" in glb_sels[_hp_idx]:
                glb_sels[_hp_idx] = "(none)"

    for (cat, _, required, _), sel in zip(PARTS, glb_sels):
        path = _glb_path(cat, sel)
        if path is None: continue
        try:
            s = trimesh.load(path, force="scene")
            s = _filter(s)
            if cat == "HairParts":
                s = _hair_parts_filter(s, sel, side_override=hair_parts_side)
            if not s.geometry: continue

            if cat == "Nose":
                # 3D nose GLBs (Nose01/03/04/05) ship a 4-vertex flat
                # `NoseLine` quad alongside the volumetric `Nose__mt_Nose`
                # mesh; 2D-only noses (Nose00/02) have just the quad. In
                # the actual game the engine textures it with the noseline
                # silhouette via the FMDB material, but our extraction
                # only ships a default gray PBR material — so the original
                # quad reads as an opaque gray plane intersecting the 3D
                # nose. Strip the shipped quad entirely, then rebuild a
                # fresh one in its place using the editor icon for the
                # selected nose (already a black silhouette with alpha)
                # as the texture in alphaMode=BLEND.
                icon_path = os.path.join(
                    _EDITOR_ICONS,
                    f"MiiEditorIcon_MiiEditor_Face_Nose{sel}_Uit.png")
                # Pull the original quad's vertex/face data so the
                # replacement sits at exactly the same position/size.
                orig_quads = [(n, g.copy()) for n, g in s.geometry.items()
                              if n.startswith("NoseLine__")
                              and isinstance(g, trimesh.Trimesh)]
                # Drop ALL NoseLine geometry from the scene.
                kept = trimesh.scene.Scene()
                for n, g in s.geometry.items():
                    if not n.startswith("NoseLine__"):
                        kept.add_geometry(g, geom_name=n)
                s = kept
                # Re-add a fresh textured quad if we have an icon to apply.
                if os.path.isfile(icon_path) and orig_quads:
                    icon_img = Image.open(icon_path).convert("RGBA")
                    # The 150×150 editor icon is mostly empty padding —
                    # the actual silhouette occupies only ~18% × ~40% of
                    # the canvas. Tight-crop the icon to the silhouette's
                    # alpha bounding box so it fills the quad at full
                    # resolution; aspect (~1:2.2) matches the quad's
                    # (~1:2.2), so no stretching either.
                    a = np.asarray(icon_img)[..., 3]
                    ys, xs = np.where(a > 0)
                    if len(xs):
                        pad = 2  # tiny breathing room
                        x0 = max(0, int(xs.min()) - pad)
                        x1 = min(icon_img.width,  int(xs.max()) + 1 + pad)
                        y0 = max(0, int(ys.min()) - pad)
                        y1 = min(icon_img.height, int(ys.max()) + 1 + pad)
                        icon_img = icon_img.crop((x0, y0, x1, y1))
                    for n, orig in orig_quads:
                        new_quad = trimesh.Trimesh(
                            vertices=orig.vertices.copy(),
                            faces=orig.faces.copy(),
                            process=False)
                        # Derive UVs per-vertex from each vertex's XY
                        # position within the quad's XY bbox (u=normalized
                        # x left→0/right→1, v=normalized y bottom→0/top→1).
                        # Hardcoding a UV array assuming a specific vertex
                        # order works for Nose01 but rotates 90° for noses
                        # whose FMDB stores vertices in a different order
                        # (Nose02/05/21/26/30 use y-major order vs Nose01's
                        # x-major). Computing UVs from positions makes the
                        # mapping robust to any vertex ordering.
                        v_xy = orig.vertices[:, :2]
                        x0, y0 = v_xy.min(axis=0)
                        x1, y1 = v_xy.max(axis=0)
                        dx = max(x1 - x0, 1e-9)
                        dy = max(y1 - y0, 1e-9)
                        u  = (v_xy[:, 0] - x0) / dx
                        v  = (v_xy[:, 1] - y0) / dy
                        corner_uv = np.column_stack([u, v]).astype(np.float32)
                        new_quad.visual = trimesh.visual.TextureVisuals(
                            uv=corner_uv,
                            material=PBRMaterial(
                                baseColorTexture=icon_img,
                                alphaMode="BLEND",
                                doubleSided=True,
                            ),
                        )
                        s.add_geometry(new_quad, geom_name=n + "_line")

            if cat == "Head":
                sockets = _get_sockets(path)
                socket_rotations = _get_socket_rotations(path)
                if face_img is not None:
                    s = _apply_face_texture_to_scene(s, face_img, skin_rgba)
                else:
                    s = _colorize(s, skin_rgba)
            elif cat in SKIN_CATS:
                s = _colorize(s, skin_rgba)
            elif cat in HAIR_CATS:
                # Drop the Hat sub-tree (and the pre-baked _Flip pair
                # when not flipping) so a single hairstyle renders
                # without overlap. For HairAllLegacy with the Flip
                # toggle on, prefer the bgyml-baked _Flip subtree;
                # fall back to a runtime X mirror when the entry has
                # IsFlippable=True but no baked flipped sub-tree.
                want_flip = False
                if hair_flip and cat in ("HairAllLegacy", "HairFront", "HairBack"):
                    bgyml_name = None
                    if cat == "HairAllLegacy" and isinstance(sel, str) \
                            and sel[:2] in ("L:", "N:"):
                        kind, sub = sel.split(":", 1)
                        bgyml_name = (f"HairAllLegacy{sub}" if kind == "L"
                                      else f"HairAll{sub}")
                    elif cat == "HairFront" and isinstance(sel, str) \
                            and sel != "(none)":
                        bgyml_name = f"HairFront{sel}"
                    elif cat == "HairBack" and isinstance(sel, str) \
                            and sel != "(none)":
                        bgyml_name = f"HairBack{sel.split(':', 1)[0]}"
                    if bgyml_name and mm.parts().get(bgyml_name, {}).get("IsFlippable"):
                        want_flip = True
                s, needs_mirror = _hair_pick_subtree(s, want_flip)
                if not s.geometry:
                    continue
                s = _colorize(s, hair_rgba)
                _apply_hair_strand_texture(s, path,
                                           hair_rgba[:3], hair_subcolor_rgba[:3],
                                           use_hair_subcolor)
                if needs_mirror:
                    s = _mirror_x(s)
            elif cat in BEARD3D_CATS:
                s = _colorize(s, beard3d_rgba)
                _apply_hair_strand_texture(s, path,
                                           beard3d_rgba[:3], beard3d_rgba[:3],
                                           False)
            elif cat in GLASS_CATS:
                s = _colorize_glass(s, glass_rgba, glass_label=sel)

            if cat == "Nose" and "nose" in sockets:
                # Translate by sockets["nose"] directly — no fudge factor.
                # The nose GLB authors the 3D mesh and the NoseLine quad
                # together in mesh-local space (quad at z≈0, mesh extending
                # back into negative z and slightly forward). Let the
                # FMDB's authored geometry decide protrusion: the quad
                # lands at sockets["nose"] (front of face), the 3D mesh
                # extends behind/in-front by its authored mesh_z range.
                # User-tweakable scale + Y offset come from the Nose tab
                # sliders. Scale step ≈ 5%/unit (range -6..6 → 70%..130%
                # — matches the nose-size knob in the in-game editor).
                # Y offset is applied in mesh-local units; ~1.5 cm/unit
                # is enough to nudge the nose up onto the bridge or down
                # toward the mouth without leaving the face.
                if nose_scale and float(nose_scale) != 0.0:
                    factor = 1.0 + 0.05 * float(nose_scale)
                    s = _apply_matrix(s, np.diag([factor, factor, factor, 1.0]))
                _NOSE_Y_PER_UNIT = 0.015
                nx, ny, nz = sockets["nose"]
                ny_off = ny - float(nose_pos_y) * _NOSE_Y_PER_UNIT
                s = _translate(s, [nx, ny_off, nz])
            elif cat == "Ear" and "ear" in sockets:
                ear = sockets["ear"]
                # User ear scale (range -6..6, step 0.5, 5%/unit so full
                # range covers ~70%..130%). Applied around the mesh
                # origin BEFORE the rotation/translation so the ear
                # grows in place at the head's `set_ear` socket instead
                # of drifting away from it.
                if ear_scale and float(ear_scale) != 0.0:
                    f = 1.0 + 0.05 * float(ear_scale)
                    s = _apply_matrix(s, np.diag([f, f, f, 1.0]))
                # Ear GLBs from the in-game FMDB are extracted with the
                # ear's main axis along ±Z and the up-vector along -Y.
                # Apply Rx(180) → Ry(90) → Rx(180) to right the up-vector
                # and rotate the main axis from +Z to +X (horizontal,
                # rolled 180°). Then apply the head GLB's set_ear bone
                # rotation (~33° tilt around an axis in the head's
                # forward/up plane) — this is the engine-authored
                # orientation that gives the ear its real-world tilt
                # (slightly forward and down) so it doesn't point
                # straight back.
                s = _rot_x_180(_rot_y_90(_rot_x_180(s)))
                ear_q = socket_rotations.get("ear", np.array([0., 0., 0., 1.]))
                R_ear = _quat_to_mat4(ear_q)
                # X-mirror copy needs the rotation conjugated by the X-mirror
                # so the geometry tilts the same way visually on the other side.
                Mx = np.eye(4); Mx[0, 0] = -1
                R_ear_mirror = Mx @ R_ear @ Mx
                s_oriented = _apply_matrix(s, R_ear)
                s_oriented_mirror = _apply_matrix(_mirror_x(s), R_ear_mirror)
                scenes.append(_translate(s_oriented, ear))
                scenes.append(_translate(s_oriented_mirror, [-ear[0], ear[1], ear[2]]))
                continue
            elif cat == "Glass" and "nose" in sockets:
                nose = sockets["nose"]
                # User-tweakable scale around the mesh origin (5%/unit)
                # and Y offset (~1.5 cm/unit) on top of the nose-anchor
                # placement. Range -6..6 covers ~70%..130% size and
                # ±9 cm vertical nudge.
                if glass_scale and float(glass_scale) != 0.0:
                    f = 1.0 + 0.05 * float(glass_scale)
                    s = _apply_matrix(s, np.diag([f, f, f, 1.0]))
                y_off = nose[1] + 0.07 - float(glass_pos_y) * 0.015
                s = _translate(s, [0.0, y_off, nose[2]])

            # Hair / beard meshes are authored against MiiHead00's
            # set_hair / set_beard sockets at world origin (0, 0, 0).
            # Other faceline shapes shift these sockets — Head11 e.g.
            # raises set_hair by +0.030 — and without applying the
            # offset the hair mesh sits at Head00's height, leaving
            # the top of a taller head mesh exposed. Translate by
            # the per-head socket offset so the hair attaches at the
            # current head's anchor.
            if cat in HAIR_CATS and "hair" in sockets:
                s = _translate(s, sockets["hair"])
            elif cat in BEARD3D_CATS and "beard" in sockets:
                s = _translate(s, sockets["beard"])

            scenes.append(s)
        except Exception as e:
            print(f"[WARN] {cat}: {e}")

    if not scenes: return None
    combined = append_scenes(scenes)
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    combined.export(tmp.name)
    tmp.close()
    _patch_glb_unlit_hair(tmp.name)
    return _model_viewer_html(tmp.name)


def _patch_glb_unlit_hair(path):
    """Mark hair / 3D-beard meshes as `KHR_materials_unlit` in the exported
    GLB. model-viewer's PBR + neutral IBL + exposure 2.5 over-amplifies
    the dark hair albedos by ~3× vs. the in-game look — even after the
    toon-derived 0.5625 dampening factor (from MainLightIntensity 0.75 ×
    DiffuseShadeStartIntensity 0.75) is pre-applied, the IBL contribution
    just adds the brightness back. Unlit cuts IBL out entirely: the
    rendered pixel = sRGB(baseColorFactor), exactly matching the linear
    color picked from CommonColorTable."""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"glTF":
            return
        json_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)
        json_bytes = f.read(json_len)
        rest = f.read()
    j = json.loads(json_bytes)
    meshes = j.get("meshes", [])
    materials = j.get("materials", [])
    # Find primitives whose mesh name suggests hair / 3D-beard, mark each
    # referenced material as unlit.
    target_mat_idx = set()
    for mesh in meshes:
        name = mesh.get("name", "")
        if "Hair" in name or "Beard" in name or "UnderCut" in name:
            for prim in mesh.get("primitives", []):
                mi = prim.get("material")
                if mi is not None:
                    target_mat_idx.add(mi)
    if not target_mat_idx:
        return
    patched = False
    for mi in target_mat_idx:
        if not 0 <= mi < len(materials):
            continue
        m = materials[mi]
        ext = m.setdefault("extensions", {})
        if "KHR_materials_unlit" in ext:
            continue
        ext["KHR_materials_unlit"] = {}
        patched = True
    if not patched:
        return
    used = j.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    new_json = json.dumps(j, separators=(",", ":")).encode()
    pad = (-len(new_json)) % 4
    new_json += b" " * pad
    new_total = 12 + 8 + len(new_json) + 8 + len(rest) - 8
    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, new_total)
    out += struct.pack("<II", len(new_json), 0x4E4F534A) + new_json
    out += rest
    with open(path, "wb") as f:
        f.write(out)


# ── Mii editor lighting (decoded from EditorMiiPreview snapshot setting) ──────
# Source bgyml: Parameter/GfxSnapshotSetting/EditorMiiPreview.gfx__SnapshotSetting
#               (inherits Default.gfx__SnapshotSetting)
# Composed values (with EditorMiiPreview overriding Default):
#   LightLatitude:  40.0      (deg above horizon)
#   LightLongitude: -20.0     (deg azimuth — slight right of camera)
#   LightColor:     (1, 1, 1) (pure white)
#   LightIntensity: 2.5       (overrides Default 6.0)
#   ShadowIntensity:1.3
#   SpecularIntensity: 3.5
#   ProjFovy:       15        (deg — narrow telephoto, very tight zoom)
#   CameraAt:       (0, 0.97, 0)
#   CameraPos:      (0, 0.96, 9.4) → orbit-radius ≈ 9.4 m
#   IsPfxBloom:     true (BloomIntensity 0.7, BloomThreshold/Clamp 1.5)
#   IsPfxToneMapping: false (no tone mapping in editor preview)
# Toon-shading params (Parameter/GfxDynamicLightToon/Island.gfx__…):
#   DiffuseIntensity 2.7, DiffuseLumaThreshold 0.5
#   SpecularIntensity 4.0, SpecularLumaThreshold 0.2
#   MainLightIntensity 0.75
# These are SHADER-level constants for the engine's cel-shader; model-viewer
# doesn't expose a custom toon-shader hook, so we map only the non-shader
# pieces (exposure, shadow intensity, camera FOV/orbit) — they're the
# observable difference vs. model-viewer's default neutral environment.
# LightIntensity 2.5 from the in-game bgyml lives inside a toon-shader
# pipeline whose unit semantics don't transfer to model-viewer's PBR. With
# `exposure=2.5` we were rendering the whole scene ~2.5× too bright. Stick
# to model-viewer's default exposure (1.0); skin reads at its natural PBR
# brightness, and unlit hair reads as sRGB_encode(baseColorFactor) — i.e.
# the binary's sRGB row exactly. ShadowIntensity / FOV / orbit are
# rendering-equivalent and stay engine-derived.
_NSO_LIGHT_INTENSITY    = 1.0
_NSO_SHADOW_INTENSITY   = 1.3
_NSO_CAMERA_FOVY_DEG    = 15.0
_NSO_CAMERA_RADIUS_M    = 9.4
_NSO_LIGHT_LATITUDE     = 40.0   # deg above horizon
_NSO_LIGHT_LONGITUDE    = -20.0  # deg azimuth (negative = right of camera)


def _model_viewer_html(glb_path):
    """Backwards-compat alias — delegates to the Three.js toon viewer."""
    return _three_toon_html(glb_path)


_THREE_TOON_TEMPLATE_PATH = os.path.join(_ASSETS, "three_toon_viewer.html")


def _three_toon_html(glb_path):
    """Render the Mii via a Three.js scene with a custom toon shader matching
    the in-game cel-shading parameters (Parameter/GfxToon/System.gfx__ToonParam
    + Parameter/GfxDynamicLightToon/Island.gfx__DynamicLightToonParam +
    Parameter/GfxSnapshotSetting/EditorMiiPreview.gfx__SnapshotSetting).

    Replaces the previous model-viewer (PBR) renderer with bgyml-faithful
    cel-shading — including the face-specific 100° ShadeStartAngle (vs body
    85°), the 5.0× back-edge rim, the 4.0× specular with 0.4 albedo tint,
    and bloom post-processing. The HTML lives in
    `assets/three_toon_viewer.html` and is loaded into an `<iframe srcdoc>`
    so its importmap+modules don't collide with Gradio's DOM."""
    st = os.stat(glb_path)
    bust = f"{int(st.st_mtime * 1000)}-{st.st_size}"
    glb_url = f"/gradio_api/file={glb_path}?v={bust}"
    with open(_THREE_TOON_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    # Inject the GLB URL by inserting a window assignment BEFORE the module
    # script. Use a unique marker that the template defines for us.
    inject = f'<script>window.MII_GLB_URL = "{glb_url}";</script>'
    # Place injection immediately after <body>
    if "<body>" in template:
        body = template.replace("<body>", "<body>\n" + inject, 1)
    else:
        body = inject + template
    # html.escape the entire document to embed via srcdoc (which expects a
    # string with double-quotes and angle brackets escaped).
    escaped = (body.replace("&", "&amp;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))
    return (
        '<iframe srcdoc="' + escaped + '" '
        'style="width:100%;height:720px;border:none;background:#F5E9C8" '
        'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def _patch_glb_unlit_lens(path):  # legacy — left for reference, not invoked
    """Re-write a GLB so the transparent-lens material uses the canonical
    glTF glass extension `KHR_materials_transmission`. Babylon.js Viewer v2
    (Gradio Model3D → @babylonjs/viewer) supports it natively and renders
    it as a true glass surface — light transmits through with the IBL
    environment unchanged.

    `alphaMode=BLEND` alone brightens the lens via the PBR/IBL pipeline
    until the small alpha is invisible, and `KHR_materials_unlit` flattens
    the colour but the lens still occluded the eyes behind it in our
    tests. With `transmissionFactor` the lens stays opaque to the
    rasteriser (so render order is irrelevant) but transmits the
    background colour through.

    trimesh's PBRMaterial doesn't expose extensions, so post-process the
    JSON chunk: find the BLEND material, swap alphaMode→OPAQUE, set
    baseColorFactor.a→1.0 (transmission handles the see-through), add
    `KHR_materials_transmission: {transmissionFactor: 0.95}`, register
    the extension in `extensionsUsed`."""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"glTF":
            return
        json_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)
        json_bytes = f.read(json_len)
        rest = f.read()
    j = json.loads(json_bytes)
    mats = j.get("materials", [])
    patched = False
    for m in mats:
        if m.get("alphaMode") != "BLEND":
            continue
        # Restore baseColorFactor alpha to 1 (transmission handles the
        # see-through; alphaMode/alpha would just confuse the renderer).
        pbr = m.setdefault("pbrMetallicRoughness", {})
        bcf = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        if len(bcf) == 4:
            bcf[3] = 1.0
            pbr["baseColorFactor"] = bcf
        m["alphaMode"] = "OPAQUE"
        ext = m.setdefault("extensions", {})
        ext["KHR_materials_transmission"] = {
            "transmissionFactor": 1.0 - _NSO_GLASS_LENS_OPACITY
        }
        patched = True
    if not patched:
        return
    used = j.setdefault("extensionsUsed", [])
    if "KHR_materials_transmission" not in used:
        used.append("KHR_materials_transmission")
    new_json = json.dumps(j, separators=(",", ":")).encode("utf-8")
    pad = (-len(new_json)) % 4
    new_json += b" " * pad
    out = bytearray()
    out += b"glTF"
    out += struct.pack("<I", 2)
    out += struct.pack("<I", 0)        # placeholder for total length
    out += struct.pack("<I", len(new_json))
    out += b"JSON"
    out += new_json
    out += rest
    out[8:12] = struct.pack("<I", len(out))
    with open(path, "wb") as f:
        f.write(bytes(out))

# ── UI ────────────────────────────────────────────────────────────────────────
_HEAD = (
    '<script type="module" '
    'src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>'
)

# Build CSS rules that swap each top-level Tab's text label for the
# in-game IconCat PNG (extracted from MiiEditor_EditorSpace_00.blarc).
# Gradio 6 mirrors `elem_id="mii-tab-Face"` onto the tab BUTTON as
# `mii-tab-Face-button` (per gradio.layouts.tabs.Tab.__init__ docstring),
# which gives us a stable target. PNGs are embedded as base64 data URIs
# so Gradio's static-file allowlist isn't part of the loop.
import base64 as _b64
_TAB_ICONS_DIR = os.path.join(_ASSETS, "tab_icons")
_TAB_ICON_NAMES = ("Face", "Hair", "Eyebrow", "Eye", "Nose",
                   "Mouth", "Ear", "Grasses", "FaceDeco")
def _tab_icon_data_uri(name):
    p = os.path.join(_TAB_ICONS_DIR, f"__Combined_IconCat{name}^s.png")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + _b64.b64encode(f.read()).decode("ascii")
_TAB_ICON_CSS = "\n".join(
    f"#mii-tab-{n}-button {{"
    f" background-image: url({uri});"
    f" background-repeat: no-repeat;"
    f" background-position: center;"
    f" background-size: 28px 28px;"
    f" color: transparent !important;"
    f" min-width: 56px;"
    f" min-height: 44px;"
    f" padding-left: 8px !important;"
    f" padding-right: 8px !important;"
    f"}}"
    for n in _TAB_ICON_NAMES
    for uri in (_tab_icon_data_uri(n),) if uri
)
_PALETTE_CSS = """
/* DOM (Gradio 6 Gallery, observed via playwright):
     .block.mii-palette
       .gallery-container
         .grid-wrap (style="height: 328px;")
           .grid-container (style="--grid-cols:10; --grid-rows:10;")
             .gallery-item × 100
               button.thumbnail-item.thumbnail-lg
                 img
   Gradio's default grid CSS uses minmax(0, 1fr) per cell which lets
   cells balloon to 80 px when the parent is wider than expected. We
   override the grid-template explicitly to "repeat(10, 30px)" so the
   layout is force-locked to 22-px tiles regardless of parent width. */
.mii-palette {
    width: 328px !important;
    flex: 0 0 328px !important;
    height: auto !important;
    min-width: 0 !important;
    # overflow: visible !important;
}
.mii-palette .gallery-container,
.mii-palette .grid-wrap {
    width: auto !important; height: auto !important;
    padding: 0 !important; margin: 0 !important;
}
.mii-palette .grid-container {
    display: grid !important;
    grid-template-columns: repeat(10, 30px) !important;
    grid-template-rows:    repeat(10, 30px) !important;
    grid-auto-rows: 30px !important;
    gap: 1px !important; row-gap: 1px !important; column-gap: 1px !important;
    padding: 0 !important; margin: 0 !important;
    width: max-content !important;
}
.mii-palette .gallery-item {
    width: 30px !important; height: 30px !important;
    min-width: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important;
    aspect-ratio: auto !important;
}
.mii-palette button.thumbnail-item {
    width: 30px !important; height: 30px !important;
    min-width: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important;
    border: 0 !important; border-radius: 2px !important;
    aspect-ratio: auto !important;
    overflow: hidden !important;
}
.mii-palette img {
    width: 30px !important; height: 30px !important;
    object-fit: cover !important;
    display: block !important;
}
/* Favourites strip: 1 column × 8 rows of the legacy HairColorOrder
   subset. Same tile size, vertical layout, sits to the left of the
   main 10×10 palette. */
.mii-palette-fav {
    width: 48px !important; max-width: 48px !important;
    flex: 0 0 48px !important;
    height: auto !important; min-width: 0 !important; overflow: visible !important;
}
.mii-palette-fav .gallery-container,
.mii-palette-fav .grid-wrap {
    width: auto !important; height: auto !important;
    padding: 0 !important; margin: 0 !important;
}
.mii-palette-fav .grid-container {
    display: grid !important;
    grid-template-columns: 30px !important;
    grid-template-rows:    repeat(8, 30px) !important;
    grid-auto-rows: 30px !important;
    gap: 1px !important; row-gap: 1px !important; column-gap: 1px !important;
    padding: 0 !important; margin: 0 !important;
    width: max-content !important;
}
.mii-palette-fav .gallery-item,
.mii-palette-fav button.thumbnail-item {
    width: 30px !important; height: 30px !important;
    min-width: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important;
    border: 0 !important; border-radius: 2px !important;
    aspect-ratio: auto !important;
    overflow: hidden !important;
}
.mii-palette-fav img {
    width: 30px !important; height: 30px !important;
    object-fit: cover !important;
    display: block !important;
}
/* Single-row palette (skin / eye / mouth / glass) — N columns × 1 row. */
.mii-palette-row {
    width: auto !important; max-width: 100% !important;
    flex: 0 0 auto !important;
    height: auto !important; min-width: 0 !important; overflow: visible !important;
}
.mii-palette-row .gallery-container,
.mii-palette-row .grid-wrap {
    width: auto !important; height: auto !important;
    padding: 0 !important; margin: 0 !important;
}
.mii-palette-row .grid-container {
    display: grid !important;
    grid-auto-flow: column !important;
    grid-auto-columns: 30px !important;
    grid-template-rows: 30px !important;
    gap: 1px !important; row-gap: 1px !important; column-gap: 1px !important;
    padding: 0 !important; margin: 0 !important;
    width: max-content !important;
}
.mii-palette-row .gallery-item,
.mii-palette-row button.thumbnail-item {
    width: 30px !important; height: 30px !important;
    min-width: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important;
    border: 0 !important; border-radius: 2px !important;
    aspect-ratio: auto !important;
    overflow: hidden !important;
}
.mii-palette-row img {
    width: 30px !important; height: 30px !important;
    object-fit: cover !important;
    display: block !important;
}
"""
with gr.Blocks(title="Tomodachi Life — 3D Face Builder") as demo:
    gr.Markdown("## Tomodachi Life — 3D Face Builder")

    # Expression dropdown is built under the viewer (left column); declared
    # here so the parts/positioning column below can still reference it.
    expression_dd = None
    with gr.Row():
        with gr.Column(scale=1):
            viewer = gr.HTML(value="", label="3D Preview", show_label=False)
            gr.Markdown("### Expression  *(MiiExpression.pack overrides)*")
            _expr_choices = ["Normal"] + sorted(mm.expressions().keys())
            expression_dd = gr.Dropdown(choices=_expr_choices, value="Normal",
                                        label="Expression",
                                        info="Picks an emotion preset that swaps eye/brow/mouth and applies PartsLocation rotate/offset/scale deltas.",
                                        interactive=True)
        with gr.Column(scale=3):
            # ── Parts: top-level tabs grouped by editor category, with
            # one subtab per sub-category (no subtabs when the group has
            # only one entry). Default selections so the demo loads with
            # a complete face: first real entry per category for the
            # listed prefixes.
            _DEFAULT_FACE = {"Eye", "Eyebrow", "Mouth"}
            _PARTS_3D = {cat: (label, required, default_idx)
                         for cat, label, required, default_idx in PARTS}
            _PARTS_2D = {prefix: label for prefix, label in FACE_PARTS}
            # (group_icon_name, group_display_label, [(kind, key), ...])
            # Tab order (per request): Face → Hair → Eyebrow → Eyes →
            # Nose → Mouth → Ears → Glasses → Deco.
            PART_GROUPS = [
                ("Face",      "Face",     [("3D", "Head")]),
                ("Hair",      "Hair",     [("3D", "HairAllLegacy"),
                                           ("3D", "HairFront"), ("3D", "HairBack")]),
                ("Eyebrow",   "Eyebrow",  [("2D", "Eyebrow")]),
                ("Eye",       "Eyes",     [("2D", "Eye"),
                                           ("2D", "EyelashUpper"), ("2D", "EyelashLower"),
                                           ("2D", "EyelidUpper"),  ("2D", "EyelidLower"),
                                           ("2D", "Highlight")]),
                ("Nose",      "Nose",     [("3D", "Nose")]),
                ("Mouth",     "Mouth",    [("2D", "Mouth")]),
                ("Ear",       "Ears",     [("3D", "Ear")]),
                ("Grasses",   "Glasses",  [("3D", "Glass")]),
                ("FaceDeco",  "Deco",     [("3D", "Beard"),    ("2D", "Beard"),
                                           ("2D", "Mustache"),
                                           ("2D", "Mole"),
                                           ("2D", "MakeUpper"),    ("2D", "MakeLower"),
                                           ("2D", "WrinkleUpper"), ("2D", "WrinkleLower")]),
            ]

            glb_state_by_cat   = {}
            glb_gallery_by_cat = {}   # cat -> (gallery, state, sel_md, opts)
            face_state_by_pref = {}
            glb_gallery_info   = []
            face_gallery_info  = []

            def _build_3d(cat):
                label, required, default_idx = _PARTS_3D[cat]
                opts = _glb_choices(cat)
                if required and len(opts) > 1:
                    # No '(none)' tile for required categories (e.g. Head):
                    # the head mesh must always be present for any other
                    # part to be placed.
                    opts = opts[1:]
                    default = (opts[default_idx]
                               if default_idx is not None and default_idx < len(opts)
                               else opts[0])
                else:
                    default = (opts[default_idx + 1]
                               if default_idx is not None and len(opts) > 1 else "(none)")
                # The unified Hair gallery (HairAllLegacy) reorders by
                # the Hair PartsOrder bgyml, so the legacy default
                # `Hair012` lands at an arbitrary index instead of
                # opts[13]. Force the by-name default here.
                if cat == "HairAllLegacy" and "L:012" in opts:
                    default = "L:012"
                # HairFront / HairBack: hide the `(none)` tile from the
                # gallery — the unified-Hair "All" tab already covers
                # the no-front-or-back case, and explicit Front/Back
                # picks are mutually exclusive with the All gallery.
                # Keep `(none)` in `opts` so the state can still hold it
                # (mutual-exclusion handlers reset it to `(none)` when
                # the All gallery picks something).
                if cat in ("HairFront", "HairBack"):
                    display_opts = [o for o in opts if o != "(none)"]
                else:
                    display_opts = opts
                images = [_glb_icon_path(cat, lbl) for lbl in display_opts]
                default_idx_in_opts = (display_opts.index(default)
                                       if default in display_opts else None)
                state = gr.State(default)
                sel_md = gr.Markdown(
                    f"**{default}**" if default != "(none)" else "*(none)*")
                gallery = gr.Gallery(
                    value=images, label=None, show_label=False,
                    columns=5, rows=5, height=800,
                    allow_preview=False, object_fit="contain",
                    selected_index=default_idx_in_opts)
                glb_state_by_cat[cat] = state
                glb_gallery_by_cat[cat] = (gallery, state, sel_md, display_opts)
                glb_gallery_info.append((gallery, state, sel_md, display_opts))

            # Specific per-prefix defaults — falls back to the first non-(none)
            # entry for any prefix in `_DEFAULT_FACE` not listed here.
            _FACE_DEFAULT_NAME = {
                "Eye":     "Eye004",
                "Eyebrow": "Eyebrow00",
                "Mouth":   "Mouth023",
            }

            def _build_2d(prefix):
                label = _PARTS_2D[prefix]
                items = _gallery_items(prefix)
                names = [n for _, n in items]
                explicit = _FACE_DEFAULT_NAME.get(prefix)
                if explicit and explicit in names:
                    default_name = explicit
                elif prefix in _DEFAULT_FACE and len(names) > 1:
                    default_name = names[1]
                else:
                    default_name = "(none)"
                default_idx_in_names = names.index(default_name) if default_name in names else None
                state = gr.State(default_name)
                sel_md = gr.Markdown(
                    f"**{default_name}**" if default_name != "(none)" else "*(none)*")
                gallery = gr.Gallery(
                    value=[img for img, _ in items], label=None, show_label=False,
                    columns=5, rows=5, height=800,
                    allow_preview=False, object_fit="contain",
                    selected_index=default_idx_in_names)
                face_state_by_pref[prefix] = state
                face_gallery_info.append((gallery, state, sel_md, names))

            # Colour controls live under the tab they affect (set inside
            # the loop below). Captured into local names so build_face's
            # all_inputs assembly downstream can address them directly.
            skin_dd = glass_dd = None
            hair_color_state    = None  # gr.State (int slot 0..99)
            eyebrow_color_state = None
            beard3d_color_state = None
            beard2d_color_state = None
            mustache_color_state= None
            eye_color_dd = mouth_color_dd = None
            mouth_no_lip_cb = eye_use_red_cb = None
            eyeline_color_state = None

            _hair_swatches_natural = _hair_swatch_paths()  # 100 PNGs, slot order
            # Mirror along the ANTI-diagonal (top-right ↔ bottom-left)
            # for the 10×10 visual layout.
            #   visible (col=c, row=r) ↔ original (col=9-r, row=9-c)
            #   T(p) = 99 - 10·(p%10) - (p//10)
            # Self-inverse on a 10×10 (verified by substitution), so the
            # same function maps both directions.
            def _antidiag10(i):
                return 99 - 10 * (i % 10) - (i // 10)
            _hair_swatches = [_hair_swatches_natural[_antidiag10(p)]
                              for p in range(100)]
            # Legacy HairColorOrder = [8, 4, 5, 1, 2, 3, 6, 7] (CommonColor-
            # Table indices). Mapped to the new editor-display slots:
            #   table[8]=Black     → slot 90
            #   table[4]=Gray      → slot 95
            #   table[5]=Olive     → slot 60
            #   table[1]=DarkBrown → slot 80
            #   table[2]=RedBrown  → slot 0
            #   table[3]=Brown     → slot 81
            #   table[6]=LightBrn  → slot 72
            #   table[7]=Blonde    → slot 74
            _FAV_HAIR_SLOTS = [90, 95, 60, 80, 0, 81, 72, 74]
            # Index into the *natural* (slot-order) list — _hair_swatches
            # is the anti-diagonal-mirrored copy used by the 10×10 view,
            # which would put the wrong colours under the favourites tile.
            _fav_swatches = [_hair_swatches_natural[i] for i in _FAV_HAIR_SLOTS]

            def _color_picker(label_text, default_slot):
                """Renders a side-by-side 1-col 'favourites' strip (the
                legacy HairColorOrder 8 colours) + the full 10×10 palette.
                Both feed the same gr.State so the user can pick from
                either."""
                state = gr.State(int(default_slot))
                with gr.Row():
                    with gr.Column(scale=0, min_width=42):
                        gr.Markdown(f"**Fav**")
                        fav_gallery = gr.Gallery(
                            value=_fav_swatches, label=None, show_label=False,
                            columns=1, rows=8, height=255,
                            allow_preview=False, object_fit="cover",
                            elem_classes=["mii-palette-fav"])
                    with gr.Column(scale=0, min_width=320):
                        gr.Markdown(f"**{label_text}**")
                        gallery = gr.Gallery(
                            value=_hair_swatches, label=None, show_label=False,
                            columns=10, rows=10, height=255,
                            allow_preview=False, object_fit="cover",
                            selected_index=_antidiag10(int(default_slot)),
                            elem_classes=["mii-palette"])
                return gallery, fav_gallery, state

            # Pre-render swatches for the small palettes (skin/eye/mouth/glass).
            _skin_lin   = [v[:3] for v in SKIN_TABLE.values()]
            _glass_lin  = [v[:3] for v in GLASS_TABLE.values()]
            _skin_swatches  = _swatch_paths("skin",  _skin_lin)
            _glass_swatches = _swatch_paths("glass", _glass_lin)
            _eye_swatches   = _swatch_paths("eye",   _NSO_EYE_B_COLORS)
            _mouth_swatches = _swatch_paths("mouth", _NSO_MOUTH_R_COLORS)

            def _full_color_picker(label_text, fav_lin_colors, fav_swatches,
                                   default_lin, fav_value_fn=None,
                                   full_value_fn=None):
                """Picker with a 1-col 'favourites' strip (the curated
                game palette) + the full 10×10 CommonColorTable palette.
                Both write the picked LINEAR (R,G,B) tuple into a shared
                gr.State; build_face uses that tuple directly.

                `fav_value_fn(idx)` overrides what the favourites click
                writes — used for Mouth, where the curated palette has
                separate lower-lip (R) and upper-lip (G) colours per
                slot and we want to preserve that pair instead of
                collapsing to a single tuple."""
                state = gr.State(tuple(default_lin))
                if fav_value_fn is None:
                    _fav_capture = [tuple(c) for c in fav_lin_colors]
                    fav_value_fn = lambda i, _f=_fav_capture: _f[int(i)]
                with gr.Row():
                    with gr.Column(scale=0, min_width=42):
                        gr.Markdown(f"**Fav**")
                        _fav_h = max(60, 31 * len(fav_swatches))
                        fav_g = gr.Gallery(
                            value=fav_swatches, label=None, show_label=False,
                            columns=1, rows=len(fav_swatches),
                            height=_fav_h,
                            allow_preview=False, object_fit="cover",
                            elem_classes=["mii-palette-fav"])
                    with gr.Column(scale=0, min_width=320):
                        gr.Markdown(f"**{label_text}**")
                        full_g = gr.Gallery(
                            value=_hair_swatches, label=None, show_label=False,
                            columns=10, rows=10, height=_fav_h,
                            allow_preview=False, object_fit="cover",
                            elem_classes=["mii-palette"])
                def _on_fav(evt: gr.SelectData, _fn=fav_value_fn):
                    # Clear the full-palette highlight so the two
                    # galleries stay mutually exclusive.
                    return _fn(int(evt.index)), gr.update(selected_index=None)
                def _on_full(evt: gr.SelectData, _fn=full_value_fn):
                    p = int(evt.index)
                    slot = 99 - 10 * (p % 10) - (p // 10)
                    rgb = tuple(_NSO_HAIR_COLORS[slot])
                    val = _fn(rgb) if _fn is not None else rgb
                    return val, gr.update(selected_index=None)
                fav_g.select(_on_fav,  outputs=[state, full_g])
                full_g.select(_on_full, outputs=[state, fav_g])
                return full_g, fav_g, state

            with gr.Tabs():
                for icon_name, group_label, subs in PART_GROUPS:
                    with gr.Tab(group_label, elem_id=f"mii-tab-{icon_name}"):
                        # Decoration is the only group whose colour belongs
                        # to a SUB-feature (Beard 3D / Beard 2D / Mustache),
                        # so its [shape | colour] split lives at the sub-tab
                        # level. Every other tab has a per-tab colour that
                        # applies regardless of which sub the user is on
                        # (e.g. one Eye colour drives all eye-overlay subs).
                        if group_label == "Deco":
                            with gr.Tabs():
                                for kind, key in subs:
                                    sublabel = (_PARTS_3D[key][0] if kind == "3D"
                                                else _PARTS_2D[key])
                                    with gr.Tab(sublabel):
                                        with gr.Row():
                                            with gr.Column(scale=3):
                                                _build_3d(key) if kind == "3D" else _build_2d(key)
                                            with gr.Column(scale=2):
                                                if kind == "3D" and key == "Beard":
                                                    beard3d_color_gallery, beard3d_color_fav, beard3d_color_state = _color_picker(
                                                        "Beard 3D Color", default_slot=90)
                                                elif kind == "2D" and key == "Beard":
                                                    beard2d_color_gallery, beard2d_color_fav, beard2d_color_state = _color_picker(
                                                        "Beard 2D Color", default_slot=90)
                                                elif kind == "2D" and key == "Mustache":
                                                    mustache_color_gallery, mustache_color_fav, mustache_color_state = _color_picker(
                                                        "Mustache Color", default_slot=90)
                            continue
                        # Eyes: each subtab is self-contained — only the
                        # "Eye" subtab gets the eye-colour picker + position
                        # sliders; the lash/lid/highlight subtabs show only
                        # their shape gallery; the new "Eyeline" subtab has
                        # its own 2-tile style gallery + colour picker.
                        if group_label == "Eyes":
                            # Per-subtab controls always need to exist for
                            # all_inputs assembly; create them here so the
                            # right-column branches below can stay simple.
                            with gr.Tabs():
                                for kind, key in subs:
                                    sublabel = _PARTS_2D[key]
                                    with gr.Tab(sublabel):
                                        with gr.Row():
                                            with gr.Column(scale=3):
                                                _build_2d(key)
                                            with gr.Column(scale=2):
                                                if key == "Eye":
                                                    eye_full_g, eye_fav_g, eye_color_dd = _full_color_picker(
                                                        "Eye Color", _NSO_EYE_B_COLORS, _eye_swatches,
                                                        default_lin=_NSO_EYE_B_COLORS[0])
                                                    with gr.Row():
                                                        eye_y_sl  = gr.Slider(-5, 5, value=0,    step=0.5, label="Eye positionY (-5=up, 5=down)")
                                                        eye_x_sl  = gr.Slider(1,  3, value=1.5,  step=0.25, label="Eye spacingX (1=close to nose, 3=far)")
                                                        eye_sc_sl = gr.Slider(-2, 2, value=0,    step=0.5,  label="Eye scale")
                                with gr.Tab("Eyeline"):
                                    with gr.Row():
                                        with gr.Column(scale=3):
                                            _eyeline_icons = [
                                                os.path.join(_EDITOR_ICONS,
                                                    "MiiEditorIcon_MiiEditor_Face_EyelineNothing_Uit.png"),
                                                os.path.join(_EDITOR_ICONS,
                                                    "MiiEditorIcon_MiiEditor_Face_Eyeline00_Uit.png"),
                                            ]
                                            eye_use_red_cb = gr.State(False)
                                            gr.Markdown("**Style**")
                                            eyeline_gallery = gr.Gallery(
                                                value=_eyeline_icons, label=None, show_label=False,
                                                columns=5, rows=5, height=800,
                                                allow_preview=False, object_fit="contain",
                                                selected_index=0)
                                            def _on_eyeline_select(evt: gr.SelectData):
                                                return bool(int(evt.index))
                                            eyeline_gallery.select(
                                                _on_eyeline_select, outputs=eye_use_red_cb)
                                        with gr.Column(scale=2):
                                            eyeline_full_g, eyeline_fav_g, eyeline_color_state = _full_color_picker(
                                                "Eyeline Color", _NSO_EYE_B_COLORS, _eye_swatches,
                                                default_lin=(1.0, 1.0, 1.0))
                            continue
                        # All other tabs: shape (or shape sub-tabs) on the
                        # left, per-tab colour + extras on the right.
                        with gr.Row():
                            with gr.Column(scale=3):
                                if len(subs) == 1:
                                    kind, key = subs[0]
                                    _build_3d(key) if kind == "3D" else _build_2d(key)
                                else:
                                    with gr.Tabs():
                                        for kind, key in subs:
                                            sublabel = (_PARTS_3D[key][0] if kind == "3D"
                                                        else _PARTS_2D[key])
                                            with gr.Tab(sublabel):
                                                _build_3d(key) if kind == "3D" else _build_2d(key)
                                                # The Flip L/R toggle for
                                                # this gallery lives under
                                                # the Hair Color picker on
                                                # the right column; only
                                                # the state is created
                                                # here so it precedes the
                                                # button's click wiring.
                                                if key == "HairAllLegacy":
                                                    hair_flip_state       = gr.State(False)
                                                    hair_parts_side_state = gr.State("Center")
                            with gr.Column(scale=2):
                                if group_label == "Face":
                                    skin_full_g, skin_fav_g, skin_dd = _full_color_picker(
                                        "Skin Tone", _skin_lin, _skin_swatches,
                                        default_lin=SKIN_TABLE["Peach"][:3])
                                elif group_label == "Hair":
                                    hair_color_gallery, hair_color_fav, hair_color_state = _color_picker(
                                        "Hair Color", default_slot=90)
                                    # Sub-colour for the per-strand highlight
                                    # stripe — corresponds to the engine's
                                    # `mii_hair_color1` uniform (G channel of
                                    # the Mim mask). Off by default; ticking
                                    # the checkbox blends the picked sub-tint
                                    # onto the bright stripes.
                                    use_hair_subcolor_cb = gr.Checkbox(
                                        value=False,
                                        label="Use sub-colour for highlight stripe")
                                    hair_subcolor_gallery, hair_subcolor_fav, hair_subcolor_state = _color_picker(
                                        "Hair Sub-Colour", default_slot=99)
                                    # Default selection is L:012 — enabled
                                    # iff that entry is IsFlippable. Per-
                                    # selection updates are wired further
                                    # below alongside the All ↔ Front/Back
                                    # mutual-exclusion handlers.
                                    _l012 = mm.parts().get("HairAllLegacy012", {})
                                    hair_flip_btn = gr.Button(
                                        "Flip L/R", size="sm",
                                        interactive=bool(_l012.get("IsFlippable")))
                                    hair_flip_btn.click(
                                        fn=lambda v: not v,
                                        inputs=hair_flip_state,
                                        outputs=hair_flip_state)
                                    # HairBack+Parts side selector — picks
                                    # which attachment slot within the
                                    # row implied by the icon's C/U/D
                                    # suffix is rendered. Available sides
                                    # depend on what the chosen Parts
                                    # ships in its GLB; per-button
                                    # interactive flags are recomputed
                                    # when the HairBack state changes.
                                    with gr.Row(equal_height=True):
                                        hp_side_c_btn  = gr.Button("Center", size="sm",
                                                                   min_width=0, scale=1, interactive=False)
                                        hp_side_lr_btn = gr.Button("L+R",    size="sm",
                                                                   min_width=0, scale=1, interactive=False)
                                        hp_side_l_btn  = gr.Button("Left",   size="sm",
                                                                   min_width=0, scale=1, interactive=False)
                                        hp_side_r_btn  = gr.Button("Right",  size="sm",
                                                                   min_width=0, scale=1, interactive=False)
                                    for _btn, _side in (
                                        (hp_side_c_btn,  "Center"),
                                        (hp_side_lr_btn, "LeftRight"),
                                        (hp_side_l_btn,  "Left"),
                                        (hp_side_r_btn,  "Right"),
                                    ):
                                        _btn.click(fn=(lambda _s=_side: _s),
                                                   inputs=None,
                                                   outputs=hair_parts_side_state)
                                elif group_label == "Eyebrow":
                                    eyebrow_color_gallery, eyebrow_color_fav, eyebrow_color_state = _color_picker(
                                        "Eyebrow Color", default_slot=60)
                                    with gr.Row():
                                        brow_y_sl  = gr.Slider(0,  6, value=0,    step=0.5, label="Eyebrow positionY")
                                        brow_x_sl  = gr.Slider(0, 12, value=1.5,  step=0.5, label="Eyebrow spacingX")
                                        brow_sc_sl = gr.Slider(-2, 2, value=0,    step=0.5, label="Eyebrow scale")
                                elif group_label == "Nose":
                                    # Nose 3D mesh has no own colour (it's
                                    # tinted as skin), so the right column
                                    # only carries the size / Y-offset
                                    # sliders. Centre slot = no offset / no
                                    # scale change; range matches the
                                    # in-game editor's nose tweaks.
                                    with gr.Row():
                                        nose_y_sl  = gr.Slider(-6, 6, value=0,
                                                               step=0.5, label="Nose positionY")
                                        nose_sc_sl = gr.Slider(-6, 6, value=0,
                                                               step=0.5, label="Nose scale")
                                elif group_label == "Mouth":
                                    # Favourites click writes a (lower, upper)
                                    # pair so the engine's split lipstick
                                    # colouring (CommonColorTable for lower lip,
                                    # UpperLipColorTable for upper lip) is
                                    # preserved. Full-palette click still
                                    # writes a single tuple → both lips share.
                                    mouth_full_g, mouth_fav_g, mouth_color_dd = _full_color_picker(
                                        "Mouth Color", _NSO_MOUTH_R_COLORS, _mouth_swatches,
                                        default_lin=(tuple(_NSO_MOUTH_R_COLORS[0]),
                                                     tuple(_NSO_MOUTH_G_COLORS[0])),
                                        fav_value_fn=lambda i: (
                                            tuple(_NSO_MOUTH_R_COLORS[int(i)]),
                                            tuple(_NSO_MOUTH_G_COLORS[int(i)]),
                                        ),
                                        # Full-palette click: derive an upper-
                                        # lip colour from the picked lower-lip
                                        # colour by darkening ≈3× (matches the
                                        # average ratio between CommonColorTable
                                        # and UpperLipColorTable across the 5
                                        # curated lipsticks).
                                        full_value_fn=lambda rgb: (
                                            tuple(rgb),
                                            tuple(c * 0.32 for c in rgb),
                                        ))
                                    _mouth_default_name = next(
                                        (n for _, n in _gallery_items("Mouth")
                                         if n != "(none)"), None)
                                    mouth_no_lip_cb = gr.Checkbox(
                                        value=_mouth_default_no_lip(_mouth_default_name),
                                        label="No lipstick (use _NoLip variant or strip R/G)",
                                        interactive=True)
                                    with gr.Row():
                                        mouth_y_sl  = gr.Slider(-3,  6, value=-2, step=0.5, label="Mouth positionY (-3=up, 6=down)")
                                        mouth_sc_sl = gr.Slider(-2,  2, value=0,    step=0.5, label="Mouth scale")
                                elif group_label == "Ears":
                                    with gr.Row():
                                        ear_sc_sl = gr.Slider(-6, 6, value=0,
                                                              step=0.5, label="Ear scale")
                                elif group_label == "Glasses":
                                    glass_full_g, glass_fav_g, glass_dd = _full_color_picker(
                                        "Glass Frame", _glass_lin, _glass_swatches,
                                        default_lin=GLASS_TABLE["Black"][:3])
                                    with gr.Row():
                                        glass_y_sl  = gr.Slider(-6, 6, value=0,
                                                                step=0.5, label="Glass positionY")
                                        glass_sc_sl = gr.Slider(-6, 6, value=0,
                                                                step=0.5, label="Glass scale")

            # HairParts no longer has its own gallery — it's driven by
            # the HairBack tile's compound `<HHH>:<PPP>@<X>` label,
            # which build_face copies into this hidden state at render
            # time. Likewise HairAll is folded into HairAllLegacy's
            # unified `L:<NNN>` / `N:<NNN>` gallery. Register empty
            # placeholders so the positional glb_states assembly below
            # still finds a key for each.
            if "HairParts" not in glb_state_by_cat:
                glb_state_by_cat["HairParts"] = gr.State("(none)")
            if "HairAll" not in glb_state_by_cat:
                glb_state_by_cat["HairAll"] = gr.State("(none)")
            # Re-order state objects to match PARTS / FACE_PARTS order so
            # build_face's positional unpack still works.
            glb_states  = [glb_state_by_cat[cat]      for cat,_,_,_ in PARTS]
            face_states = [face_state_by_pref[pref]   for pref, _   in FACE_PARTS]


    pos_sliders = [eye_y_sl, eye_x_sl, eye_sc_sl,
                   brow_y_sl, brow_x_sl, brow_sc_sl,
                   mouth_y_sl, mouth_sc_sl,
                   nose_y_sl, nose_sc_sl,
                   glass_y_sl, glass_sc_sl,
                   ear_sc_sl]
    # build_face unpacks args positionally — order matters here.
    # Hair/Eyebrow/Beard/Mustache colours are gr.State (int slot index).
    color_dds = [eye_color_dd, mouth_color_dd,
                 hair_color_state, eyebrow_color_state,
                 beard3d_color_state, beard2d_color_state,
                 mustache_color_state]
    expr_dds  = [expression_dd]
    extras    = [mouth_no_lip_cb, eye_use_red_cb, eyeline_color_state,
                 hair_flip_state, hair_parts_side_state,
                 hair_subcolor_state, use_hair_subcolor_cb]
    all_inputs = (glb_states + face_states + [skin_dd, glass_dd]
                  + pos_sliders + color_dds + expr_dds + extras)

    # Wire gallery → state. The full palette is anti-diagonal-mirrored,
    # so click position needs the inverse mapping back to slot index.
    # Favourites strip maps row index → slot via _FAV_HAIR_SLOTS. Each
    # click also clears the OTHER gallery's `selected_index` so the
    # two stay mutually exclusive (no orphan highlight in fav after a
    # full-palette pick).
    def _on_color_select(evt: gr.SelectData):
        p = int(evt.index)
        return 99 - 10 * (p % 10) - (p // 10), gr.update(selected_index=None)
    def _on_fav_select(evt: gr.SelectData):
        return int(_FAV_HAIR_SLOTS[evt.index]), gr.update(selected_index=None)
    for gallery, fav, state in [
        (hair_color_gallery,    hair_color_fav,    hair_color_state),
        (hair_subcolor_gallery, hair_subcolor_fav, hair_subcolor_state),
        (eyebrow_color_gallery, eyebrow_color_fav, eyebrow_color_state),
        (beard3d_color_gallery, beard3d_color_fav, beard3d_color_state),
        (beard2d_color_gallery, beard2d_color_fav, beard2d_color_state),
        (mustache_color_gallery,mustache_color_fav,mustache_color_state),
    ]:
        gallery.select(_on_color_select, outputs=[state, fav])
        fav.select(_on_fav_select, outputs=[state, gallery])
        state.change(build_face, inputs=all_inputs, outputs=viewer)

    for ctrl in [skin_dd, glass_dd] + pos_sliders + [eye_color_dd, mouth_color_dd] + expr_dds + extras:
        ctrl.change(build_face, inputs=all_inputs, outputs=viewer)

    # Mouth gallery icons are tinted with the selected mouth_color —
    # regenerate the gallery's value when the colour dropdown changes.
    _mouth_gallery = next((g for g, _, _, ns in face_gallery_info
                           if any(n.startswith("Mouth") for n in ns if n != "(none)")), None)
    _mouth_gallery_names = next((ns for _, _, _, ns in face_gallery_info
                                 if any(n.startswith("Mouth") for n in ns if n != "(none)")), None)
    if _mouth_gallery is not None:
        def _retint_mouth_gallery(color_val):
            # color_val is now a linear (R,G,B) tuple from the swatch
            # picker; pass through to _face_icon_path which threads it
            # into _mouth_colored_icon's tuple-aware path.
            return [_face_icon_path("Mouth", n, mouth_color=color_val)
                    if n != "(none)" else _none_icon_path("Mouth")
                    for n in _mouth_gallery_names]
        mouth_color_dd.change(_retint_mouth_gallery,
                              inputs=mouth_color_dd, outputs=_mouth_gallery)

    # Same retint dance for the Eye gallery — composite the colour layer
    # (white iris fill) with the outline overlay using the picked iris
    # colour so the gallery tile reflects what the rendered face shows.
    _eye_gallery = next((g for g, _, _, ns in face_gallery_info
                         if any(n.startswith("Eye") and not n.startswith(("Eyebrow","Eyelid","Eyelash"))
                                for n in ns if n != "(none)")), None)
    _eye_gallery_names = next((ns for _, _, _, ns in face_gallery_info
                               if any(n.startswith("Eye") and not n.startswith(("Eyebrow","Eyelid","Eyelash"))
                                      for n in ns if n != "(none)")), None)
    if _eye_gallery is not None:
        def _retint_eye_gallery(color_val):
            return [_face_icon_path("Eye", n, eye_color=color_val)
                    if n != "(none)" else _none_icon_path("Eye")
                    for n in _eye_gallery_names]
        eye_color_dd.change(_retint_eye_gallery,
                            inputs=eye_color_dd, outputs=_eye_gallery)

    def _make_select_fn(name_list):
        def on_select(evt: gr.SelectData):
            name = name_list[evt.index] if evt.index < len(name_list) else "(none)"
            return name, (f"**{name}**" if name != "(none)" else "*(none)*")
        return on_select

    for gallery, state, sel_md, names in glb_gallery_info + face_gallery_info:
        gallery.select(_make_select_fn(names), outputs=[state, sel_md])
        state.change(build_face, inputs=all_inputs, outputs=viewer)

    # The unified-Hair "All" gallery (HairAllLegacy) renders the entire
    # head of hair on its own; picking a HairFront or HairBack mesh on
    # top would double-up. Wire mutual exclusion: selecting on either
    # side resets the other to `(none)` (state + markdown + gallery
    # highlight). HairFront/HairBack galleries don't surface a `(none)`
    # tile, so the "cleared" state shows no highlight.
    def _clear_to_none(picked, sibling_opts):
        if picked == "(none)":
            return gr.update(), gr.update(), gr.update()
        none_idx = sibling_opts.index("(none)") if "(none)" in sibling_opts else None
        return ("(none)", "*(none)*", gr.Gallery(selected_index=none_idx))
    if "HairAllLegacy" in glb_gallery_by_cat:
        hal_g, hal_s, hal_md, hal_opts = glb_gallery_by_cat["HairAllLegacy"]
        for side in ("HairFront", "HairBack"):
            if side not in glb_gallery_by_cat:
                continue
            sg, ss, smd, sopts = glb_gallery_by_cat[side]
            hal_s.change(lambda v, _so=sopts: _clear_to_none(v, _so),
                         inputs=hal_s, outputs=[ss,  smd,  sg])
            ss.change(lambda v, _so=hal_opts: _clear_to_none(v, _so),
                      inputs=ss,  outputs=[hal_s, hal_md, hal_g])
        # Enable the Flip L/R button only when the active hair (across
        # the three Hair subtabs — All, Front, Back) is flagged
        # IsFlippable in its bgyml. For non-flippable picks the toggle
        # would be a no-op, so disable it AND reset hair_flip_state to
        # False so a stale flip from a previous selection doesn't
        # linger when the user switches hairs.
        def _hair_bgyml_name(cat, v):
            if not isinstance(v, str) or v == "(none)":
                return None
            if cat == "HairAllLegacy" and v[:2] in ("L:", "N:"):
                kind, sub = v.split(":", 1)
                return f"HairAllLegacy{sub}" if kind == "L" else f"HairAll{sub}"
            if cat == "HairFront":
                return f"HairFront{v}"
            if cat == "HairBack":
                # Compound `<HHH>:<PPP>@<X>` reduces to the HairBack mesh.
                return f"HairBack{v.split(':', 1)[0]}"
            return None
        def _flip_btn_state(hal_v, hf_v, hb_v):
            for c, v in (("HairAllLegacy", hal_v),
                         ("HairFront",     hf_v),
                         ("HairBack",      hb_v)):
                n = _hair_bgyml_name(c, v)
                if n and mm.parts().get(n, {}).get("IsFlippable"):
                    return gr.Button(interactive=True), gr.update()
            return gr.Button(interactive=False), False
        _flip_inputs = [hal_s]
        for side in ("HairFront", "HairBack"):
            if side in glb_gallery_by_cat:
                _flip_inputs.append(glb_gallery_by_cat[side][1])
            else:
                _flip_inputs.append(gr.State("(none)"))
        for src in _flip_inputs:
            src.change(_flip_btn_state, inputs=_flip_inputs,
                       outputs=[hair_flip_btn, hair_flip_state])
        # HairBack compound `<HHH>:<PPP>@<X>` toggles which Parts side
        # buttons are usable: read the row from `<X>` and intersect with
        # the parts' available positions. Bare HairBack picks (no parts)
        # disable all four. Also reset hair_parts_side_state to a side
        # that's actually present so the render doesn't silently fall
        # back through the priority chain.
        if "HairBack" in glb_gallery_by_cat:
            hb_state = glb_gallery_by_cat["HairBack"][1]
            def _side_btn_state(hb_v):
                avail = set()
                if isinstance(hb_v, str) and ":" in hb_v:
                    row, base = _hair_parts_row(hb_v)
                    pos = _hair_parts_available_positions(base)
                    if row:
                        avail = {p[len(row):] for p in pos
                                 if p.startswith(row)}
                first = next((s for s in _HAIR_PARTS_POS_PRIORITY_SIDES
                              if s in avail), None)
                return (
                    gr.Button(interactive=("Center"    in avail)),
                    gr.Button(interactive=("LeftRight" in avail)),
                    gr.Button(interactive=("Left"      in avail)),
                    gr.Button(interactive=("Right"     in avail)),
                    first if first else gr.update(),
                )
            hb_state.change(_side_btn_state, inputs=hb_state,
                            outputs=[hp_side_c_btn, hp_side_lr_btn,
                                     hp_side_l_btn, hp_side_r_btn,
                                     hair_parts_side_state])

    # When the Mouth gallery selection changes, reset the no-lipstick
    # toggle to the new mouth's per-mouth default (mouths with no
    # _NoLip variant default to no-lip; mouths with one default to
    # with-lip).
    for gallery, state, sel_md, names in face_gallery_info:
        if names and any(n != "(none)" and n.startswith("Mouth") for n in names):
            def _mouth_select(evt: gr.SelectData, _names=names):
                name = _names[evt.index] if evt.index < len(_names) else "(none)"
                return _mouth_default_no_lip(name)
            gallery.select(_mouth_select, outputs=mouth_no_lip_cb)
            break

    demo.load(lambda *a: build_face(*a), inputs=all_inputs, outputs=viewer)

if __name__ == "__main__":
    # Remote-access knobs — pick at launch time via env vars:
    #   MII_SHARE=1            → publish via Gradio's gradio.live tunnel
    #   MII_HOST=0.0.0.0       → listen on all interfaces (LAN access)
    #   MII_PORT=7860          → custom port (default 7860)
    #   MII_AUTH=user:password → require basic auth
    share = bool(int(os.environ.get("MII_SHARE", "0")))
    host  = os.environ.get("MII_HOST", "127.0.0.1")
    port  = int(os.environ.get("MII_PORT", "7860"))
    auth  = os.environ.get("MII_AUTH")
    auth_tuple = tuple(auth.split(":", 1)) if auth and ":" in auth else None
    demo.launch(
        inbrowser=not share and host == "127.0.0.1",
        share=share, server_name=host, server_port=port, auth=auth_tuple,
        head=_HEAD,
        # `css=` moved from gr.Blocks() to launch() in Gradio 6.0.
        # Without this, the .mii-palette tile-size rules were silently
        # dropped, leaving Gradio's default Gallery layout (each tile
        # filled the row, gallery scrolled).
        css=_PALETTE_CSS + "\n" + _TAB_ICON_CSS,
        allowed_paths=[MIIPARTS_DIR, _EDITOR_ICONS,
                       tempfile.gettempdir(),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_glass_icons"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_mouth_icons"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_hair_swatches"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_skin_swatches"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_glass_swatches"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_eye_swatches"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_mouth_swatches"),
                       os.path.join(tempfile.gettempdir(), "mii_face_renderer_eye_icons")])
