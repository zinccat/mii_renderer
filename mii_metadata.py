"""Loader for converted Mii BGYML metadata.

Parses the YAML produced by NintendoToolbox --convert from:
  - MiiParts.pack/Mii/Parts/<name>.mii__Parts.bgyml.yaml
  - Mii/PartsOrder/<cat>.mii__PartsOrder.bgyml.yaml
  - MiiExpression.pack/Mii/Expression/<name>.mii__Expression.bgyml.yaml
  - MiiPartsLocation.pack/Mii/PartsLocation/<name>.mii__PartsLocation.bgyml.yaml
  - Mii_root/MiiEyeAccessoryParam.byml.yaml  (per-eye overlay placement table)

Tiny ad-hoc parser: handles flow form `{ a: 1, b: foo }`, block form,
and the `!u 0x...` tag (returned as int).
The eye-accessory rstbl is a complex block-list YAML — parsed with PyYAML
and a custom `!u`/`!h32` constructor.
"""
import os, re, glob

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mii_metadata")

_NUM_RE = re.compile(r'^-?\d+(?:\.\d+)?$')
_U_RE   = re.compile(r'^!u\s+(0x[0-9a-fA-F]+|\d+)$')

def _coerce(v):
    v = v.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v in ("true", "True"):  return True
    if v in ("false", "False"): return False
    m = _U_RE.match(v)
    if m:
        return int(m.group(1), 0)
    if _NUM_RE.match(v):
        return float(v) if "." in v else int(v)
    if v.startswith("{") and v.endswith("}"):
        return _parse_flow(v[1:-1])
    if v.startswith("[") and v.endswith("]"):
        return [_coerce(x) for x in _split_top(v[1:-1])]
    return v  # bare string

def _split_top(s):
    """Split on commas that are not inside braces/brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[": depth += 1
        elif ch in "}]": depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        tail = "".join(cur).strip()
        if tail:
            out.append(tail)
    return out

def _parse_flow(body):
    """Parse `key: val, key: val` (flow mapping body)."""
    d = {}
    for item in _split_top(body):
        if ":" not in item: continue
        k, v = item.split(":", 1)
        d[k.strip()] = _coerce(v)
    return d

def _parse_yaml(text):
    """Parse one of the small bgyml.yaml files into dict (or list for Order: form)."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return _parse_flow(text[1:-1])
    out = {}
    cur_key, cur_list = None, None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("- "):
            if cur_list is not None:
                cur_list.append(_coerce(line[2:]))
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip(); v = v.strip()
            if v == "":
                cur_key, cur_list = k, []
                out[k] = cur_list
            else:
                out[k] = _coerce(v)
                cur_key, cur_list = None, None
    return out

# ── Loaders ───────────────────────────────────────────────────────────────────

_PARTS_DIR  = f"{ROOT}/MiiParts.pack/Mii/Parts"
_ORDER_DIR  = f"{ROOT}/Mii_root/PartsOrder"
_EXPR_DIR   = f"{ROOT}/MiiExpression.pack/Mii/Expression"
_LOC_DIR    = f"{ROOT}/MiiPartsLocation.pack/Mii/PartsLocation"
_EYEACC_PATH = f"{ROOT}/Mii_root/MiiEyeAccessoryParam.byml.yaml"

_parts_cache  = None
_order_cache  = None
_expr_cache   = None
_loc_cache    = None
_eyeacc_cache = None

def parts():
    """{name: {Category, FileName, TextureName, PartsIndex, AxisForExpression,
                IsVisibleInEditor, EditorIconName, OffsetRotate?, IsMouthOpen?, ...}}"""
    global _parts_cache
    if _parts_cache is not None:
        return _parts_cache
    out = {}
    if os.path.isdir(_PARTS_DIR):
        for path in glob.glob(f"{_PARTS_DIR}/*.bgyml.yaml"):
            name = os.path.basename(path).replace(".mii__Parts.bgyml.yaml", "")
            try:
                out[name] = _parse_yaml(open(path).read())
            except Exception as e:
                print(f"[mii_metadata] parts {name}: {e}")
    _parts_cache = out
    return out

def parts_order():
    """{category_string: [parts_index_list_in_editor_order]}.

    Note: PartsIndex from parts() may not match these indices for every
    category — Mole/Mouth use different schemes. Use cautiously."""
    global _order_cache
    if _order_cache is not None:
        return _order_cache
    out = {}
    if os.path.isdir(_ORDER_DIR):
        for path in glob.glob(f"{_ORDER_DIR}/*.bgyml.yaml"):
            cat = os.path.basename(path).replace(".mii__PartsOrder.bgyml.yaml", "")
            d = _parse_yaml(open(path).read())
            out[cat] = d.get("Order", [])
    _order_cache = out
    return out

def expressions():
    """{expr_name: {EyePartsLName, EyePartsRName, MouthPartsName, EyebrowLocationName,
                     EyeLocationName, IsEnableLipSync, LipSyncMouthPartsName, ...}}"""
    global _expr_cache
    if _expr_cache is not None:
        return _expr_cache
    out = {}
    if os.path.isdir(_EXPR_DIR):
        for path in glob.glob(f"{_EXPR_DIR}/*.bgyml.yaml"):
            name = os.path.basename(path).replace(".mii__Expression.bgyml.yaml", "")
            out[name] = _parse_yaml(open(path).read())
    _expr_cache = out
    return out

def parts_locations():
    """{loc_name: {Rotate, PositionY, Scale, IsOffsetXxx, ...} or empty}"""
    global _loc_cache
    if _loc_cache is not None:
        return _loc_cache
    out = {}
    if os.path.isdir(_LOC_DIR):
        for path in glob.glob(f"{_LOC_DIR}/*.bgyml.yaml"):
            name = os.path.basename(path).replace(".mii__PartsLocation.bgyml.yaml", "")
            out[name] = _parse_yaml(open(path).read())
    _loc_cache = out
    return out

def _eye_accessory_table():
    """Hash-keyed dict of MiiEyeAccessoryParam rows. Row key = ComponentsHash
    EyeAccessoryRef value from each Eye*.bgyml. Loaded lazily with PyYAML."""
    global _eyeacc_cache
    if _eyeacc_cache is not None:
        return _eyeacc_cache
    out = {}
    if os.path.exists(_EYEACC_PATH):
        try:
            import yaml
            yaml.SafeLoader.add_constructor(
                '!u',   lambda l, n: int(l.construct_scalar(n), 0))
            yaml.SafeLoader.add_constructor(
                '!h32', lambda l, n: l.construct_mapping(n))
            with open(_EYEACC_PATH) as f:
                rows = yaml.safe_load(f) or []
            out = {r['__RowId']: r for r in rows if '__RowId' in r}
        except Exception as e:
            print(f"[mii_metadata] eye-accessory table: {e}")
    _eyeacc_cache = out
    return out

def eye_accessory(eye_name):
    """Return the MiiEyeAccessoryParam row for an Eye part name (e.g. 'Eye066'),
    or {} if not found. Each row holds per-overlay placement:
      <Cat>Pos {X,Y}, <Cat>Rotate, <Cat>Scale, <Cat>Aspect, <Cat>DefaultPartsIndex
    where Cat ∈ {EyelashUpper, EyelashLower, EyelidUpper, EyelidLower, Highlight}.
    Pos uses a 32-unit grid centred at (16,16) on the eye texture."""
    p = parts().get(eye_name, {})
    h = p.get("ComponentsHash", {})
    ref = h.get("EyeAccessoryRef") if isinstance(h, dict) else None
    if ref is None:
        return {}
    return _eye_accessory_table().get(ref, {})

def visible_names_in_order(category):
    """Return part NAMES (e.g. ['Eye060', 'Eye041', ...]) for one category,
    filtered to IsVisibleInEditor=true and sorted by PartsOrder.

    Falls back to lexical order on filename if PartsOrder is missing."""
    p = parts()
    matching = [(n, m) for n, m in p.items()
                if m.get("Category") == category and m.get("IsVisibleInEditor", False)]
    order = parts_order().get(category, [])
    if not order:
        return [n for n, _ in sorted(matching)]
    rank = {idx: i for i, idx in enumerate(order)}
    matching.sort(key=lambda nm: rank.get(nm[1].get("PartsIndex"), 1 << 30))
    return [n for n, _ in matching]

if __name__ == "__main__":
    print(f"parts:        {len(parts())}")
    print(f"parts_order:  {list(parts_order())}")
    print(f"expressions:  {len(expressions())}")
    print(f"locations:    {len(parts_locations())}")
    for cat in ("Eye", "Eyebrow", "Mouth", "Mole"):
        names = visible_names_in_order(cat)
        print(f"{cat:8s} visible={len(names)}  first 8={names[:8]}")
    print(f"Smile expression: {expressions().get('Smile')}")
    print(f"Anger expression: {expressions().get('Anger')}")
    print(f"EyeAnger location: {parts_locations().get('EyeAnger')}")
    print(f"MouthOpenBig location: {parts_locations().get('MouthOpenBig')}")
