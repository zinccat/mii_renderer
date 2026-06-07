# Mii Face Renderer — guiding principles

## Hard rules (no exceptions)

1. **No empirical / "looks-right" tunings.** Every numeric constant in
   `face_3d_demo.py` must be derived from one of:
   - The decompressed Switch 2 NSO binary (`/tmp/nso_re/`) — float-pool
     literals, decompiled formulas, `nn::mii::*` SDK symbol values.
   - The `bgyml` metadata under `assets/mii_metadata/`.
   - The BNTX texture data (alpha-weighted bounding boxes, native sizes
     from BNTX headers).
   - The GLB Mask mesh UV layout (`assets/glb/MiiHead*.glb`).
   - The FFL/FFLNX reference sources (Wii U Mii rendering — `aboood40091/FFL`)
     when explicitly cross-checked against the Switch 2 binary.
   If a constant is needed but cannot be sourced this way, the answer is
   "we don't know" — not a hand-fit number.

2. **Anchor face-feature positions to the engine's eye RECTANGLE,
   not the sprite frame.** Our eye textures are 152×128 with iris
   content occupying only the central 38–92 rows. The engine's
   `eye.scale.y` matrix-input rectangle corresponds to the **visible
   iris content area**, not the texture frame. So lash/lid/highlight
   position formulas that use "eye top" or "eye bottom" must use
   the iris content bounding box (computed per-sprite from alpha
   mask), not the sprite frame.

3. **When stuck, dig deeper into the binary.** Use Ghidra
   (`/opt/homebrew/Cellar/ghidra/12.0.4/`) on the synthesized ELF
   `/tmp/nso_re/main.elf`. Project at `/tmp/ghidra_proj/nso_main`.
   Decompile the relevant functions, follow string xrefs, walk the
   call graph. Do not guess.

4. **Document provenance.** Every NSO-derived constant in code must
   have a comment citing the source VA (e.g. `0.0345479 @ 0x2601e10`).
   FORMULAS.md and EMPIRICAL_AUDIT.md in `/tmp/nso_re/` track the
   confidence level of each derivation.

## Quick references

- Engine compositor function (Switch 2): `FUN_710016facc` — switch on
  feature type (cases 0..21).
- Lash/lid/highlight compositor: `FUN_71001733a0` (type 0xd) — formula
  in `_overlay_placement`. Other variants at `FUN_7100175630` (0xe),
  etc.
- Matrix builder (CalcMVMatrix equivalent): `FUN_7100171910`.
- Eye compositor: `FUN_71001737d0` (writes eye.pos and eye.scale to
  the descriptor at param_1[7], [8], [2], [3]).
- Face mask render target size: 256×256 (`nn::mii::ImageDatabase::ImageWidth/Height`
  @ sdk.nso 0x987b94/b98).
- Y-down ortho convention: `-2/256` matrices in main.nso .data
  (e.g. 0x35cef7c, 0x35cef90, 0x35cefa4).
