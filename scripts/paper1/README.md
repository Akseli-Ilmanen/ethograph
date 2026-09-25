# Paper 1 figure scripts

Every script here is run directly (`python scripts/paper1/<name>.py`), reads a
folder under `projects/paper/` or the example-data cache, and writes
`<name>.{pdf,svg,png}` next to its inputs or into `projects/paper/<figure>/`.

| Script | Figure |
|---|---|
| `crossval_bars.py` | Frame and segmental F1 per bird, MLP vs C2F-TCN (`projects/paper/crossval_birds`) |
| `changepoint_bars.py` | Segmental F1 with vs without changepoint features (`projects/paper/changepoint`) |
| `embedding_heatmaps.py` | S3D embeddings and the `s3d` variable of one trial as GUI-style sorted heatmaps with the label strip (`projects/paper/video_feats`) |
| `embedding_grid.py` | S3D and FERAL embeddings of two videos as a 2 × 2 grid of sorted heatmaps, no labels (`projects/paper/video_feats`) |
| `s3d_ablation_bars.py` | Segmental F1 of the MSc S3D ablation, top-20 vs all 1024 dims; values hardcoded from `s3d_ablation.ipynb`, whose source `.npy` files are on the lab machine (`projects/paper/s3d_ablation`) |
| `figure_review.py` | The review workflow per event kind: bulk grid → in depth, with the confidence knobs (state: trial and segment thresholds over the entropy curve; point: focus, ratio and the α trade-off) (`projects/paper/figure1`) |
| `session.py` | Shared session loading and the `save_pdf` helper for the neural figures |

## Exporting figures CorelDRAW can open

CorelDRAW fails on matplotlib's defaults in three separate ways. Every script
sets all three fixes; copy the block when adding a script.

```python
mpl.rcParams.update(
    {
        "svg.fonttype": "none",       # 1
        "pdf.use14corefonts": True,   # 2
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "path.simplify": False,
    }
)
...
for artist in [*ax.patches, *ax.collections, *ax.lines]:
    artist.set_clip_on(False)         # 3
```

1. **SVG text.** `svg.fonttype = "path"` writes each glyph once into `<defs>`
   and places it with `<use>`; Corel drops those, so the text disappears.
   `"none"` writes real `<text>` elements, which Corel reads.
2. **PDF fonts.** The default (`pdf.fonttype = 3`) embeds Type 3 fonts and
   `42` on its own embeds subsetted CID fonts (`Type0` / `CIDFontType2`).
   Corel reports both as a corrupt file. `pdf.use14corefonts = True` uses the
   14 standard PDF fonts, which are never embedded, so Corel substitutes
   Helvetica and opens the file. Text therefore renders in Helvetica, not
   DejaVu Sans.
3. **SVG clipping.** Matplotlib wraps every bar, marker and line in a group
   with an SVG `clip-path`, and Corel drops clipped groups, so the bars vanish
   while the (unclipped) text survives. Turning `clip_on` off on every drawn
   artist removes the clip-paths. Only do this for artists that lie inside
   the axes anyway.

4. **No math text.** A `$...$` label (subscripts, Greek) is rendered by
   matplotlib's own DejaVu fonts, which brings the CID subset back into the
   PDF and `<use>` glyphs back into the SVG regardless of the settings above.
   Build subscripts from separate plain-text pieces instead
   (`s3d_ablation_bars.py::draw_legend`).

Save the PDF through `PdfPages` with `bbox_inches=None`, the way
`session.py::save_pdf` does; `bbox_inches="tight"` is fine for the PNG.

Check a written file without opening Corel:

```bash
# fonts in the PDF: want only Helvetica / Type1, never Type3 or CIDFontType2
python -c "import re;b=open('fig.pdf','rb').read();print(set(re.findall(rb'/Subtype\s*/(Type\d|TrueType|CIDFontType\d)',b)))"
# SVG: want <text> elements present and zero clip-paths
grep -c "<text" fig.svg; grep -c "clip-path" fig.svg
```
