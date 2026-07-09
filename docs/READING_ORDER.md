# Reading Order for Complex OCR Layouts

OCR quality is not only about recognizing characters. In brochures, triptychs, infographics, Japanese-style layouts, manga pages, or creative flyers, the hardest problem is often the **reading order**.

This project now includes a lightweight reading-order layer for block OCR.

## Problem

The old block OCR logic assumed:

```text
top-to-bottom, left-to-right
```

That works for many Western documents, but fails for:

- three-panel brochures / triptychs
- multi-column layouts
- right-to-left panel flow
- vertical reading layouts
- bottom-up creative layouts
- mixed infographic blocks

## Strategy names

Set the desired strategy with the `OCR_READING_ORDER` environment variable.

```bash
OCR_READING_ORDER=triptych-ltr python main.py brochure.png --format txt -o ./output
```

Available strategies:

| Strategy | Meaning |
|---|---|
| `auto` | Infer a reasonable order from block geometry. Default. |
| `ltr-tb` | Left-to-right inside rows, rows top-to-bottom. Legacy Western order. |
| `rtl-tb` | Right-to-left inside rows, rows top-to-bottom. |
| `tb-lr` | Top-to-bottom inside columns, columns left-to-right. |
| `tb-rl` | Top-to-bottom inside columns, columns right-to-left. Useful for some vertical/Japanese-like layouts. |
| `bu-lr` | Bottom-up inside columns, columns left-to-right. |
| `bu-rl` | Bottom-up inside columns, columns right-to-left. |
| `triptych-ltr` | Three-panel layout, panels left-to-right, text top-to-bottom. |
| `triptych-rtl` | Three-panel layout, panels right-to-left, text top-to-bottom. |
| `triptych-bottom-up` | Three-panel layout, panels left-to-right, text bottom-up inside each panel. |

## Examples

Normal Western flyer:

```bash
OCR_READING_ORDER=ltr-tb python main.py flyer.png --format txt -o ./output
```

Typical tri-fold brochure:

```bash
OCR_READING_ORDER=triptych-ltr python main.py triptico.png --format txt -o ./output
```

Reverse panel tri-fold:

```bash
OCR_READING_ORDER=triptych-rtl python main.py triptico.png --format txt -o ./output
```

Bottom-up creative layout:

```bash
OCR_READING_ORDER=triptych-bottom-up python main.py triptico.png --format txt -o ./output
```

Vertical columns, right-to-left panel flow:

```bash
OCR_READING_ORDER=tb-rl python main.py japanese_layout.png --format txt -o ./output
```

## How it works

1. The image is split into visual text blocks.
2. Blocks keep bounding boxes: `x`, `y`, `width`, `height`.
3. The reading-order strategy sorts those boxes.
4. OCR is applied block by block.
5. The final text is emitted in the chosen logical order.

This keeps layout ordering separate from OCR recognition.

## Design boundary

This is a heuristic layout-order layer, not a full document-layout ML model.

It is lightweight and local-first. It does not require deep learning, GPU, cloud OCR, or embeddings.

For very complex layouts, the recommended workflow is to test several strategies and compare output:

```bash
OCR_READING_ORDER=triptych-ltr python main.py triptico.png --format txt -o ./out_ltr
OCR_READING_ORDER=triptych-rtl python main.py triptico.png --format txt -o ./out_rtl
OCR_READING_ORDER=triptych-bottom-up python main.py triptico.png --format txt -o ./out_bottom
```

Future improvement: expose this as a first-class CLI flag, e.g. `--reading-order triptych-ltr`, instead of only through `OCR_READING_ORDER`.
