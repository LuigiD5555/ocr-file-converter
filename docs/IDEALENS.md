# IdeaLens – Important Idea Detection

IdeaLens is a lightweight post-normalization stage for `ocr-file-converter`.
It finds the most important terms, keyphrases, and idea anchors inside extracted text from PDFs, images, Office files, archives, copied browser text, or stdin.

It is intentionally not an LLM and not a deep-learning component. The default implementation uses local statistical signals:

- keyword frequency
- keyphrase co-occurrence
- graph-like term connectivity
- early structural position
- relation markers such as definition, contrast, rule, cause/effect, exception, example, and conclusion

## Why it belongs in OCR File Converter

The project already turns messy sources into normalized text. IdeaLens adds one optional output layer:

```text
raw document → OCR/extraction → normalization → IdeaLens → ideas.json
```

Origami can then consume `.ideas.json` as a compact conceptual source, while OCR File Converter remains independent.

## CLI examples

Generate only an idea report from a PDF:

```bash
python main.py huge-book.pdf --format ideas -o ./output
```

Generate regular JSON plus an IdeaLens sidecar:

```bash
python main.py huge-book.pdf --format json --ideas -o ./output
```

Analyze text copied from a Brave tab through stdin:

```bash
xclip -selection clipboard -o | python main.py --stdin --format ideas -o ./output
```

Or:

```bash
cat article.txt | python main.py --stdin --format ideas -o ./output
```

Customize limits:

```bash
python main.py document.pdf \
  --format ideas \
  --idea-limit 80 \
  --idea-phrase-limit 80 \
  --idea-anchor-limit 30 \
  --language spa \
  -o ./output
```

Use custom stopwords:

```bash
python main.py document.pdf --format ideas --idea-stopwords config/spanish_stopwords.txt
```

## Output shape

```json
{
  "source": "document.pdf",
  "language": "auto",
  "text_length": 123456,
  "sentence_count": 1200,
  "keywords": [
    {"text": "devengo", "score": 7.21, "occurrences": 18, "first_position": 0.03, "kind": "keyword"}
  ],
  "keyphrases": [
    {"text": "reconocimiento ingresos", "score": 9.88, "occurrences": 5, "first_position": 0.07, "kind": "keyphrase"}
  ],
  "anchors": [
    {
      "id": "idea_0001",
      "text": "Los ingresos se reconocen cuando se devengan, no necesariamente cuando se cobran.",
      "score": 5.33,
      "kind": "concept",
      "position": 0.11,
      "keywords": ["ingresos", "devengan", "cobran"],
      "why_important": ["contains high-ranking document terms"]
    }
  ],
  "outline": ["..."],
  "algorithm": "idealens.statistical_graph.v1"
}
```

## Design boundary

IdeaLens does not replace OCR, Origami, embeddings, or a RAG system.

- OCR File Converter extracts and normalizes text.
- IdeaLens finds important conceptual signals.
- Origami can later compress/rehydrate those signals into a thought capsule.
