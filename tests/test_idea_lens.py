from src.idea_lens import IdeaLens, IdeaLensConfig


def test_idea_lens_extracts_keywords_keyphrases_and_anchors():
    text = """
    El devengo contable permite reconocer ingresos cuando se generan.
    El devengo no es lo mismo que el flujo de efectivo.
    Por ejemplo, una venta a crédito puede reconocerse antes de cobrarse.
    Por lo tanto, confundir devengo con cobro distorsiona los estados financieros.
    """
    lens = IdeaLens(IdeaLensConfig(keyword_limit=10, keyphrase_limit=10, anchor_limit=5))
    report = lens.analyze(text, source="test", language="spa")

    assert report.source == "test"
    assert report.keywords
    assert report.keyphrases
    assert report.anchors
    assert any("devengo" in item.text for item in report.keywords)
    assert any(anchor.kind in {"cause_effect", "example", "concept"} for anchor in report.anchors)
