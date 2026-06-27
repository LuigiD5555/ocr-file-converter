"""Reading-order strategies for multi-block OCR layouts.

This module turns detected OCR bounding boxes into a logical reading order.
It is intentionally heuristic and dependency-light: the goal is to support
brochures, triptychs, multi-column pages, vertical layouts, and unusual flows
without requiring ML layout models.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence


class BoxLike(Protocol):
    """Structural protocol for bounding boxes used by OCR layout modules."""

    x: int
    y: int
    width: int
    height: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ReadingOrderConfig:
    """Controls how text blocks are ordered after detection.

    strategy values:
    - auto: infer a robust order from block geometry.
    - ltr-tb: left-to-right inside rows, rows top-to-bottom.
    - rtl-tb: right-to-left inside rows, rows top-to-bottom.
    - tb-lr: top-to-bottom inside columns, columns left-to-right.
    - tb-rl: top-to-bottom inside columns, columns right-to-left.
    - bu-lr: bottom-up inside columns, columns left-to-right.
    - bu-rl: bottom-up inside columns, columns right-to-left.
    - triptych-ltr: three-panel style, panels left-to-right, text top-to-bottom.
    - triptych-rtl: three-panel style, panels right-to-left, text top-to-bottom.
    - triptych-bottom-up: three-panel style, panels left-to-right, text bottom-up.
    """

    strategy: str = "auto"
    row_threshold_ratio: float = 0.025
    column_threshold_ratio: float = 0.035
    triptych_tolerance_ratio: float = 0.18


def order_boxes(
    boxes: Sequence[BoxLike],
    *,
    page_width: int,
    page_height: int,
    config: ReadingOrderConfig | None = None,
) -> List[BoxLike]:
    """Return boxes sorted according to a reading-order strategy."""

    if not boxes:
        return []

    config = config or ReadingOrderConfig()
    strategy = (config.strategy or "auto").lower().replace("_", "-")

    if strategy == "auto":
        strategy = infer_reading_order(boxes, page_width=page_width, page_height=page_height, config=config)

    if strategy == "ltr-tb":
        return _rows_then_columns(boxes, page_height=page_height, config=config, x_reverse=False, y_reverse=False)
    if strategy == "rtl-tb":
        return _rows_then_columns(boxes, page_height=page_height, config=config, x_reverse=True, y_reverse=False)
    if strategy == "tb-lr":
        return _columns_then_rows(boxes, page_width=page_width, config=config, x_reverse=False, y_reverse=False)
    if strategy == "tb-rl":
        return _columns_then_rows(boxes, page_width=page_width, config=config, x_reverse=True, y_reverse=False)
    if strategy == "bu-lr":
        return _columns_then_rows(boxes, page_width=page_width, config=config, x_reverse=False, y_reverse=True)
    if strategy == "bu-rl":
        return _columns_then_rows(boxes, page_width=page_width, config=config, x_reverse=True, y_reverse=True)
    if strategy == "triptych-ltr":
        return _triptych_order(boxes, page_width=page_width, x_reverse=False, y_reverse=False)
    if strategy == "triptych-rtl":
        return _triptych_order(boxes, page_width=page_width, x_reverse=True, y_reverse=False)
    if strategy == "triptych-bottom-up":
        return _triptych_order(boxes, page_width=page_width, x_reverse=False, y_reverse=True)

    # Safe fallback: the legacy order.
    return _rows_then_columns(boxes, page_height=page_height, config=config, x_reverse=False, y_reverse=False)


def infer_reading_order(
    boxes: Sequence[BoxLike],
    *,
    page_width: int,
    page_height: int,
    config: ReadingOrderConfig | None = None,
) -> str:
    """Infer a reading-order strategy from the geometry of detected boxes."""

    config = config or ReadingOrderConfig()
    if _looks_like_triptych(boxes, page_width=page_width, config=config):
        return "triptych-ltr"

    column_groups = _group_by_axis(boxes, axis="x", threshold=max(20, int(page_width * config.column_threshold_ratio)))
    row_groups = _group_by_axis(boxes, axis="y", threshold=max(20, int(page_height * config.row_threshold_ratio)))

    # Many vertical groups and fewer rows usually means a column layout.
    if len(column_groups) >= 2 and len(column_groups) <= max(2, len(row_groups)):
        return "tb-lr"

    return "ltr-tb"


def _looks_like_triptych(
    boxes: Sequence[BoxLike],
    *,
    page_width: int,
    config: ReadingOrderConfig,
) -> bool:
    """Detect a likely tri-fold / three-panel layout."""

    if len(boxes) < 3:
        return False

    thirds = [0, 0, 0]
    for box in boxes:
        center_x = box.x + box.width / 2
        bucket = min(2, max(0, int(center_x / max(page_width / 3, 1))))
        thirds[bucket] += 1

    non_empty = sum(1 for count in thirds if count > 0)
    if non_empty < 3:
        return False

    total = sum(thirds)
    expected = total / 3
    tolerance = max(1.0, expected * (1.0 + config.triptych_tolerance_ratio))
    return all(abs(count - expected) <= tolerance for count in thirds)


def _rows_then_columns(
    boxes: Sequence[BoxLike],
    *,
    page_height: int,
    config: ReadingOrderConfig,
    x_reverse: bool,
    y_reverse: bool,
) -> List[BoxLike]:
    threshold = max(20, int(page_height * config.row_threshold_ratio))
    rows = _group_by_axis(boxes, axis="y", threshold=threshold)
    rows.sort(key=lambda group: _group_center(group, axis="y"), reverse=y_reverse)

    ordered: List[BoxLike] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda box: box.x, reverse=x_reverse))
    return ordered


def _columns_then_rows(
    boxes: Sequence[BoxLike],
    *,
    page_width: int,
    config: ReadingOrderConfig,
    x_reverse: bool,
    y_reverse: bool,
) -> List[BoxLike]:
    threshold = max(20, int(page_width * config.column_threshold_ratio))
    columns = _group_by_axis(boxes, axis="x", threshold=threshold)
    columns.sort(key=lambda group: _group_center(group, axis="x"), reverse=x_reverse)

    ordered: List[BoxLike] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda box: box.y, reverse=y_reverse))
    return ordered


def _triptych_order(
    boxes: Sequence[BoxLike],
    *,
    page_width: int,
    x_reverse: bool,
    y_reverse: bool,
) -> List[BoxLike]:
    panels = defaultdict(list)
    for box in boxes:
        center_x = box.x + box.width / 2
        panel = min(2, max(0, int(center_x / max(page_width / 3, 1))))
        panels[panel].append(box)

    panel_ids = sorted(panels.keys(), reverse=x_reverse)
    ordered: List[BoxLike] = []
    for panel_id in panel_ids:
        panel_boxes = panels[panel_id]
        ordered.extend(sorted(panel_boxes, key=lambda box: (box.y, box.x), reverse=False if not y_reverse else True))
    return ordered


def _group_by_axis(boxes: Sequence[BoxLike], *, axis: str, threshold: int) -> List[List[BoxLike]]:
    key = (lambda box: box.x) if axis == "x" else (lambda box: box.y)
    sorted_boxes = sorted(boxes, key=key)
    groups: List[List[BoxLike]] = []
    current: List[BoxLike] = []
    current_center = None

    for box in sorted_boxes:
        center = _box_center(box, axis=axis)
        if current_center is None:
            current = [box]
            current_center = center
            continue
        if abs(center - current_center) <= threshold:
            current.append(box)
            current_center = _group_center(current, axis=axis)
        else:
            groups.append(current)
            current = [box]
            current_center = center

    if current:
        groups.append(current)
    return groups


def _box_center(box: BoxLike, *, axis: str) -> float:
    if axis == "x":
        return box.x + box.width / 2
    return box.y + box.height / 2


def _group_center(boxes: Iterable[BoxLike], *, axis: str) -> float:
    values = [_box_center(box, axis=axis) for box in boxes]
    return sum(values) / max(len(values), 1)
