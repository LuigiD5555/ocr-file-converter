from src.reading_order import ReadingOrderConfig, order_boxes


class Box:
    def __init__(self, x, y, width=10, height=10):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x2 = x + width
        self.y2 = y + height


def coords(boxes):
    return [(box.x, box.y) for box in boxes]


def test_ltr_tb_orders_rows_then_columns():
    boxes = [Box(100, 100), Box(10, 10), Box(100, 10), Box(10, 100)]
    ordered = order_boxes(
        boxes,
        page_width=200,
        page_height=200,
        config=ReadingOrderConfig(strategy="ltr-tb"),
    )
    assert coords(ordered) == [(10, 10), (100, 10), (10, 100), (100, 100)]


def test_tb_rl_orders_columns_right_to_left_then_top_down():
    boxes = [Box(10, 100), Box(100, 10), Box(10, 10), Box(100, 100)]
    ordered = order_boxes(
        boxes,
        page_width=200,
        page_height=200,
        config=ReadingOrderConfig(strategy="tb-rl"),
    )
    assert coords(ordered) == [(100, 10), (100, 100), (10, 10), (10, 100)]


def test_triptych_bottom_up_orders_each_panel_bottom_to_top():
    boxes = [
        Box(10, 10), Box(10, 100),
        Box(110, 10), Box(110, 100),
        Box(210, 10), Box(210, 100),
    ]
    ordered = order_boxes(
        boxes,
        page_width=300,
        page_height=200,
        config=ReadingOrderConfig(strategy="triptych-bottom-up"),
    )
    assert coords(ordered) == [
        (10, 100), (10, 10),
        (110, 100), (110, 10),
        (210, 100), (210, 10),
    ]
