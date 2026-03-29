from shapely.geometry import Point, box

import pytest


class _FakeMSP:
    def __init__(self):
        self.polylines = []
        self.lines = []
        self.circles = []
        self.ellipses = []

    def add_lwpolyline(self, pts, close=False, dxfattribs=None):
        self.polylines.append((list(pts), close, dxfattribs or {}))

    def add_line(self, p0, p1, dxfattribs=None):
        self.lines.append((p0, p1, dxfattribs or {}))

    def add_circle(self, center, radius, dxfattribs=None):
        self.circles.append((center, radius, dxfattribs or {}))

    def add_ellipse(self, **kwargs):
        self.ellipses.append(kwargs)


def test_canonical_icon_aliases_and_dining_chair_prefix(dxf_drawing_module):
    module = dxf_drawing_module

    assert module._canonical_icon_id("working_table") == "desk"
    assert module._canonical_icon_id("working_chair") == "chair"
    assert module._canonical_icon_id("tv_unit") == "tv"
    assert module._canonical_icon_id("dining_chair_4") == "dining_chair"
    assert module._canonical_icon_id("sofa") == "sofa"


def test_ellipse_selects_major_axis_and_validates_radii(dxf_drawing_module):
    module = dxf_drawing_module
    msp = _FakeMSP()

    module._ellipse(msp, 0, 0, rx=4, ry=2, layer=module.L_FURN)
    module._ellipse(msp, 0, 0, rx=2, ry=4, layer=module.L_FURN)

    assert msp.ellipses[0]["major_axis"] == (4.0, 0.0, 0.0)
    assert pytest.approx(msp.ellipses[0]["ratio"], rel=1e-9) == 0.5
    assert msp.ellipses[1]["major_axis"] == (0.0, 4.0, 0.0)
    assert pytest.approx(msp.ellipses[1]["ratio"], rel=1e-9) == 0.5

    with pytest.raises(ValueError, match="Ellipse radii"):
        module._ellipse(msp, 0, 0, rx=0, ry=1, layer=module.L_FURN)


def test_compute_bed_box_changes_head_side_based_on_door_corner(dxf_drawing_module):
    module = dxf_drawing_module

    # Door near top-left => opposite is bottom-right => head_side bottom.
    res1 = module._compute_bed_box(0, 0, 10, 8, is_master=False, door_pt=(0.2, 7.9))
    assert res1 is not None
    assert res1["head_side"] == "bottom"

    # Door near bottom-left => opposite is top-right => head_side top.
    res2 = module._compute_bed_box(0, 0, 10, 8, is_master=True, door_pt=(0.1, 0.1))
    assert res2 is not None
    assert res2["head_side"] == "top"


def test_is_valid_piece_rejects_outside_collision_and_door_clearance(dxf_drawing_module):
    module = dxf_drawing_module

    room = box(0, 0, 10, 10)
    occupied = [box(2, 2, 4, 4)]
    door = box(7.5, 7.5, 8.5, 8.5)

    assert module._is_valid_piece(room, box(2.5, 2.5, 3.5, 3.5), occupied, None, 0.0) is False  # intersects occupied
    assert module._is_valid_piece(room, box(-1, -1, 1, 1), [], None, 0.0) is False  # outside room
    assert module._is_valid_piece(room, box(7.8, 7.8, 8.2, 8.2), [], door, 0.6) is False  # too close to door
    assert module._is_valid_piece(room, box(5, 5, 6, 6), [], door, 0.6) is True


def test_draw_icon_by_bbox_dispatch_and_error(dxf_drawing_module, monkeypatch):
    module = dxf_drawing_module

    calls = []
    monkeypatch.setattr(module, "_draw_single_bed_in_bbox", lambda *args, **kwargs: calls.append("single_bed"))
    monkeypatch.setattr(module, "_draw_double_bed_in_bbox", lambda *args, **kwargs: calls.append("double_bed"))
    monkeypatch.setattr(module, "_draw_toilet", lambda *args, **kwargs: calls.append("toilet"))

    module._draw_icon_by_bbox(object(), "single_bed", 0, 0, 2, 4)
    module._draw_icon_by_bbox(object(), "double_bed", 0, 0, 2, 4)
    module._draw_icon_by_bbox(object(), "toilet", 0, 0, 2, 4)

    assert calls == ["single_bed", "double_bed", "toilet"]

    with pytest.raises(ValueError, match="Unsupported furniture icon"):
        module._draw_icon_by_bbox(object(), "not_supported", 0, 0, 1, 1)


def test_draw_icon_from_footprint_round_table_path(dxf_drawing_module, monkeypatch):
    module = dxf_drawing_module

    captured = {}

    def fake_round(msp, cx, cy, r):
        captured["cx"] = cx
        captured["cy"] = cy
        captured["r"] = r

    monkeypatch.setattr(module, "_draw_dining_table_round", fake_round)

    fp = Point(0, 0).buffer(2.0, resolution=32)
    module._draw_icon_from_footprint(object(), "dining_table_round", fp)

    assert abs(captured["cx"]) < 1e-6
    assert abs(captured["cy"]) < 1e-6
    assert captured["r"] > 1.8


def test_draw_symbol_functions_emit_primitives(dxf_drawing_module):
    module = dxf_drawing_module
    msp = _FakeMSP()

    module._draw_toilet(msp, 0, 0, 1.0, 1.5)
    module._draw_sink(msp, 2, 0, 1.0, 1.0)
    module._draw_shower(msp, 4, 0, 1.2)
    module._draw_kitchen_sink(msp, 6, 0, 1.4, 0.8)
    module._draw_fridge(msp, 8, 0, 1.0, 1.8)
    module._draw_tv(msp, 10, 0, 1.6, 0.9)
    module._draw_coffee_table(msp, 12, 0, 1.2, 0.7)
    module._draw_dining_table_rect(msp, 14, 0, 1.6, 1.0)
    module._draw_sofa(msp, 16, 0, 2.4, 1.1)
    module._draw_rug(msp, 19, 0, 2.0, 1.2)
    module._draw_dining_chair(msp, 22, 0, 0.8, 0.8)
    module._draw_closet(msp, 24, 0, 1.8, 0.9)
    module._draw_dresser(msp, 27, 0, 1.2, 0.7)
    module._draw_work_desk(msp, 29, 0, 1.6, 0.8)
    module._draw_office_chair(msp, 32, 0, 0.8, 0.8)

    assert len(msp.polylines) > 20
    assert len(msp.lines) > 10
    assert len(msp.circles) > 5
    assert len(msp.ellipses) >= 2


def test_draw_piece_by_name_and_poly_outline_helpers(dxf_drawing_module, monkeypatch):
    module = dxf_drawing_module

    calls = []
    monkeypatch.setattr(module, "_draw_closet", lambda *args, **kwargs: calls.append("closet"))
    monkeypatch.setattr(module, "_draw_dresser", lambda *args, **kwargs: calls.append("dresser"))
    monkeypatch.setattr(module, "_draw_work_desk", lambda *args, **kwargs: calls.append("desk"))
    monkeypatch.setattr(module, "_draw_office_chair", lambda *args, **kwargs: calls.append("chair"))

    module._draw_piece_by_name(object(), "closet", (0, 0, 1, 1))
    module._draw_piece_by_name(object(), "dresser", (0, 0, 1, 1))
    module._draw_piece_by_name(object(), "desk", (0, 0, 1, 1))
    module._draw_piece_by_name(object(), "chair", (0, 0, 1, 1))

    assert calls == ["closet", "dresser", "desk", "chair"]

    msp = _FakeMSP()
    module._draw_poly_outline(msp, box(0, 0, 2, 2), module.L_FURN)
    assert len(msp.polylines) == 1


def test_single_bed_reference_and_bed_wall_helpers(dxf_drawing_module):
    module = dxf_drawing_module

    w, h = module._single_bed_reference_size(0, 0, 6, 4)
    assert w > 0 and h > 0

    wall = module._bed_wall_from_bbox(0, 0, 10, 8, (0.4, 0.5, 3.0, 4.0))
    assert wall in {"left", "right", "top", "bottom"}


