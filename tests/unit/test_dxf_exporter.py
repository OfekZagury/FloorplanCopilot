from shapely.geometry import MultiPolygon, Point, Polygon, box

import pytest


class _FakeText:
    def set_placement(self, *_args, **_kwargs):
        return self


class _FakeMSP:
    def __init__(self):
        self.polylines = []
        self.lines = []
        self.ellipses = []
        self.texts = []

    def add_lwpolyline(self, pts, close=False, dxfattribs=None):
        self.polylines.append((list(pts), close, dxfattribs or {}))

    def add_line(self, p0, p1, dxfattribs=None):
        self.lines.append((p0, p1, dxfattribs or {}))

    def add_ellipse(self, **kwargs):
        self.ellipses.append(kwargs)

    def add_text(self, text, dxfattribs=None):
        self.texts.append((text, dxfattribs or {}))
        return _FakeText()


class _FakeLayers:
    def __init__(self):
        self.created = []

    def new(self, name, dxfattribs):
        self.created.append((name, dxfattribs))


class _FakeDoc:
    def __init__(self):
        self.layers = _FakeLayers()
        self._msp = _FakeMSP()
        self.saved_path = None

    def modelspace(self):
        return self._msp

    def saveas(self, path):
        self.saved_path = path


def test_footprint_to_manual_item_rect_and_circle(dxf_exporter_module):
    module = dxf_exporter_module

    rect_item = module._footprint_to_manual_item("working_table", box(0, 0, 2, 1))
    assert rect_item["icon"] == "desk"
    assert rect_item["display_label"] == "working_table"
    assert rect_item["shape"] == "rect"
    assert rect_item["w"] == 2.0

    round_fp = Point(0, 0).buffer(1.0, resolution=32)
    round_item = module._footprint_to_manual_item("dining_table_round", round_fp)
    assert round_item["shape"] == "circle"
    assert round_item["icon"] == "dining_table_round"
    assert round_item["r"] > 0.9


def test_draw_manual_furniture_item_validates_shape_and_icon(dxf_exporter_module, monkeypatch):
    module = dxf_exporter_module

    draw_calls = []
    monkeypatch.setattr(module, "_draw_dining_table_round", lambda msp, x, y, r: draw_calls.append((x, y, r)))
    monkeypatch.setattr(module, "_draw_icon_by_bbox", lambda *args, **kwargs: draw_calls.append("bbox"))

    with pytest.raises(ValueError, match="Unsupported circular icon"):
        module._draw_manual_furniture_item(_FakeMSP(), {"icon": "chair", "shape": "circle", "x": 1, "y": 1, "r": 1})

    module._draw_manual_furniture_item(_FakeMSP(), {"icon": "dining_table_round", "shape": "circle", "x": 1, "y": 1, "r": 1})
    module._draw_manual_furniture_item(_FakeMSP(), {"icon": "single_bed", "shape": "rect", "x": 2, "y": 2, "w": 2, "h": 1})

    assert len(draw_calls) == 2


def test_place_all_furniture_manual_mode_short_circuit(dxf_exporter_module, monkeypatch):
    module = dxf_exporter_module

    calls = {"manual": 0, "bed": 0}
    monkeypatch.setattr(module, "_draw_manual_furniture", lambda msp, mf: calls.__setitem__("manual", calls["manual"] + 1) or 3)
    monkeypatch.setattr(module, "_place_bedroom", lambda *args, **kwargs: calls.__setitem__("bed", calls["bed"] + 1))

    module.place_all_furniture(_FakeMSP(), {"manual_furniture": {"enabled": True, "items": [{"icon": "single_bed"}]}})

    assert calls["manual"] == 1
    assert calls["bed"] == 0


def test_place_all_furniture_auto_calls_room_placers(dxf_exporter_module, monkeypatch):
    module = dxf_exporter_module

    bed_calls, bath_calls, kitchen_calls, living_calls = [], [], [], []
    monkeypatch.setattr(module, "_place_bedroom", lambda msp, poly, is_master, door_geom=None: bed_calls.append((poly.area, is_master, door_geom is not None)))
    monkeypatch.setattr(module, "_place_bathroom", lambda msp, poly: bath_calls.append(poly.area))
    monkeypatch.setattr(module, "_place_kitchen", lambda msp, poly: kitchen_calls.append(poly.area))
    monkeypatch.setattr(module, "_place_living_room", lambda msp, poly, kitchen_geoms=None, opening_geoms=None: living_calls.append((len(kitchen_geoms or []), len(opening_geoms or []))) or True)

    bedrooms = MultiPolygon([box(0, 0, 4, 4), box(10, 0, 12, 2)])  # first is master by area
    bathrooms = MultiPolygon([box(0, 10, 2, 12)])
    kitchens = MultiPolygon([box(5, 10, 8, 12)])
    living = MultiPolygon([box(0, 20, 8, 26)])
    door = box(0, 0, 0.5, 0.5)

    playground = {
        "bedroom": bedrooms,
        "bathroom": bathrooms,
        "kitchen": kitchens,
        "living": living,
        "door": door,
    }

    module.place_all_furniture(_FakeMSP(), playground)

    assert len(bed_calls) == 2
    assert sum(1 for _, is_master, _ in bed_calls if is_master) == 1
    assert len(bath_calls) == 1
    assert len(kitchen_calls) == 1
    assert len(living_calls) == 1
    assert living_calls[0][0] == 1  # kitchens forwarded


def test_export_to_dxf_writes_layers_draws_structure_and_saves(dxf_exporter_module, monkeypatch, tmp_path):
    module = dxf_exporter_module

    fake_doc = _FakeDoc()
    monkeypatch.setattr(module.ezdxf, "new", lambda setup=True: fake_doc)

    place_calls = []
    monkeypatch.setattr(module, "place_all_furniture", lambda msp, pg: place_calls.append((msp, pg)))

    out_path = tmp_path / "export_test.dxf"
    playground = {
        "wall": box(0, 0, 10, 10),
        "door": box(0, 0, 1, 0.4),
        "living": Polygon([(1, 1), (9, 1), (9, 9), (1, 9)]),
    }

    module.export_to_dxf(playground, output_path=str(out_path))

    created_layer_names = [name for name, _ in fake_doc.layers.created]
    assert {module.L_WALLS, module.L_ROOMS, module.L_DOORS, module.L_LABELS, module.L_FURN}.issubset(set(created_layer_names))
    assert fake_doc.saved_path == str(out_path)
    assert len(place_calls) == 1
    assert len(fake_doc._msp.polylines) >= 2
    assert len(fake_doc._msp.ellipses) >= 1  # door swing ellipse
    assert any("LIVING" in t[0] for t in fake_doc._msp.texts)

