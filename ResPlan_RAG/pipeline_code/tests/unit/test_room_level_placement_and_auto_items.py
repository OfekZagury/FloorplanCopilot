from shapely.geometry import Point, Polygon, box


def test_generate_auto_furniture_items_seeds_bed_bath_kitchen(auto_items_module):
    module = auto_items_module

    playground = {
        "bedroom": box(0, 0, 6, 5),
        "bathroom": box(10, 0, 13, 4),
        "kitchen": box(0, 10, 5, 13),
        "door": box(0, 0, 0.6, 0.6),
    }

    items = module.generate_auto_furniture_items(playground)
    icons = [it["icon"] for it in items]

    assert "double_bed" in icons or "single_bed" in icons
    assert "toilet" in icons
    assert "sink" in icons
    assert "shower" in icons
    assert "fridge" in icons
    assert "kitchen_sink" in icons

    bed = next(it for it in items if it["icon"] in {"single_bed", "double_bed"})
    assert "head_side" in bed
    assert bed["shape"] == "rect"


def test_generate_auto_furniture_items_graceful_on_missing_geometry(auto_items_module):
    module = auto_items_module
    assert module.generate_auto_furniture_items({}) == []


def test_place_bathroom_orientation_branching(room_place_module, monkeypatch):
    module = room_place_module

    calls = []
    monkeypatch.setattr(module, "_draw_toilet", lambda *args: calls.append("toilet"))
    monkeypatch.setattr(module, "_draw_sink", lambda *args: calls.append("sink"))
    monkeypatch.setattr(module, "_draw_shower", lambda *args: calls.append("shower"))

    module._place_bathroom(object(), box(0, 0, 6, 3))
    module._place_bathroom(object(), box(0, 0, 3, 6))

    # Each call places exactly three fixtures regardless of orientation.
    assert calls.count("toilet") == 2
    assert calls.count("sink") == 2
    assert calls.count("shower") == 2


def test_place_kitchen_orientation_branching(room_place_module, monkeypatch):
    module = room_place_module

    calls = []
    monkeypatch.setattr(module, "_draw_fridge", lambda *args: calls.append("fridge"))
    monkeypatch.setattr(module, "_draw_kitchen_sink", lambda *args: calls.append("kitchen_sink"))

    module._place_kitchen(object(), box(0, 0, 7, 3))
    module._place_kitchen(object(), box(0, 0, 3, 7))

    assert calls.count("fridge") == 2
    assert calls.count("kitchen_sink") == 2


def test_place_living_room_invalid_geometry_returns_false(room_place_module):
    module = room_place_module
    assert module._place_living_room(object(), Polygon(), return_items=False) is False
    assert module._place_living_room(object(), Polygon(), return_items=True) == []


def test_place_living_room_success_path_with_controlled_helpers(room_place_module, monkeypatch):
    module = room_place_module

    living = box(0, 0, 10, 8)
    frame = {"u_min": 0.0, "u_max": 10.0, "v_min": 0.0, "v_max": 8.0, "diag": 12.8}

    monkeypatch.setattr(module, "_living_local_frame", lambda poly: frame)
    monkeypatch.setattr(module, "_end_zone_polygon", lambda *args, **kwargs: living)
    monkeypatch.setattr(module, "_local_bounds_of_geom", lambda *args, **kwargs: (1.0, 1.0, 9.0, 7.0))
    monkeypatch.setattr(module, "_dims_from_area", lambda area, ratio_wh: (2.0, 1.0))

    def fake_fit_rect(_poly, _frame, w, h, target_u, target_v, **kwargs):
        return {"u": target_u, "v": target_v, "footprint": box(target_u - w / 2.0, target_v - h / 2.0, target_u + w / 2.0, target_v + h / 2.0)}

    monkeypatch.setattr(module, "_fit_rect_inside", fake_fit_rect)
    monkeypatch.setattr(module, "_fit_circle_inside", lambda *args, **kwargs: {"u": 5.0, "v": 4.0, "footprint": Point(5, 4).buffer(1.0)})
    monkeypatch.setattr(module, "_build_dining_rect", lambda *_args, **_kwargs: [
        {"name": "dining_table_rect", "footprint": box(4, 3, 6, 5)},
        {"name": "dining_chair_1", "footprint": box(4, 5, 4.5, 5.5)},
        {"name": "dining_chair_2", "footprint": box(5, 5, 5.5, 5.5)},
        {"name": "dining_chair_3", "footprint": box(6, 5, 6.5, 5.5)},
        {"name": "dining_chair_4", "footprint": box(4, 2.5, 4.5, 3)},
        {"name": "dining_chair_5", "footprint": box(5, 2.5, 5.5, 3)},
        {"name": "dining_chair_6", "footprint": box(6, 2.5, 6.5, 3)},
    ])
    monkeypatch.setattr(module, "_build_dining_round", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "_all_items_inside", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, "_is_layout_valid", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, "_score_layout", lambda *_args, **_kwargs: 0.5)

    drawn = {"items": None}
    monkeypatch.setattr(module, "_draw_living_items", lambda msp, items: drawn.__setitem__("items", items))

    ok = module._place_living_room(object(), living)
    assert ok is True
    assert drawn["items"] is not None
    assert any(it["name"] == "sofa" for it in drawn["items"])


def test_room_helpers_fit_and_layout_validity(room_helpers_module):
    module = room_helpers_module

    living = box(0, 0, 10, 10)
    frame = module._living_local_frame(living)

    fit_rect = module._fit_rect_inside(living, frame, w=2.0, h=1.0, target_u=0.0, target_v=0.0)
    assert fit_rect is not None

    fit_circle = module._fit_circle_inside(living, frame, r=0.8, target_u=0.0, target_v=0.0)
    assert fit_circle is not None

    too_big_circle = module._fit_circle_inside(living, frame, r=50.0, target_u=0.0, target_v=0.0)
    assert too_big_circle is None

    items_ok = [
        {"name": "sofa", "footprint": box(1, 1, 2, 2)},
        {"name": "tv_unit", "footprint": box(3, 1, 4, 2)},
    ]
    assert module._all_items_inside(living, items_ok) is True

    items_bad = [{"name": "sofa", "footprint": box(-1, -1, 1, 1)}]
    assert module._all_items_inside(living, items_bad) is False

    opening = [box(6.0, 6.0, 6.5, 6.5)]
    assert module._is_layout_valid(living, items_ok, opening, clearance=0.01) is True

    # Collision path: two non-exempt items almost touching.
    colliding = [
        {"name": "sofa", "footprint": box(1, 1, 2, 2)},
        {"name": "coffee_table", "footprint": box(1.99, 1, 3, 2)},
    ]
    assert module._is_layout_valid(living, colliding, [], clearance=0.01) is False


def test_place_bedroom_runs_extra_furniture_pipeline(room_place_module, monkeypatch):
    module = room_place_module

    draw_calls = []
    monkeypatch.setattr(module, "_draw_double_bed", lambda *args, **kwargs: draw_calls.append("double_bed"))
    monkeypatch.setattr(module, "_draw_single_bed", lambda *args, **kwargs: draw_calls.append("single_bed"))
    monkeypatch.setattr(module, "_compute_bed_box", lambda *args, **kwargs: {"bbox": (2, 2, 4, 5), "head_side": "top"})
    monkeypatch.setattr(module, "_bed_wall_from_bbox", lambda *args, **kwargs: "left")
    monkeypatch.setattr(module, "_single_bed_reference_size", lambda *args, **kwargs: (1.5, 3.0))
    monkeypatch.setattr(module, "_dims_from_area_ratio", lambda area, ratio_wh: (1.0, 0.5))
    monkeypatch.setattr(module, "_bbox_polygon", lambda x0, y0, x1, y1: box(x0, y0, x1, y1))
    monkeypatch.setattr(module, "_is_valid_piece", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "_draw_piece_by_name", lambda _msp, name, bbox: draw_calls.append(name))

    module._place_bedroom(object(), box(0, 0, 10, 8), is_master=True, door_geom=box(0, 0, 0.5, 0.5))

    assert "double_bed" in draw_calls
    # Extra pieces are attempted when validation passes.
    assert any(name in draw_calls for name in {"closet", "dresser", "desk", "chair"})


def test_room_helpers_zone_and_scoring_apis(room_helpers_module):
    module = room_helpers_module

    living = box(0, 0, 10, 8)
    frame = module._living_local_frame(living)

    zone = module._end_zone_polygon(living, frame, axis="u", side=-1, depth=2.0, inset=0.2)
    assert not zone.is_empty

    bounds = module._local_bounds_of_geom(frame, zone)
    assert bounds is not None
    assert len(bounds) == 4

    dining_round = module._build_dining_round(frame, 0.0, 0.0, area_budget=6.0, clearance=0.2)
    dining_rect = module._build_dining_rect(frame, 0.0, 0.0, area_budget=6.0, clearance=0.2)
    assert len(dining_round) == 7
    assert len(dining_rect) == 7

    score = module._score_layout(
        [{"name": "dining_table_rect", "footprint": box(1, 1, 2, 2)}],
        [box(3, 1, 4, 2)],
        weight=1.0,
        diag=10.0,
    )
    assert score >= 0.0


