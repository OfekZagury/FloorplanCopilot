import copy

import pytest


def test_search_candidates_for_web_builds_cards_and_skips_missing(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    monkeypatch.setattr(module, "search_vector_db", lambda q, where_filter, n_results: [(10, "d1"), (20, "d2")])

    def fake_get_playground(apt_id):
        if apt_id == 20:
            return None
        return {"area": 50.0}

    captured = []

    def fake_build_card(rank, apt_id, desc, playground):
        captured.append((rank, apt_id, desc, playground.get("apartment_id")))
        return {"apartment_id": apt_id, "rank": rank}

    monkeypatch.setattr(module, "get_playground_from_sqlite", fake_get_playground)
    monkeypatch.setattr(module, "build_candidate_card", fake_build_card)

    cards = module.search_candidates_for_web("query", bedrooms=3, bathrooms=2, balconies=0, n_results=10)

    assert cards == [{"apartment_id": 10, "rank": 1}]
    assert captured == [(1, 10, "d1", 10)]


def test_prepare_selected_playground_for_web_scales_and_seeds_manual_items(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    candidate_card = {"playground": {"area": 100.0, "wall": "w"}}

    monkeypatch.setattr(module, "scale_apartment", lambda pg, area: {"area": area, "scaled": True})
    monkeypatch.setattr(module, "generate_auto_furniture_items", lambda pg: [{"icon": "single_bed"}])

    out = module.prepare_selected_playground_for_web(candidate_card, target_area_sqm=144.0)

    assert out["area"] == 144.0
    assert out["scaled"] is True
    assert out["manual_furniture"]["enabled"] is True
    assert out["manual_furniture"]["confirmed"] is True
    assert out["manual_furniture"]["items"] == [{"icon": "single_bed"}]


def test_prepare_selected_playground_for_web_keeps_original_immutable(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    base = {"area": 100.0, "nested": {"x": 1}}
    card = {"playground": base}

    monkeypatch.setattr(module, "generate_auto_furniture_items", lambda pg: [])

    out = module.prepare_selected_playground_for_web(card, target_area_sqm=None)

    assert out is not base
    assert out["nested"] is not base["nested"]
    assert base == {"area": 100.0, "nested": {"x": 1}}


def test_prepare_selected_playground_for_web_requires_candidate(retrieval_engine_module):
    module = retrieval_engine_module
    with pytest.raises(ValueError, match="No candidate card"):
        module.prepare_selected_playground_for_web({})


def test_retrieve_apartment_happy_path_with_editor_return(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    selected_playground = {"area": 90.0, "wall": "w"}

    monkeypatch.setattr(module, "_collect_manual_filters", lambda existing=None: {"bedrooms": 3, "bathrooms": 2, "balconies": 0})
    monkeypatch.setattr(module, "_build_where_filter", lambda fv: {"$and": []})
    monkeypatch.setattr(module, "_print_active_filters", lambda wf: None)
    monkeypatch.setattr(module, "search_vector_db", lambda q, f, n_results: [(5, "candidate description")])
    monkeypatch.setattr(module, "get_playground_from_sqlite", lambda apt_id: copy.deepcopy(selected_playground))
    monkeypatch.setattr(module, "plot_top_candidates", lambda data: None)
    monkeypatch.setattr(module, "generate_auto_furniture_items", lambda pg: [{"icon": "fridge"}, {"icon": "sink"}])

    expected_editor_output = {"ok": True, "area": 90.0}
    monkeypatch.setattr(module, "_run_mandatory_manual_editor", lambda seeded_pg: expected_editor_output)

    answers = iter([
        "1",  # choose first candidate
        "",   # keep original area
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    result = module.retrieve_apartment("4 room apartment")

    assert result == expected_editor_output


def test_retrieve_apartment_retries_after_invalid_area_then_scales(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    monkeypatch.setattr(module, "_collect_manual_filters", lambda existing=None: {"bedrooms": None, "bathrooms": None, "balconies": None})
    monkeypatch.setattr(module, "_build_where_filter", lambda fv: None)
    monkeypatch.setattr(module, "_print_active_filters", lambda wf: None)
    monkeypatch.setattr(module, "search_vector_db", lambda q, f, n_results: [(8, "desc")])
    monkeypatch.setattr(module, "get_playground_from_sqlite", lambda apt_id: {"area": 80.0, "wall": "w"})
    monkeypatch.setattr(module, "plot_top_candidates", lambda data: None)

    calls = {"scaled": 0}

    def fake_scale(pg, target):
        calls["scaled"] += 1
        return {"area": target, "wall": "scaled"}

    monkeypatch.setattr(module, "scale_apartment", fake_scale)
    monkeypatch.setattr(module, "generate_auto_furniture_items", lambda pg: [])
    monkeypatch.setattr(module, "_run_mandatory_manual_editor", lambda seeded_pg: seeded_pg)

    answers = iter([
        "1",      # choose candidate
        "abc",    # invalid area
        "120.5",  # valid area
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    result = module.retrieve_apartment("query")

    assert calls["scaled"] == 1
    assert result["area"] == 120.5
    assert result["manual_furniture"]["enabled"] is True


def test_retrieve_apartment_returns_none_after_repeated_empty_search(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    monkeypatch.setattr(module, "_collect_manual_filters", lambda existing=None: {"bedrooms": None, "bathrooms": None, "balconies": None})
    monkeypatch.setattr(module, "_build_where_filter", lambda fv: None)
    monkeypatch.setattr(module, "_print_active_filters", lambda wf: None)
    monkeypatch.setattr(module, "search_vector_db", lambda q, f, n_results: [])

    answers = iter([
        "q",  # attempt 1 action
        "",   # keep query
        "c",  # attempt 2 action -> clear filters
        "new query",  # new query
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    out = module.retrieve_apartment("initial")
    assert out is None


def test_retrieve_apartment_breaks_when_all_candidate_geometries_missing(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    monkeypatch.setattr(module, "_collect_manual_filters", lambda existing=None: {"bedrooms": None, "bathrooms": None, "balconies": None})
    monkeypatch.setattr(module, "_build_where_filter", lambda fv: None)
    monkeypatch.setattr(module, "_print_active_filters", lambda wf: None)
    monkeypatch.setattr(module, "search_vector_db", lambda q, f, n_results: [(1, "d")])
    monkeypatch.setattr(module, "get_playground_from_sqlite", lambda apt_id: None)

    out = module.retrieve_apartment("query")
    assert out is None


def test_retrieve_apartment_handles_invalid_choice_then_refine_then_success(retrieval_engine_module, monkeypatch):
    module = retrieval_engine_module

    monkeypatch.setattr(module, "_collect_manual_filters", lambda existing=None: {"bedrooms": None, "bathrooms": None, "balconies": None})
    monkeypatch.setattr(module, "_build_where_filter", lambda fv: None)
    monkeypatch.setattr(module, "_print_active_filters", lambda wf: None)
    monkeypatch.setattr(module, "plot_top_candidates", lambda data: None)

    calls = {"n": 0}

    def fake_search(query, where_filter, n_results):
        calls["n"] += 1
        return [(3, "desc")]

    monkeypatch.setattr(module, "search_vector_db", fake_search)
    monkeypatch.setattr(module, "get_playground_from_sqlite", lambda apt_id: {"area": 70.0, "wall": "w"})
    monkeypatch.setattr(module, "generate_auto_furniture_items", lambda pg: [])
    monkeypatch.setattr(module, "_run_mandatory_manual_editor", lambda seeded_pg: seeded_pg)

    answers = iter([
        "99",          # invalid candidate number
        "",            # refine search
        "make bigger", # feedback
        "k",           # keep filters
        "1",           # choose on next attempt
        "",            # keep area
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    out = module.retrieve_apartment("query")
    assert out["manual_furniture"]["enabled"] is True
    assert calls["n"] >= 2


