import json
import pickle
import sqlite3

import numpy as np
import pytest
from shapely.geometry import Polygon, box, mapping


class _FakeCollection:
    def __init__(self, query_result):
        self._query_result = query_result
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self._query_result


class _FakeChromaClient:
    def __init__(self, path, collection):
        self.path = path
        self._collection = collection
        self.last_collection_name = None

    def get_collection(self, name):
        self.last_collection_name = name
        return self._collection


def test_build_where_filter_exact_match_clauses(retrieval_helpers_module):
    module = retrieval_helpers_module

    assert module._build_where_filter({"bedrooms": None, "bathrooms": None, "balconies": None}) is None
    assert module._build_where_filter({"bedrooms": 3, "bathrooms": None, "balconies": None}) == {
        "bedrooms": {"$eq": 3}
    }
    assert module._build_where_filter({"bedrooms": 3, "bathrooms": 2, "balconies": 1}) == {
        "$and": [
            {"bedrooms": {"$eq": 3}},
            {"bathrooms": {"$eq": 2}},
            {"balconies": {"$eq": 1}},
        ]
    }


def test_scale_apartment_scales_geojson_and_updates_area(retrieval_helpers_module):
    module = retrieval_helpers_module

    original_wall = box(0, 0, 10, 10)
    original_bed = box(1, 1, 3, 3)
    playground = {
        "area": 100.0,
        "wall": mapping(original_wall),
        "bedroom": mapping(original_bed),
        "note": "keep",
    }

    scaled = module.scale_apartment(playground, 400.0)

    assert scaled["area"] == 400.0
    assert scaled["note"] == "keep"

    scaled_wall = module._to_shapely(scaled["wall"])
    scaled_bed = module._to_shapely(scaled["bedroom"])

    assert pytest.approx(scaled_wall.bounds[2], rel=1e-6) == 20.0
    assert pytest.approx(scaled_wall.bounds[3], rel=1e-6) == 20.0
    assert pytest.approx(scaled_bed.bounds[2], rel=1e-6) == 6.0
    assert pytest.approx(scaled_bed.bounds[3], rel=1e-6) == 6.0


def test_scale_apartment_validates_inputs(retrieval_helpers_module):
    module = retrieval_helpers_module

    with pytest.raises(ValueError, match="target_area_sqm"):
        module.scale_apartment({"area": 100.0}, 0)

    with pytest.raises(ValueError, match=r"playground\['area'\]"):
        module.scale_apartment({"area": 0.0}, 20.0)

    with pytest.raises(ValueError, match="No geometry objects"):
        module.scale_apartment({"area": 20.0, "x": 1}, 40.0)


def test_search_vector_db_applies_where_and_parses_ids(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    fake_collection = _FakeCollection(
        {
            "ids": [["10", "20"]],
            "documents": [["d1", "d2"]],
        }
    )
    fake_client = _FakeChromaClient(path="dummy", collection=fake_collection)

    monkeypatch.setattr(module.config, "CHROMA_DB_PATH", "db/path", raising=False)
    monkeypatch.setattr(module.config, "CHROMA_COLLECTION_NAME", "floorplans", raising=False)
    monkeypatch.setattr(module.chromadb, "PersistentClient", lambda path: fake_client)

    out = module.search_vector_db("query", where_filter={"bedrooms": {"$eq": 3}}, n_results=5)

    assert out == [(10, "d1"), (20, "d2")]
    assert fake_collection.query_calls[0]["query_texts"] == ["query"]
    assert fake_collection.query_calls[0]["n_results"] == 5
    assert fake_collection.query_calls[0]["where"] == {"bedrooms": {"$eq": 3}}


def test_search_vector_db_returns_empty_when_no_matches(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    fake_collection = _FakeCollection({"ids": [[]], "documents": [[]]})
    fake_client = _FakeChromaClient(path="dummy", collection=fake_collection)
    monkeypatch.setattr(module.chromadb, "PersistentClient", lambda path: fake_client)

    assert module.search_vector_db("query") == []


def test_get_playground_from_sqlite_roundtrip(retrieval_helpers_module, monkeypatch, tmp_path):
    module = retrieval_helpers_module

    db_path = tmp_path / "apartments.db"
    monkeypatch.setattr(module.config, "SQLITE_DB_PATH", str(db_path), raising=False)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE apartments (id INTEGER PRIMARY KEY, raw_data BLOB)")
    expected = {"id": 7, "area": 99.0}
    cur.execute("INSERT INTO apartments(id, raw_data) VALUES (?, ?)", (7, pickle.dumps(expected)))
    conn.commit()
    conn.close()

    assert module.get_playground_from_sqlite(7) == expected
    assert module.get_playground_from_sqlite(8) is None


def test_refine_search_parameters_parses_json_and_markdown_fence(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    payload = {
        "refined_query": "open kitchen layout",
        "metadata_filter": {"bathrooms": {"$lt": 3}},
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "gpt-5"
            return _Resp(fenced)

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(module, "client", _FakeClient())

    q, mf = module.refine_search_parameters("orig", "desc", "feedback")
    assert q == "open kitchen layout"
    assert mf == {"bathrooms": {"$lt": 3}}


def test_refine_search_parameters_fallback_on_parse_error(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    class _Msg:
        content = "not-json"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            return _Resp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(module, "client", _FakeClient())

    q, mf = module.refine_search_parameters("orig", "desc", "feedback")
    assert q == "orig feedback"
    assert mf is None


def test_build_candidate_card_counts_rooms_and_embeds_preview(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    monkeypatch.setattr(module, "build_candidate_preview_image", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))

    playground = {
        "area": 85.0,
        "bedroom": mapping(Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])),
        "bathroom": mapping(Polygon([(4, 0), (6, 0), (6, 2), (4, 2)])),
        "balcony": mapping(Polygon([(0, 4), (2, 4), (2, 6), (0, 6)])),
    }

    card = module.build_candidate_card(1, 111, "nice", playground)

    assert card["rank"] == 1
    assert card["apartment_id"] == 111
    assert card["area"] == 85.0
    assert card["bedrooms"] == 1
    assert card["bathrooms"] == 1
    assert card["balconies"] == 1
    assert isinstance(card["preview_image"], np.ndarray)


def test_prompt_optional_int_retries_until_valid(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module
    answers = iter(["abc", "-1", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    out = module._prompt_optional_int("Exact bedrooms", min_value=0, default=None)
    assert out == 2


def test_collect_manual_filters_reads_three_values(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module
    monkeypatch.setattr(module, "_prompt_optional_int", lambda label, min_value=0, default=None: 1 if "bedrooms" in label else 2)
    out = module._collect_manual_filters(existing={"bedrooms": 3})
    assert out == {"bedrooms": 1, "bathrooms": 2, "balconies": 2}


def test_scale_apartment_returns_same_object_when_area_unchanged(retrieval_helpers_module):
    module = retrieval_helpers_module
    pg = {"area": 100.0, "wall": box(0, 0, 10, 10)}
    out = module.scale_apartment(pg, 100.0)
    assert out is pg


def test_scale_apartment_scales_list_geometry_values(retrieval_helpers_module):
    module = retrieval_helpers_module
    pg = {
        "area": 25.0,
        "wall": box(0, 0, 5, 5),
        "windows": [mapping(box(1, 1, 2, 1.5)), "note"],
    }
    out = module.scale_apartment(pg, 100.0)
    scaled_window = module._to_shapely(out["windows"][0])
    assert pytest.approx(scaled_window.bounds[2], rel=1e-6) == 4.0
    assert out["windows"][1] == "note"


def test_extract_polygons_handles_polygon_multipolygon_and_none(retrieval_helpers_module):
    module = retrieval_helpers_module
    p1 = box(0, 0, 1, 1)
    p2 = box(2, 2, 3, 3)
    mp = module.MultiPolygon([p1, p2])

    assert len(module._extract_polygons(None)) == 0
    assert len(module._extract_polygons(p1)) == 1
    assert len(module._extract_polygons(mp)) == 2


def test_plot_top_candidates_draws_without_crashing(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module
    monkeypatch.setattr(module, "plot_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.plt, "show", lambda: None)

    module.plot_top_candidates([
        (1, 77, "a" * 700, {"area": 88.1}),
    ])


def test_build_candidate_preview_image_handles_plot_error_branch(retrieval_helpers_module, monkeypatch):
    module = retrieval_helpers_module

    def raise_plot(*args, **kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(module, "plot_plan", raise_plot)
    img = module.build_candidate_preview_image(1, 1, "desc", {"area": 44.0})
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


