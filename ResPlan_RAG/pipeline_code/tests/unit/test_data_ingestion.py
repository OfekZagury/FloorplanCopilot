import pickle
import sqlite3

import networkx as nx
from shapely.geometry import box


def test_aspect_label_classifies_shapes(data_ingestion_module):
    aspect_label = data_ingestion_module.aspect_label

    assert aspect_label(box(0, 0, 4, 1)) == "elongated"
    assert aspect_label(box(0, 0, 2, 2)) == "square"
    assert aspect_label(box(0, 0, 2, 3)) is None

    class Degenerate:
        bounds = (0, 0, 0, 5)

    assert aspect_label(Degenerate()) is None


def test_create_apartment_description_rich_topology(data_ingestion_module, monkeypatch):
    module = data_ingestion_module

    g = nx.Graph()
    g.add_node("living", type="living")
    g.add_node("kitchen", type="kitchen")
    g.add_node("bed1", type="bedroom", geometry=box(0, 0, 5, 1))
    g.add_node("bed2", type="bedroom", geometry=box(0, 0, 6, 1))
    g.add_node("bath", type="bathroom")
    g.add_node("bal", type="balcony")
    g.add_node("storage", type="storage")
    g.add_node("front", type="front_door")

    g.add_edge("front", "living", type="via_door")
    g.add_edge("bath", "bed1", type="via_door")
    g.add_edge("bath", "living", type="via_door")
    g.add_edge("bal", "living", type="via_door")
    g.add_edge("bal", "bed1", type="via_door")
    g.add_edge("storage", "kitchen", type="via_door")

    monkeypatch.setattr(module, "plan_to_graph", lambda plan: g)

    plan = {"area": 120.5, "storage": box(0, 0, 1, 1)}
    text, metadata = module.create_apartment_description(plan)

    assert metadata == {"area": 120.5, "bedrooms": 2, "bathrooms": 1, "balconies": 1}

    assert "A residential apartment layout." in text
    assert "It includes 2 bedrooms, 1 bathrooms, 1 balconies." in text
    assert "open layout with the living room connected to the kitchen" in text
    assert "front door opens directly into the main living space" in text
    assert "A bedroom has an elongated layout." in text
    assert text.count("A bedroom has an elongated layout.") == 1
    assert "dedicated storage space accessible from the kitchen" in text
    assert "shared wrap-around balcony is accessible from both the living area and a bedroom" in text
    assert "One bathroom is en-suite, connected directly to a bedroom." in text
    assert "One bathroom is accessible from the main living area." in text

    assert "The living room opens onto a balcony." not in text
    assert "A bedroom has direct balcony access." not in text


def test_create_apartment_description_falls_back_when_graph_fails(data_ingestion_module, monkeypatch):
    module = data_ingestion_module

    def raise_graph(_):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(module, "plan_to_graph", raise_graph)

    text, metadata = module.create_apartment_description({"area": 42.0})

    assert text == "A residential apartment layout."
    assert metadata == {"area": 42.0, "bedrooms": 0, "bathrooms": 0, "balconies": 0}


def test_init_sqlite_db_creates_apartments_table(data_ingestion_module, monkeypatch, tmp_path):
    module = data_ingestion_module
    db_path = tmp_path / "ingestion_test.db"
    monkeypatch.setattr(module.config, "SQLITE_DB_PATH", str(db_path), raising=False)

    conn = module.init_sqlite_db()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='apartments'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


class FakeCollection:
    def __init__(self):
        self.add_calls = []

    def add(self, documents, metadatas, ids):
        self.add_calls.append(
            {
                "documents": list(documents),
                "metadatas": list(metadatas),
                "ids": list(ids),
            }
        )


class FakePersistentClient:
    def __init__(self, path, collection):
        self.path = path
        self._collection = collection
        self.collection_name = None

    def get_or_create_collection(self, name):
        self.collection_name = name
        return self._collection


def test_main_ingests_non_null_ids_and_persists_to_sqlite_and_chroma(
    data_ingestion_module, monkeypatch, tmp_path
):
    module = data_ingestion_module

    db_path = tmp_path / "apartments_raw.db"
    chroma_path = tmp_path / "chroma_db"
    dataset_path = tmp_path / "plans.pkl"

    plans = [
        {"id": 1, "area": 10.0},
        {"id": None, "area": 20.0},
        {"id": 2, "area": 30.0},
    ]
    dataset_path.write_bytes(pickle.dumps(plans))

    monkeypatch.setattr(module.config, "SQLITE_DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(module.config, "CHROMA_DB_PATH", str(chroma_path), raising=False)
    monkeypatch.setattr(module.config, "CHROMA_COLLECTION_NAME", "floorplans", raising=False)
    monkeypatch.setattr(module.config, "RESPLAN_DATA_PATH", str(dataset_path), raising=False)

    fake_collection = FakeCollection()

    def fake_client_factory(path):
        return FakePersistentClient(path=path, collection=fake_collection)

    monkeypatch.setattr(module.chromadb, "PersistentClient", fake_client_factory)
    monkeypatch.setattr(module, "tqdm", lambda items, **kwargs: items)
    monkeypatch.setattr(
        module,
        "create_apartment_description",
        lambda plan: (
            f"description-{plan['id']}",
            {"area": float(plan.get("area", 0.0)), "bedrooms": 0, "bathrooms": 0, "balconies": 0},
        ),
    )

    module.main()

    assert len(fake_collection.add_calls) == 1
    call = fake_collection.add_calls[0]
    assert call["ids"] == ["1", "2"]
    assert call["documents"] == ["description-1", "description-2"]

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT id, raw_data FROM apartments ORDER BY id").fetchall()
    assert [r[0] for r in rows] == [1, 2]
    restored_plan_1 = pickle.loads(rows[0][1])
    assert restored_plan_1["id"] == 1


def test_main_batches_every_100_records(data_ingestion_module, monkeypatch, tmp_path):
    module = data_ingestion_module

    db_path = tmp_path / "apartments_batch.db"
    dataset_path = tmp_path / "plans_batch.pkl"
    plans = [{"id": i, "area": float(i)} for i in range(1, 102)]
    dataset_path.write_bytes(pickle.dumps(plans))

    monkeypatch.setattr(module.config, "SQLITE_DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(module.config, "CHROMA_DB_PATH", str(tmp_path / "chroma_batch"), raising=False)
    monkeypatch.setattr(module.config, "CHROMA_COLLECTION_NAME", "floorplans", raising=False)
    monkeypatch.setattr(module.config, "RESPLAN_DATA_PATH", str(dataset_path), raising=False)

    fake_collection = FakeCollection()
    monkeypatch.setattr(
        module.chromadb,
        "PersistentClient",
        lambda path: FakePersistentClient(path=path, collection=fake_collection),
    )
    monkeypatch.setattr(module, "tqdm", lambda items, **kwargs: items)
    monkeypatch.setattr(
        module,
        "create_apartment_description",
        lambda plan: (
            f"desc-{plan['id']}",
            {"area": float(plan["area"]), "bedrooms": 0, "bathrooms": 0, "balconies": 0},
        ),
    )

    module.main()

    assert len(fake_collection.add_calls) == 2
    assert len(fake_collection.add_calls[0]["ids"]) == 100
    assert len(fake_collection.add_calls[1]["ids"]) == 1
    assert fake_collection.add_calls[0]["ids"][0] == "1"
    assert fake_collection.add_calls[0]["ids"][-1] == "100"
    assert fake_collection.add_calls[1]["ids"] == ["101"]

