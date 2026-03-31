import pickle
import sqlite3
import chromadb
from tqdm import tqdm
import networkx as nx

# Import configuration and the utility functions from the provided file
from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.resplan_utils import plan_to_graph

logger = config.logger


def aspect_label(polygon):
    """Helper function to determine if a room is elongated or square based on its bounding box."""
    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny
    if w == 0 or h == 0:
        return None

    ratio = max(w, h) / min(w, h)
    if ratio > 2.0:
        return "elongated"
    if ratio < 1.3:
        return "square"
    return None  # neutral, don't add noise


def create_apartment_description(plan):
    """
    Generates a pure topological semantic description in ENGLISH for the Vector DB.
    Ignores area completely in the text to allow the RAG to focus on layout,
    since area is handled parametrically later.
    """
    try:
        G = plan_to_graph(plan)
    except Exception:
        logger.warning(
            "plan_to_graph failed for plan id=%s; using empty graph fallback.",
            plan.get("id"),
            exc_info=True,
        )
        G = nx.Graph()

    room_counts = {}
    for node, data in G.nodes(data=True):
        r_type = data.get("type")
        if r_type:
            room_counts[r_type] = room_counts.get(r_type, 0) + 1

    bedrooms = room_counts.get("bedroom", 0)
    bathrooms = room_counts.get("bathroom", 0)
    balconies = room_counts.get("balcony", 0)
    has_living = "living" in room_counts
    has_kitchen = "kitchen" in room_counts

    # Pure layout description
    desc_parts = ["A residential apartment layout."]

    rooms_info = []
    if bedrooms > 0:
        rooms_info.append(f"{bedrooms} bedrooms")
    if bathrooms > 0:
        rooms_info.append(f"{bathrooms} bathrooms")
    if balconies > 0:
        rooms_info.append(f"{balconies} balconies")

    if rooms_info:
        desc_parts.append("It includes " + ", ".join(rooms_info) + ".")

    if has_living and has_kitchen:
        desc_parts.append(
            "The design features an open layout with the living room connected to the kitchen."
        )
    elif has_living:
        desc_parts.append("The design includes a central living room.")

    # --- NEW 1: Entry Flow ---
    front_door_node = None
    for node, data in G.nodes(data=True):
        if data.get("type") == "front_door":
            front_door_node = node
            break

    if front_door_node:
        entry_connections = [
            G.nodes[neighbor].get("type") for neighbor in G.neighbors(front_door_node)
        ]
        if "living" in entry_connections:
            desc_parts.append(
                "The front door opens directly into the main living space."
            )
        elif "corridor" in entry_connections or "hall" in entry_connections:
            desc_parts.append(
                "The entrance features a formal entryway or corridor for added privacy."
            )
        else:
            desc_parts.append(
                "The apartment has a distinct entry area separated from the main living room."
            )

    # --- Suggestion 4: Shape compactness per room ---
    added_aspects = set()
    for node, data in G.nodes(data=True):
        if data.get("type") == "bedroom":
            geom = data.get("geometry")
            if geom:
                # Make sure aspect_label function is defined outside or imported
                label = aspect_label(geom)
                if label and label not in added_aspects:
                    desc_parts.append(f"A bedroom has an {label} layout.")
                    added_aspects.add(label)

    # --- NEW 2: Storage and Utility Presence ---
    storage_poly = plan.get("storage")
    if storage_poly and not storage_poly.is_empty:
        storage_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "storage"]
        if storage_nodes:
            storage_neighbors = [
                G.nodes[neighbor].get("type")
                for neighbor in G.neighbors(storage_nodes[0])
            ]
            if "kitchen" in storage_neighbors:
                desc_parts.append(
                    "There is a dedicated storage space accessible from the kitchen, functioning as a pantry."
                )
            else:
                desc_parts.append(
                    "The layout includes dedicated internal storage or closet space."
                )
        else:
            desc_parts.append("The layout includes dedicated internal storage space.")

    # --- NEW 3: Shared/Wrap-around Balconies ---
    shared_balcony_found = False
    balcony_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "balcony"]

    for b_node in balcony_nodes:
        connected_room_types = set(
            [G.nodes[neighbor].get("type") for neighbor in G.neighbors(b_node)]
        )
        if "living" in connected_room_types and "bedroom" in connected_room_types:
            desc_parts.append(
                "A shared wrap-around balcony is accessible from both the living area and a bedroom, creating an excellent continuous indoor-outdoor flow."
            )
            shared_balcony_found = True
            break

    # --- Suggestion 1: Graph-based room connectivity ---
    has_ensuite = False
    has_public_bath = False

    # If a shared balcony is found, we set these to True so we don't add redundant single-balcony sentences
    living_balcony = shared_balcony_found
    bedroom_balcony = shared_balcony_found

    for u, v, data in G.edges(data=True):
        edge_type = data.get("type")
        type_u = G.nodes[u].get("type")
        type_v = G.nodes[v].get("type")
        room_pair = {type_u, type_v}

        # Detect en-suite: bathroom connected directly to a bedroom
        if (
            room_pair == {"bathroom", "bedroom"}
            and edge_type == "via_door"
            and not has_ensuite
        ):
            desc_parts.append(
                "One bathroom is en-suite, connected directly to a bedroom."
            )
            has_ensuite = True

        # Detect shared/public bathroom: bathroom connected to living
        if (
            room_pair == {"bathroom", "living"}
            and edge_type == "via_door"
            and not has_public_bath
        ):
            desc_parts.append("One bathroom is accessible from the main living area.")
            has_public_bath = True

        # Balcony off living vs off bedroom
        if room_pair == {"balcony", "living"} and not living_balcony:
            desc_parts.append("The living room opens onto a balcony.")
            living_balcony = True

        if room_pair == {"balcony", "bedroom"} and not bedroom_balcony:
            desc_parts.append("A bedroom has direct balcony access.")
            bedroom_balcony = True

    semantic_text = " ".join(desc_parts)

    # Keep area strictly in metadata for the scaling math later
    metadata = {
        "area": float(plan.get("area", 0.0)),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "balconies": int(balconies),
    }

    return semantic_text, metadata


def init_sqlite_db():
    """
    Initializes the SQLite database and creates the table
    if it doesn't exist.
    """
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    # Create a table with two columns: the ID and the raw compressed object
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS apartments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       raw_data
                       BLOB
                   )
                   """)
    conn.commit()
    return conn


def main():
    logger.info("Initializing databases...")

    # 1. Initialize SQLite (Document store for raw data)
    sqlite_conn = init_sqlite_db()
    sqlite_cursor = sqlite_conn.cursor()

    # 2. Initialize ChromaDB (Vector store for semantic search)
    # This uses the persistent client which saves data to the local disk
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # For POC, we use Chroma's default embedding function.
    # To use Gemini embeddings later, you would pass an embedding_function argument here.
    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    logger.info("Loading ResPlan dataset from %s...", config.RESPLAN_DATA_PATH)
    with open(config.RESPLAN_DATA_PATH, "rb") as f:
        all_plans = pickle.load(f)

    logger.info("Loaded %s plans. Starting ingestion pipeline...", len(all_plans))

    # We use batching for ChromaDB to optimize performance
    batch_size = 100
    docs_batch = []
    metadatas_batch = []
    ids_batch = []

    skipped_without_id = 0

    # Iterate with a progress bar
    for plan in tqdm(all_plans, desc="Processing Plans"):
        # The dataset provides an 'id', we use it to link both databases
        plan_id = plan.get("id")
        if plan_id is None:
            skipped_without_id += 1
            continue

        str_id = str(plan_id)

        # ---------------------------------------------------------
        # Step A: Process and save the raw geometry object to SQLite
        # ---------------------------------------------------------
        # Compress the entire dictionary into binary format using pickle
        pickled_plan = pickle.dumps(plan)
        sqlite_cursor.execute(
            """
            INSERT OR REPLACE INTO apartments (id, raw_data)
            VALUES (?, ?)
        """,
            (plan_id, pickled_plan),
        )

        # ---------------------------------------------------------
        # Step B: Generate description and metadata for ChromaDB
        # ---------------------------------------------------------
        description, metadata = create_apartment_description(plan)

        docs_batch.append(description)
        metadatas_batch.append(metadata)
        ids_batch.append(str_id)

        # Insert into ChromaDB in batches
        if len(docs_batch) >= batch_size:
            logger.debug("Flushing Chroma batch with %s items.", len(docs_batch))
            collection.add(
                documents=docs_batch, metadatas=metadatas_batch, ids=ids_batch
            )
            # Clear batches
            docs_batch, metadatas_batch, ids_batch = [], [], []

    # Add any remaining documents in the final batch
    if docs_batch:
        logger.debug("Flushing final Chroma batch with %s items.", len(docs_batch))
        collection.add(documents=docs_batch, metadatas=metadatas_batch, ids=ids_batch)

    # Commit and close SQLite connection
    sqlite_conn.commit()
    sqlite_conn.close()

    if skipped_without_id:
        logger.warning("Skipped %s plans without an 'id' field.", skipped_without_id)
    logger.info("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
