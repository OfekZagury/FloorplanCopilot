import os
import pickle
import sqlite3
import chromadb
from tqdm import tqdm
import networkx as nx

# Import configuration and the utility functions from the provided file
import config
from resplan_utils import plan_to_graph


def create_apartment_description(plan):
    """
    Analyzes the apartment plan and its room graph to generate
    a semantic text description in Hebrew for the Vector DB,
    along with metadata.
    """
    # Build the topological graph of the apartment
    try:
        G = plan_to_graph(plan)
    except Exception as e:
        # Fallback to an empty graph if generation fails for a specific plan
        G = nx.Graph()

    area = plan.get('area', 0.0)

    # Count the number of rooms by type
    room_counts = {}
    for node, data in G.nodes(data=True):
        r_type = data.get("type")
        if r_type:
            room_counts[r_type] = room_counts.get(r_type, 0) + 1

    # Extract specific counts with defaults to 0
    bedrooms = room_counts.get('bedroom', 0)
    bathrooms = room_counts.get('bathroom', 0)
    balconies = room_counts.get('balcony', 0)
    has_living = 'living' in room_counts
    has_kitchen = 'kitchen' in room_counts

    # Construct a descriptive semantic text in Hebrew for embedding
    desc_parts = [f"Residential apartment with an area of {area:.1f} square meters."]

    if bedrooms > 0 or bathrooms > 0:
        rooms_text = "apartment includes"
        if bedrooms > 0:
            rooms_text += f" {bedrooms} bedrooms,"
        if bathrooms > 0:
            rooms_text += f" {bathrooms} bathrooms"
        if balconies > 0:
            rooms_text += f" ו-{balconies}  balconies"
        else:
            # Replace the last comma with a period
            rooms_text = rooms_text.rstrip(',') + "."
        desc_parts.append(rooms_text)

    if has_living and has_kitchen:
        desc_parts.append("The apartment has a living room connected to the kitchen area.")
    elif has_living:
        desc_parts.append("The apartment has a central living room area.")

    semantic_text = " ".join(desc_parts)

    # Prepare strictly structured metadata for hard filtering
    metadata = {
        "area": float(area),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "balconies": int(balconies)
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
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS apartments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       raw_data
                       BLOB
                   )
                   ''')
    conn.commit()
    return conn


def main():
    print("Initializing databases...")

    # 1. Initialize SQLite (Document store for raw data)
    sqlite_conn = init_sqlite_db()
    sqlite_cursor = sqlite_conn.cursor()

    # 2. Initialize ChromaDB (Vector store for semantic search)
    # This uses the persistent client which saves data to the local disk
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # For POC, we use Chroma's default embedding function.
    # To use Gemini embeddings later, you would pass an embedding_function argument here.
    collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    print(f"Loading ResPlan dataset from {config.RESPLAN_DATA_PATH}...")
    with open(config.RESPLAN_DATA_PATH, 'rb') as f:
        all_plans = pickle.load(f)

    print(f"Loaded {len(all_plans)} plans. Starting ingestion pipeline...")

    # We use batching for ChromaDB to optimize performance
    batch_size = 100
    docs_batch = []
    metadatas_batch = []
    ids_batch = []

    # Iterate with a progress bar
    for plan in tqdm(all_plans, desc="Processing Plans"):
        # The dataset provides an 'id', we use it to link both databases
        plan_id = plan.get('id')
        if plan_id is None:
            continue

        str_id = str(plan_id)

        # ---------------------------------------------------------
        # Step A: Process and save the raw geometry object to SQLite
        # ---------------------------------------------------------
        # Compress the entire dictionary into binary format using pickle
        pickled_plan = pickle.dumps(plan)
        sqlite_cursor.execute('''
            INSERT OR REPLACE INTO apartments (id, raw_data)
            VALUES (?, ?)
        ''', (plan_id, pickled_plan))

        # ---------------------------------------------------------
        # Step B: Generate description and metadata for ChromaDB
        # ---------------------------------------------------------
        description, metadata = create_apartment_description(plan)

        docs_batch.append(description)
        metadatas_batch.append(metadata)
        ids_batch.append(str_id)

        # Insert into ChromaDB in batches
        if len(docs_batch) >= batch_size:
            collection.add(
                documents=docs_batch,
                metadatas=metadatas_batch,
                ids=ids_batch
            )
            # Clear batches
            docs_batch, metadatas_batch, ids_batch = [], [], []

    # Add any remaining documents in the final batch
    if docs_batch:
        collection.add(
            documents=docs_batch,
            metadatas=metadatas_batch,
            ids=ids_batch
        )

    # Commit and close SQLite connection
    sqlite_conn.commit()
    sqlite_conn.close()

    print("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()