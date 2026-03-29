import sqlite3
import chromadb
import pickle

# Import our configuration file
import config


def search_vector_db(query_text, n_results=1):
    """
    Searches ChromaDB for the most semantically similar apartment description.
    Returns the ID of the best match.
    """
    # Initialize the ChromaDB client pointing to our local folder
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    # Perform the semantic search
    # Note: We are using Chroma's default embedding function here,
    # which matches what we used during data ingestion.
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    # Check if we got any results
    if not results['ids'][0]:
        print("No matching apartments found in the Vector DB.")
        return None

    # Extract the ID of the top result
    best_match_id = results['ids'][0][0]

    # Optional: Print the matched description for debugging purposes
    matched_description = results['documents'][0][0]
    print(f"[Vector DB] Found match (ID: {best_match_id}): {matched_description}")

    return int(best_match_id)


def get_playground_from_sqlite(apartment_id):
    """
    Fetches the raw, complete geometry dictionary from SQLite using the apartment ID.
    This dictionary acts as the 'Playground' context for the LLM.
    """
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Query the blob data by ID
    cursor.execute("SELECT raw_data FROM apartments WHERE id = ?", (apartment_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        print(f"Error: Apartment ID {apartment_id} not found in SQLite DB.")
        return None

    # The data is in the first column of the row.
    # We use pickle.loads to convert the binary blob back into a live Python dictionary
    raw_blob = row[0]
    playground_dict = pickle.loads(raw_blob)

    return playground_dict


def retrieve_apartment(query_text):
    """
    Main orchestration function:
    1. Converts text to vector and finds the best ID.
    2. Retrieves the full geometric dictionary (Playground) from SQLite.
    """
    print(f"Searching for: '{query_text}'...")

    # Step 1: Semantic Search
    apartment_id = search_vector_db(query_text)

    if not apartment_id:
        return None

    # Step 2: Fetch the Playground object
    playground_data = get_playground_from_sqlite(apartment_id)

    print(f"[SQLite] Successfully retrieved full geometric data for apartment {apartment_id}.")
    return playground_data


# ==========================================
# Quick Test Block
# ==========================================
if __name__ == "__main__":
    # You can test the engine directly by running this file
    test_query = "דירת 4 חדרים עם סלון פתוח למטבח ומרפסת"

    playground = retrieve_apartment(test_query)

    if playground:
        print("\n--- Playground Data Overview ---")
        # Print some basic info to verify the dictionary is loaded correctly
        print(f"Total Area: {playground.get('area', 'N/A')}")

        # Verify that complex Shapely objects (like multi-polygons) are intact
        walls = playground.get('wall')
        if walls:
            print(f"Wall Geometry Type: {type(walls)}")