import sqlite3
import chromadb
import pickle
import matplotlib.pyplot as plt

# Import our configuration file
import config
# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan


def search_vector_db(query_text, n_results=1):
    """
    Searches ChromaDB for the most semantically similar apartment description.
    Returns the ID and the matched description of the best match.
    """
    # Initialize the ChromaDB client pointing to our local folder
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    # Perform the semantic search
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    # Check if we got any results
    if not results['ids'][0]:
        print("No matching apartments found in the Vector DB.")
        return None, None

    # Extract the ID and description of the top result
    best_match_id = results['ids'][0][0]
    matched_description = results['documents'][0][0]

    return int(best_match_id), matched_description


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


def retrieve_apartment(initial_query):
    """
    Main orchestration function with user confirmation loop.
    1. Converts text to vector and finds the best ID.
    2. Presents description and plot to user.
    3. Asks for confirmation. Retries with a new prompt if needed (up to 3 times).
    """
    attempts = 0
    max_attempts = 3
    current_query = initial_query

    while attempts < max_attempts:
        attempts += 1
        print(f"\n--- Attempt {attempts}/{max_attempts} ---")
        print(f"Searching for: '{current_query}'...")

        # Step 1: Semantic Search to get ID and Description
        apartment_id, description = search_vector_db(current_query)

        if not apartment_id:
            # If no results, prompt for a new query and try again
            print("Could not find a matching apartment.")
            if attempts < max_attempts:
                current_query = input("Please enter a new prompt to try again: ").strip()
                continue
            else:
                break

        # Step 2: Fetch the full Playground object
        playground_data = get_playground_from_sqlite(apartment_id)

        if not playground_data:
            # If data is missing (unexpected), prompt for new query
            print("Error retrieving data for the matched apartment.")
            if attempts < max_attempts:
                current_query = input("Please enter a new prompt to try again: ").strip()
                continue
            else:
                break

        print(f"[SQLite] Successfully retrieved full geometric data for apartment {apartment_id}.")

        # ==========================================
        # Step 3: Present and Confirm
        # ==========================================
        print(f"\n[Vector DB] The selected apartment's description is:\n'{description}'\n")

        print("[ResPlan Utils] Plotting the plan. A window should open...")
        # Plot the plan using the imported utility function
        plot_plan(playground_data)
        # Display the plot
        plt.show()

        # Loop for user confirmation input
        while True:
            confirmation = input("Does this apartment meet your requirements? (yes/no): ").strip().lower()

            if confirmation == 'yes':
                print("\nPerfect! Proceeding with this apartment plan.")
                return playground_data
            elif confirmation == 'no':
                break  # Exit the confirmation loop to retry with a new prompt
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # If we reach here, the user said 'no'
        print("Understood. Let's try to find a different apartment.")
        if attempts < max_attempts:
            current_query = input("Please enter a new prompt for a different search: ").strip()

    # If the loop completes without returning, we failed after all attempts
    print("\nExceeded maximum attempts to find a suitable apartment. Exiting.")
    return None


# ==========================================
# Quick Test Block
# ==========================================
if __name__ == "__main__":
    # In a real application, you'd get this from your main.py / chatbot interface
    first_query = input("Describe the apartment you're looking for: ").strip()

    if first_query:
        playground = retrieve_apartment(first_query)

        if playground:
            print("\n--- Playground Data Successfully Retrieved ---")
            print(f"Total Area: {playground.get('area', 'N/A')}")