import os
import sqlite3
import chromadb
import pickle
import matplotlib.pyplot as plt
import json
import google.genai as genai
from openai import OpenAI


# Import our configuration file
import config
# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))



def refine_search_parameters(original_query, rejected_description, user_feedback):
    """
    Uses the LLM to understand the user's feedback and translate it into:
    1. A refined semantic text query.
    2. A strict metadata filter for ChromaDB (e.g., {"bathrooms": {"$lt": 4}}).
    """
    print("[LLM] Analyzing feedback and reformulating query...")

    prompt = f"""
    You are an AI assistant helping to refine a search for an apartment floorplan.

    The user originally searched for: "{original_query}"
    The system provided an apartment with this description: "{rejected_description}"
    The user rejected it and gave this feedback: "{user_feedback}"

    Task:
    1. Formulate a new, complete semantic search query in Hebrew based on the feedback.
    2. Create a metadata filter for ChromaDB if the user asked for numerical constraints (more/less rooms). 
       Available metadata keys are: 'bedrooms', 'bathrooms', 'balconies', 'area'.
       Allowed ChromaDB operators: $eq, $ne, $gt, $gte, $lt, $lte.

    Respond ONLY with a valid JSON object in the following format (no markdown tags):
    {{
        "refined_query": "the complete new hebrew search string",
        "metadata_filter": {{"bathrooms": {{"$lt": 4}}}} 
    }}
    Note: If no numerical filter is needed, output null for metadata_filter.
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    try:
        # Parse the JSON response
        raw_text = (response.choices[0].message.content or "").strip()
        # Remove markdown formatting if the model added it
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()

        result = json.loads(raw_text)
        return result.get("refined_query", original_query), result.get("metadata_filter")
    except Exception as e:
        print(f"Failed to parse LLM refinement. Fallback to basic search. Error: {e}")
        # Fallback in case of failure: just append the feedback to the query
        return f"{original_query} {user_feedback}", None


def search_vector_db(query_text, where_filter=None, n_results=1):
    """
    Searches ChromaDB for the most semantically similar apartment description.
    Applies strict metadata filtering if 'where_filter' is provided.
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    # Build the query arguments dynamically
    query_kwargs = {
        "query_texts": [query_text],
        "n_results": n_results
    }

    # Add the hard filter if the LLM generated one
    if where_filter:
        query_kwargs["where"] = where_filter
        print(f"[Vector DB] Applying hard filter: {where_filter}")

    # Perform the search
    results = collection.query(**query_kwargs)

    if not results['ids'][0]:
        print("No matching apartments found in the Vector DB.")
        return None, None

    best_match_id = results['ids'][0][0]
    matched_description = results['documents'][0][0]

    return int(best_match_id), matched_description


def get_playground_from_sqlite(apartment_id):
    """
    Fetches the raw, complete geometry dictionary from SQLite using the apartment ID.
    """
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT raw_data FROM apartments WHERE id = ?", (apartment_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return pickle.loads(row[0])


def retrieve_apartment(initial_query):
    """
    Main orchestration function with user confirmation and LLM-powered feedback loop.
    """
    attempts = 0
    max_attempts = 3
    current_query = initial_query
    current_filter = None  # Starts with no metadata filter

    while attempts < max_attempts:
        attempts += 1
        print(f"\n--- Attempt {attempts}/{max_attempts} ---")
        print(f"Searching for: '{current_query}'...")

        # Step 1: Semantic Search with optional filter
        apartment_id, description = search_vector_db(current_query, current_filter)

        if not apartment_id:
            print("Could not find a matching apartment.")
            if attempts < max_attempts:
                current_query = input("Please enter a new prompt to try again: ").strip()
                current_filter = None  # Reset filter on completely new prompt
                continue
            else:
                break

        # Step 2: Fetch the full Playground object
        playground_data = get_playground_from_sqlite(apartment_id)

        if not playground_data:
            print("Error retrieving data for the matched apartment.")
            break

        print(f"[SQLite] Successfully retrieved full geometric data for apartment {apartment_id}.")

        # Step 3: Present and Confirm
        print(f"\n[Vector DB] The selected apartment's description is:\n'{description}'\n")

        print("[ResPlan Utils] Plotting the plan. A window should open...")
        plot_plan(playground_data)
        plt.show()

        # Loop for user confirmation input
        is_accepted = False
        while True:
            confirmation = input("Does this apartment meet your requirements? (yes/no): ").strip().lower()

            if confirmation == 'yes':
                print("\nPerfect! Proceeding with this apartment plan.")
                is_accepted = True
                break
            elif confirmation == 'no':
                break  # Break inner loop, move to feedback collection
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        if is_accepted:
            return playground_data

        # If user said 'no', collect feedback and use LLM to refine
        if attempts < max_attempts:
            feedback = input("\nWhat would you like to change? (e.g., 'I want fewer bathrooms'): ").strip()

            # Call our helper function to get the new query and metadata filter
            current_query, current_filter = refine_search_parameters(
                original_query=current_query,
                rejected_description=description,
                user_feedback=feedback
            )

    print("\nExceeded maximum attempts to find a suitable apartment. Exiting.")
    return None




if __name__ == "__main__":
    # In a real application, you'd get this from your main.py / chatbot interface
    first_query = input("Describe the apartment you're looking for: ").strip()

    if first_query:
        playground = retrieve_apartment(first_query)

        if playground:
            print("\n--- Playground Data Successfully Retrieved ---")
            print(f"Total Area: {playground.get('area', 'N/A')}")