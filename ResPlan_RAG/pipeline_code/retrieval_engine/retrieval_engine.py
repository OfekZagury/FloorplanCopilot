import os
import sqlite3
import chromadb
import pickle
import matplotlib.pyplot as plt
import json
import math
import copy
from openai import OpenAI
from shapely.affinity import scale as shapely_scale
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape, mapping

# Import our configuration file
from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.dxf_exporting.generate_auto_furnitue_items import generate_auto_furniture_items

from ResPlan_RAG.pipeline_code.retrieval_engine.interactive_cli import _run_mandatory_manual_editor
# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))

# How many top candidates to retrieve and show each round
TOP_N = 10


def scale_apartment(playground: dict, target_area_sqm: float) -> dict:
    """
    Returns a scaled deep-copied apartment dictionary so its area matches target_area_sqm.
    Supports native Shapely geometries, GeoJSON dictionaries, and lists of GeoJSONs.
    """
    if target_area_sqm <= 0:
        raise ValueError("target_area_sqm must be a positive number.")

    current_area = float(playground.get('area', 1.0))
    if math.isclose(target_area_sqm, current_area, rel_tol=1e-9, abs_tol=1e-9):
        return playground

    if current_area <= 0:
        raise ValueError("playground['area'] must be a positive number.")

    scale_factor = math.sqrt(target_area_sqm / current_area)

    def to_shapely(geom_data):
        if isinstance(geom_data, BaseGeometry):
            return geom_data
        elif isinstance(geom_data, dict) and 'type' in geom_data and 'coordinates' in geom_data:
            return shape(geom_data)
        return None

    wall_geom = to_shapely(playground.get('wall'))

    if wall_geom:
        min_x, min_y, _, _ = wall_geom.bounds
    else:
        min_x = None
        min_y = None
        for value in playground.values():
            geom = to_shapely(value)
            if geom:
                gx_min, gy_min, _, _ = geom.bounds
                min_x = gx_min if min_x is None else min(min_x, gx_min)
                min_y = gy_min if min_y is None else min(min_y, gy_min)

        if min_x is None or min_y is None:
            raise ValueError("No geometry objects found in playground for scaling.")

    scaled_playground = copy.deepcopy(playground)

    for key, value in scaled_playground.items():
        if isinstance(value, list):
            new_list = []
            for item in value:
                geom = to_shapely(item)
                if geom:
                    scaled_geom = shapely_scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(min_x, min_y))
                    new_list.append(mapping(scaled_geom) if isinstance(item, dict) else scaled_geom)
                else:
                    new_list.append(item)
            scaled_playground[key] = new_list
        else:
            geom = to_shapely(value)
            if geom:
                scaled_geom = shapely_scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(min_x, min_y))
                scaled_playground[key] = mapping(scaled_geom) if isinstance(value, dict) else scaled_geom

    scaled_playground['area'] = float(target_area_sqm)
    return scaled_playground


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
    1. Formulate a new, complete semantic search query in English based on the feedback.
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
        raw_text = (response.choices[0].message.content or "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()

        result = json.loads(raw_text)
        return result.get("refined_query", original_query), result.get("metadata_filter")
    except Exception as e:
        print(f"Failed to parse LLM refinement. Fallback to basic search. Error: {e}")
        return f"{original_query} {user_feedback}", None

def _prompt_optional_int(label, min_value=0, default=None):
    """
    Prompt for an optional integer. Enter keeps default (or None if no default).
    """
    while True:
        default_hint = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{default_hint} (press Enter to skip): ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            value = int(raw)
            if value >= min_value:
                return value
        print(f"Please enter an integer >= {min_value}, or press Enter to skip.")


def _collect_manual_filters(existing=None):
    """
    Collect hard numeric constraints from user (no LLM parsing).
    Only exact filters for bedrooms/bathrooms/balconies.
    """
    existing = existing or {}
    print("\n--- Optional Hard Filters (Exact Match) ---")
    print("Leave blank for any field you do not want to constrain.")

    bedrooms = _prompt_optional_int("Exact bedrooms", min_value=0, default=existing.get("bedrooms"))
    bathrooms = _prompt_optional_int("Exact bathrooms", min_value=0, default=existing.get("bathrooms"))
    balconies = _prompt_optional_int("Exact balconies", min_value=0, default=existing.get("balconies"))

    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balconies": balconies,
    }


def _build_where_filter(filter_values):
    """
    Build ChromaDB where filter from numeric constraints.
    No area filter by design.
    """
    clauses = []

    bedrooms = filter_values.get("bedrooms")
    bathrooms = filter_values.get("bathrooms")
    balconies = filter_values.get("balconies")

    if bedrooms is not None:
        clauses.append({"bedrooms": {"$eq": int(bedrooms)}})
    if bathrooms is not None:
        clauses.append({"bathrooms": {"$eq": int(bathrooms)}})
    if balconies is not None:
        clauses.append({"balconies": {"$eq": int(balconies)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _print_active_filters(where_filter):
    if where_filter:
        print(f"[Filter] Active hard filter: {where_filter}")
    else:
        print("[Filter] No hard filters active.")

def search_vector_db(query_text, where_filter=None, n_results=TOP_N):
    """
    Searches ChromaDB for the top-N most semantically similar apartment descriptions.
    Returns a list of (apartment_id, description) tuples, ordered by similarity.
    Applies strict metadata filtering if 'where_filter' is provided.
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    query_kwargs = {
        "query_texts": [query_text],
        "n_results": n_results
    }

    if where_filter:
        query_kwargs["where"] = where_filter
        print(f"[Vector DB] Applying hard filter: {where_filter}")

    results = collection.query(**query_kwargs)

    if not results['ids'][0]:
        print("No matching apartments found in the Vector DB.")
        return []

    candidates = [
        (int(apt_id), description)
        for apt_id, description in zip(results['ids'][0], results['documents'][0])
    ]

    return candidates


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


def plot_top_candidates(candidates_data):
    """
    Plots all candidate apartments in a single figure as a grid of subplots.
    Each subplot is clearly labelled with its selection number so the user
    knows exactly what to type to pick it.

    Args:
        candidates_data: list of (rank, apartment_id, description, playground_dict)
    """
    n = len(candidates_data)
    if n == 0:
        return

    # Build a roughly-square grid (e.g. 10 plans → 2 rows × 5 cols)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5 + 1))
    fig.suptitle(
        "Top Matching Apartment Plans\nType the option number in the terminal to select",
        fontsize=13, fontweight='bold'
    )

    # Flatten axes array for easy indexing (works for 1-row and multi-row cases)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for rank, apt_id, description, playground in candidates_data:
        ax = axes_flat[rank - 1]

        area = playground.get('area', 'N/A')
        area_str = f"{area:.1f} sqm" if isinstance(area, (int, float)) else str(area)

        # Build the subplot title — rank number is prominent so it's easy to read
        subplot_title = f"Option {rank}  ·  {area_str}"

        try:
            # Pass ax and title directly so plot_plan draws into the right subplot
            plot_plan(playground, ax=ax, title=subplot_title, legend=False, tight=False)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot error:\n{e}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='red')
            ax.set_title(subplot_title, fontsize=9, fontweight='bold')

        # Draw a bold coloured border around the axes so each option feels like a card
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a7bd5')
            spine.set_linewidth(2.5)

        # Add a prominent option-number badge in the top-left corner of the plot
        ax.text(
            0.02, 0.97, f"#{rank}",
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#3a7bd5', edgecolor='none', alpha=0.85)
        )

    # Hide any unused subplots in the grid
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def retrieve_apartment(initial_query):
    """
    Main orchestration function.
    Each attempt retrieves the top-N candidates, plots them in a grid,
    and lets the user pick one by number (or refine search).
    Uses deterministic hard filters (manual numeric input), no LLM translation.
    Flow: select apartment -> auto item generation -> GUI edit -> final export data.
    """

    attempts = 0
    max_attempts = 3
    current_query = initial_query

    # Collect hard numeric constraints once before first attempt
    filter_values = _collect_manual_filters()
    current_filter = _build_where_filter(filter_values)

    while attempts < max_attempts:
        attempts += 1
        print(f"\n--- Attempt {attempts}/{max_attempts} ---")
        print(f"Searching for: '{current_query}'")
        _print_active_filters(current_filter)

        # Step 1: Retrieve top-N candidates from ChromaDB
        candidates = search_vector_db(current_query, current_filter, n_results=TOP_N)

        if not candidates:
            print("Could not find any matching apartments with current query/filters.")
            if attempts < max_attempts:
                action = input("Choose: [E]dit filters, [C]lear filters, [Q]uery-only retry (keep filters): ").strip().lower()

                if action == "e":
                    filter_values = _collect_manual_filters(existing=filter_values)
                    current_filter = _build_where_filter(filter_values)
                elif action == "c":
                    filter_values = {"bedrooms": None, "bathrooms": None, "balconies": None}
                    current_filter = None

                new_query = input("Enter a new search prompt (or press Enter to keep current): ").strip()
                if new_query:
                    current_query = new_query
                continue
            else:
                break

        # Step 2: Fetch full geometry for every candidate
        print(f"[SQLite] Fetching geometric data for {len(candidates)} candidates...")
        candidates_data = []  # (rank, apt_id, description, playground)
        for rank, (apt_id, description) in enumerate(candidates, start=1):
            playground = get_playground_from_sqlite(apt_id)
            if playground is None:
                print(f"  [Warning] Could not fetch data for apartment {apt_id}, skipping.")
                continue
            candidates_data.append((rank, apt_id, description, playground))

        if not candidates_data:
            print("Failed to retrieve geometry for all candidates.")
            break

        # Step 3: Show all plans in one grid figure
        print(f"\n[ResPlan Utils] Plotting {len(candidates_data)} plans. A window should open...")
        print("[Info] Apartment size can always be adjusted after you pick a plan.")
        plot_top_candidates(candidates_data)

        # Print summary table in console
        print("\n--- Candidate Summary ---")
        for rank, apt_id, description, playground in candidates_data:
            area = playground.get('area', 'N/A')
            area_str = f"{area:.1f} sqm" if isinstance(area, (int, float)) else str(area)
            print(f"  #{rank:>2}  ID {apt_id:<8}  {area_str:<12}  {description}")
        print("-------------------------")

        # Step 4: Ask user to pick one (or reject all)
        selected_playground = None

        while True:
            choice = input(f"\nEnter the number of the plan you want (1-{len(candidates_data)}), "
                f"or press Enter to refine your search: ").strip()

            if choice == "":
                # User wants to search again
                break

            if choice.isdigit():
                choice_int = int(choice)
                if 1 <= choice_int <= len(candidates_data):
                    _, apt_id, _, selected_playground = candidates_data[choice_int - 1]
                    selected_playground["apartment_id"] = apt_id
                    print(f"\nYou selected plan #{choice_int} (ID: {apt_id}).")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(candidates_data)}.")
            else:
                print("Invalid input. Enter a number or press Enter to refine.")

        # If a plan was chosen, proceed to area scaling and auto-seeded manual editor
        if selected_playground is not None:
            final_playground = selected_playground
            while True:
                target_area_input = input("Enter desired apartment area in sqm (press Enter to keep original area): ").strip()

                if not target_area_input:
                    final_playground = selected_playground
                    break

                try:
                    target_area_sqm = float(target_area_input)
                    if target_area_sqm <= 0:
                        print("Please enter a positive number for area.")
                        continue
                    final_playground = scale_apartment(selected_playground, target_area_sqm)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value (e.g., 85.5), "
                          "or press Enter to keep the original area.")

            print("\n[Pipeline] Generating automatic furniture items for editing...")
            auto_items = generate_auto_furniture_items(final_playground)
            seeded_playground = copy.deepcopy(final_playground)
            seeded_playground["manual_furniture"] = {
                "enabled": True,
                "confirmed": True,
                "items": auto_items,
            }

            print(f"[Pipeline] Auto-generated {len(auto_items)} item(s). Opening editor for move/add/delete...")
            return _run_mandatory_manual_editor(seeded_playground)


        # User pressed Enter -> refine query and optionally filters
        if attempts < max_attempts:
            feedback = input("\nWhat would you like to change? (e.g., 'open kitchen, more privacy'): ").strip()
            if feedback:
                current_query = f"{current_query}. {feedback}"

            filter_action = input("Filter action: [K]eep, [E]dit, [C]lear: ").strip().lower()
            if filter_action == "e":
                filter_values = _collect_manual_filters(existing=filter_values)
                current_filter = _build_where_filter(filter_values)
            elif filter_action == "c":
                filter_values = {"bedrooms": None, "bathrooms": None, "balconies": None}
                current_filter = None
        else:
            print("\nNo more attempts remaining.")

    print("\nExceeded maximum attempts to find a suitable apartment. Exiting.")
    return None


if __name__ == "__main__":
    first_query = input("Describe the apartment you're looking for (note: apartment size can be adjusted after selection): ").strip()

    if first_query:
        playground = retrieve_apartment(first_query)

        if playground:
            print("\n--- Playground Data Successfully Retrieved ---")
            print(f"Total Area: {playground.get('area', 'N/A')}")
            print("\n[Verification] Plotting the selected apartment...")
            print("Tip: The shape will look the same. Check the X/Y axis values to see the new dimensions!")
            plot_plan(playground)
            plt.show()