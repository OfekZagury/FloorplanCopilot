import os
import sqlite3
import chromadb
import pickle
import matplotlib.pyplot as plt
import json
import textwrap
import math
import copy
from shapely.affinity import scale as shapely_scale
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape, mapping
from openai import OpenAI

from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan

# How many top candidates to retrieve and show each round
TOP_N = 10

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))


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
    Plots candidate apartments one-by-one in separate figures for better visibility.
    Also prints each apartment description inside the plot.

    Args:
        candidates_data: list of (rank, apartment_id, description, playground_dict)
    """
    if not candidates_data:
        return

    for rank, apt_id, description, playground in candidates_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        area = playground.get("area", "N/A")
        area_str = f"{area:.1f} sqm" if isinstance(area, (int, float)) else str(area)
        subplot_title = f"Option {rank}  ·  ID {apt_id}  ·  {area_str}"

        try:
            plot_plan(playground, ax=ax, title=subplot_title, legend=True, tight=False)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Plot error:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            ax.set_title(subplot_title, fontsize=11, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_edgecolor("#3a7bd5")
            spine.set_linewidth(2.5)

        ax.text(
            0.02,
            0.97,
            f"#{rank}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#3a7bd5",
                edgecolor="none",
                alpha=0.85,
            ),
        )

        desc = str(description).strip() if description is not None else "N/A"
        if len(desc) > 600:
            desc = desc[:597] + "..."
        wrapped_desc = textwrap.fill(desc, width=95)

        fig.suptitle(
            f"{wrapped_desc}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )


        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()