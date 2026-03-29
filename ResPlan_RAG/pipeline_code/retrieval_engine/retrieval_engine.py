import matplotlib.pyplot as plt
import copy
from typing import Dict, Any, List, Optional

from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.dxf_exporting.generate_auto_furnitue_items import (
    generate_auto_furniture_items,
)
from ResPlan_RAG.pipeline_code.retrieval_engine.interactive_cli import (
    _run_mandatory_manual_editor,
)
from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine_helpers import (
    _collect_manual_filters,
    _build_where_filter,
    _print_active_filters,
    search_vector_db,
    get_playground_from_sqlite,
    plot_top_candidates,
    scale_apartment,
    TOP_N,
    build_candidate_card,
)

# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan

logger = config.logger


def search_candidates_for_web(
    query: str, bedrooms=None, bathrooms=None, balconies=None, n_results=TOP_N
) -> List[Dict[str, Any]]:
    """Return top candidates and rendered preview payloads for web UIs."""
    filter_values = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balconies": balconies,
    }
    where_filter = _build_where_filter(filter_values)
    candidates = search_vector_db(query, where_filter=where_filter, n_results=n_results)
    cards = []

    for rank, (apt_id, description) in enumerate(candidates, start=1):
        playground = get_playground_from_sqlite(apt_id)
        if playground is None:
            continue
        playground["apartment_id"] = apt_id
        cards.append(build_candidate_card(rank, apt_id, description, playground))

    return cards


def prepare_selected_playground_for_web(
    candidate_card: Dict[str, Any], target_area_sqm: Optional[float] = None
) -> Dict[str, Any]:
    """Scale selected candidate if needed and seed auto furniture for web editing."""
    if not candidate_card:
        raise ValueError("No candidate card provided")

    base_playground = copy.deepcopy(candidate_card["playground"])

    if target_area_sqm is not None and float(target_area_sqm) > 0:
        base_playground = scale_apartment(base_playground, float(target_area_sqm))

    auto_items = generate_auto_furniture_items(base_playground)
    base_playground["manual_furniture"] = {
        "enabled": True,
        "confirmed": True,
        "items": auto_items,
    }
    return base_playground


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
        logger.info("--- Attempt %s/%s ---", attempts, max_attempts)
        logger.info("Searching for: '%s'", current_query)
        _print_active_filters(current_filter)

        # Step 1: Retrieve top-N candidates from ChromaDB
        candidates = search_vector_db(current_query, current_filter, n_results=TOP_N)

        if not candidates:
            logger.warning(
                "Could not find any matching apartments with current query/filters."
            )
            if attempts < max_attempts:
                action = (
                    input(
                        "Choose: [E]dit filters, [C]lear filters, [Q]uery-only retry (keep filters): "
                    )
                    .strip()
                    .lower()
                )

                if action == "e":
                    filter_values = _collect_manual_filters(existing=filter_values)
                    current_filter = _build_where_filter(filter_values)
                elif action == "c":
                    filter_values = {
                        "bedrooms": None,
                        "bathrooms": None,
                        "balconies": None,
                    }
                    current_filter = None

                new_query = input(
                    "Enter a new search prompt (or press Enter to keep current): "
                ).strip()
                if new_query:
                    current_query = new_query
                continue
            else:
                break

        # Step 2: Fetch full geometry for every candidate
        logger.info(
            "[SQLite] Fetching geometric data for %s candidates...", len(candidates)
        )
        candidates_data = []  # (rank, apt_id, description, playground)
        for rank, (apt_id, description) in enumerate(candidates, start=1):
            playground = get_playground_from_sqlite(apt_id)
            if playground is None:
                logger.warning(
                    "Could not fetch data for apartment %s, skipping.", apt_id
                )
                continue
            candidates_data.append((rank, apt_id, description, playground))

        if not candidates_data:
            logger.error("Failed to retrieve geometry for all candidates.")
            break

        # Step 3: Show all plans
        logger.info(
            "[ResPlan Utils] Plotting %s plans. A window should open...",
            len(candidates_data),
        )
        logger.info(
            "[Info] Apartment size can always be adjusted after you pick a plan."
        )
        plot_top_candidates(candidates_data)

        # Step 4: Ask user to pick one (or reject all)
        selected_playground = None

        while True:
            choice = input(
                f"\nEnter the number of the plan you want (1-{len(candidates_data)}), "
                f"or press Enter to refine your search: "
            ).strip()

            if choice == "":
                # User wants to search again
                break

            if choice.isdigit():
                choice_int = int(choice)
                if 1 <= choice_int <= len(candidates_data):
                    _, apt_id, _, selected_playground = candidates_data[choice_int - 1]
                    selected_playground["apartment_id"] = apt_id
                    logger.info("You selected plan #%s (ID: %s).", choice_int, apt_id)
                    break
                else:
                    logger.warning(
                        "Please enter a number between 1 and %s.", len(candidates_data)
                    )
            else:
                logger.warning(
                    "Invalid input. Enter a number or press Enter to refine."
                )

        # If a plan was chosen, proceed to area scaling and auto-seeded manual editor
        if selected_playground is not None:
            final_playground = selected_playground
            while True:
                target_area_input = input(
                    "Enter desired apartment area in sqm (press Enter to keep original area): "
                ).strip()

                if not target_area_input:
                    final_playground = selected_playground
                    break

                try:
                    target_area_sqm = float(target_area_input)
                    if target_area_sqm <= 0:
                        logger.warning("Please enter a positive number for area.")
                        continue
                    final_playground = scale_apartment(
                        selected_playground, target_area_sqm
                    )
                    break
                except ValueError:
                    logger.warning(
                        "Invalid input. Please enter a numeric value (e.g., 85.5), or press Enter to keep the original area."
                    )

            logger.info(
                "[Pipeline] Generating automatic furniture items for editing..."
            )
            auto_items = generate_auto_furniture_items(final_playground)
            seeded_playground = copy.deepcopy(final_playground)
            seeded_playground["manual_furniture"] = {
                "enabled": True,
                "confirmed": True,
                "items": auto_items,
            }

            logger.info(
                "[Pipeline] Auto-generated %s item(s). Opening editor for move/add/delete...",
                len(auto_items),
            )
            return _run_mandatory_manual_editor(seeded_playground)

        # User pressed Enter -> refine query and optionally filters
        if attempts < max_attempts:
            feedback = input(
                "\nWhat would you like to change? (e.g., 'open kitchen, more privacy'): "
            ).strip()
            if feedback:
                current_query = f"{current_query}. {feedback}"

            filter_action = (
                input("Filter action: [K]eep, [E]dit, [C]lear: ").strip().lower()
            )
            if filter_action == "e":
                filter_values = _collect_manual_filters(existing=filter_values)
                current_filter = _build_where_filter(filter_values)
            elif filter_action == "c":
                filter_values = {"bedrooms": None, "bathrooms": None, "balconies": None}
                current_filter = None
        else:
            logger.info("No more attempts remaining.")

    logger.warning("Exceeded maximum attempts to find a suitable apartment. Exiting.")
    return None


if __name__ == "__main__":
    first_query = input(
        "Describe the apartment you're looking for (note: apartment size can be adjusted after selection): "
    ).strip()

    if first_query:
        playground = retrieve_apartment(first_query)

        if playground:
            logger.info("--- Playground Data Successfully Retrieved ---")
            logger.info("Total Area: %s", playground.get("area", "N/A"))
            logger.info("[Verification] Plotting the selected apartment...")
            logger.info(
                "Tip: The shape will look the same. Check the X/Y axis values to see the new dimensions!"
            )
            plot_plan(playground)
            plt.show()
