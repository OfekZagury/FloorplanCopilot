import matplotlib.pyplot as plt
import copy

from ResPlan_RAG.pipeline_code.dxf_exporting.generate_auto_furnitue_items import generate_auto_furniture_items
from ResPlan_RAG.pipeline_code.retrieval_engine.interactive_cli import _run_mandatory_manual_editor
from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine_helpers import _collect_manual_filters, \
    _build_where_filter, _print_active_filters, search_vector_db, get_playground_from_sqlite, plot_top_candidates, \
    scale_apartment, TOP_N

# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan


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