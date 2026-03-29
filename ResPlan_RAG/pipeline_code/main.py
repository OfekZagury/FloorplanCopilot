from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine import retrieve_apartment
from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporter import export_to_dxf


def run_architectural_pipeline():
    print("--- Starting ResPlan Architectural Pipeline ---")

    # ==========================================
    # Phase 1 & 2: Retrieval and Scaling
    # ==========================================
    # retrieve_apartment() handles:
    #   - RAG vector search over the plan database
    #   - Interactive top-10 candidate display and user selection
    #   - Parametric area scaling to the user's target size
    #   - Auto furniture generation followed by GUI editing
    user_query = input("Describe the apartment layout you are looking for: ")
    playground_data = retrieve_apartment(user_query)

    if not playground_data:
        print("No suitable apartment found or process aborted. Exiting.")
        return

    current_area = playground_data.get('area', 'Unknown')
    print(f"\n[Pipeline] Proceeding with apartment — area: {current_area} sqm.")

    # ==========================================
    # Phase 3: DXF Export
    # ==========================================
    # Export uses the user-confirmed edited items from the auto-seeded editor.
    filename = "final_architectural_design.dxf"
    print(f"\n--- Generating DXF File: {filename} ---")

    export_to_dxf(playground_data, output_path=filename)

    print("\n=======================================================")
    print(f"SUCCESS! You can now open '{filename}' in AutoCAD.")
    print("=======================================================")


if __name__ == "__main__":
    run_architectural_pipeline()