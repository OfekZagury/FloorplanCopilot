from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine import retrieve_apartment
from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporter import export_to_dxf

logger = config.logger


def run_architectural_pipeline():
    logger.info("--- Starting ResPlan Architectural Pipeline ---")

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
        logger.warning("No suitable apartment found or process aborted. Exiting.")
        return

    current_area = playground_data.get('area', 'Unknown')
    logger.info("[Pipeline] Proceeding with apartment - area: %s sqm.", current_area)

    # ==========================================
    # Phase 3: DXF Export
    # ==========================================
    # Export uses the user-confirmed edited items from the auto-seeded editor.
    filename = "final_architectural_design.dxf"
    logger.info("--- Generating DXF File: %s ---", filename)

    export_to_dxf(playground_data, output_path=filename)

    logger.info("=======================================================")
    logger.info("SUCCESS! You can now open '%s' in AutoCAD.", filename)
    logger.info("=======================================================")


if __name__ == "__main__":
    if getattr(config, "ENABLE_WEB_EDITOR", False):
        from ResPlan_RAG.pipeline_code.gradio_app import main as run_gradio_app
        run_gradio_app()
    else:
        run_architectural_pipeline()
