import os
import json
from openai import OpenAI
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))


def summarize_playground_for_llm(playground):
    """
    Converts the GeoJSON/Shapely geometries in the playground dictionary
    into a simplified JSON-friendly format (bounding boxes and areas)
    so the LLM can understand the spatial context accurately.
    """
    spatial_context = {
        "total_area": playground.get("area", 0),
        "rooms": {}
    }

    # Helper function to convert GeoJSON dict to Shapely geometry
    def to_shapely(geom_data):
        if isinstance(geom_data, BaseGeometry):
            return geom_data
        elif isinstance(geom_data, dict) and 'type' in geom_data and 'coordinates' in geom_data:
            return shape(geom_data)
        return None

    room_types = ['bedroom', 'bathroom', 'living', 'kitchen', 'balcony']
    for rt in room_types:
        geom_data = playground.get(rt)
        if not geom_data:
            continue

        geom = to_shapely(geom_data)
        if not geom or geom.is_empty:
            continue

        # Handle multiple rooms of the same type (MultiPolygon)
        if geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
            for i, polygon in enumerate(geom.geoms):
                spatial_context["rooms"][f"{rt}_{i + 1}"] = {
                    "bounds": polygon.bounds,  # (minx, miny, maxx, maxy)
                    "area": polygon.area
                }
        else:
            # Single room
            spatial_context["rooms"][rt] = {
                "bounds": geom.bounds,
                "area": geom.area
            }

    return json.dumps(spatial_context, indent=2)


def generate_architectural_plan(playground, user_request):
    """
    Constructs the prompt with the Playground context and calls the LLM.
    """
    playground_json = summarize_playground_for_llm(playground)

    system_instruction = """
    You are an expert architectural AI assistant. 
    I am providing you with a 'Playground' context, which contains the exact bounding boxes and areas of the rooms in a retrieved floorplan.

    CRITICAL INSTRUCTIONS:
    1. Use the provided Playground JSON strictly as your binding context.
    2. I want you to leave a placeholder for any final result or subjective design choice that I (the architect) need to decide on.
    3. You must invent your own data (like coordinates for furniture, electrical outlets, etc.) STRICTLY in accordance with the physical boundaries defined in the Playground context. Do not place items outside room bounds.
    4. You must understand the arguments expected by typical CAD drafting functions (e.g., x, y coordinates, length, width, rotation) and invent values accordingly.

    Output your plan as a structured JSON containing a list of objects to place, their exact invented coordinates based on the bounds, and explicitly use "PLACEHOLDER" for values the architect should fill.
    """

    prompt = f"""
    User Request: {user_request}

    Current Playground Spatial Data (Limits & Bounds):
    {playground_json}

    Please generate the interior placement plan according to the system instructions.
    """

    print("Calling OpenAI API to generate the design plan...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": system_instruction},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine import retrieve_apartment

    test_query = "4-room apartment with a living room open to the kitchen and a balcony"
    print("1. Retrieving Playground...")
    playground_data = retrieve_apartment(test_query)

    if playground_data:
        architect_request = "I want to place a double bed in each of the bedrooms, and a power point for a TV screen in the living room."
        print("\n2. Sending context and request to LLM...")
        llm_output = generate_architectural_plan(playground_data, architect_request)

        print("\n--- LLM Output ---")
        print(llm_output)