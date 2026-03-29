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
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point

try:
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QPen, QColor, QBrush, QPolygonF
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QProgressDialog,
        QPushButton,
        QVBoxLayout,
    )
    HAS_QT = True
except Exception:
    HAS_QT = False


# Import our configuration file
import config
from dxf_exporter import generate_auto_furniture_items
# Import the image plotting utility
from ResPlan_RAG.resplan_dataset_files.resplan_utils import plot_plan

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))

# How many top candidates to retrieve and show each round
TOP_N = 10

MANUAL_FURNITURE_CATALOG = {
    "kitchen_sink": {"shape": "rect", "default_w": 14.0, "default_h": 8.0, "rooms": ["kitchen"]},
    "fridge": {"shape": "rect", "default_w": 10.0, "default_h": 14.0, "rooms": ["kitchen"]},
    "tv": {"shape": "rect", "default_w": 16.0, "default_h": 4.0, "rooms": ["living"]},
    "coffee_table": {"shape": "rect", "default_w": 12.0, "default_h": 8.0, "rooms": ["living"]},
    "dining_table_rect": {"shape": "rect", "default_w": 18.0, "default_h": 10.0, "rooms": ["living"]},
    "dining_table_round": {"shape": "circle", "default_r": 7.0, "rooms": ["living"]},
    "sofa": {"shape": "rect", "default_w": 24.0, "default_h": 10.0, "rooms": ["living"]},
    "rug": {"shape": "rect", "default_w": 20.0, "default_h": 12.0, "rooms": ["living"]},
    "dining_chair": {"shape": "rect", "default_w": 6.0, "default_h": 6.0, "rooms": ["living"]},
    "single_bed": {"shape": "rect", "default_w": 18.0, "default_h": 34.0, "rooms": ["bedroom"]},
    "double_bed": {"shape": "rect", "default_w": 26.0, "default_h": 36.0, "rooms": ["bedroom"]},
    "closet": {"shape": "rect", "default_w": 20.0, "default_h": 6.0, "rooms": ["bedroom"]},
    "dresser": {"shape": "rect", "default_w": 10.0, "default_h": 5.0, "rooms": ["bedroom"]},
    "desk": {"shape": "rect", "default_w": 12.0, "default_h": 7.0, "rooms": ["bedroom"]},
    "chair": {"shape": "rect", "default_w": 5.0, "default_h": 5.0, "rooms": ["bedroom"]},
    "toilet": {"shape": "rect", "default_w": 8.0, "default_h": 12.0, "rooms": ["bathroom"]},
    "sink": {"shape": "rect", "default_w": 8.0, "default_h": 8.0, "rooms": ["bathroom"]},
    "shower": {"shape": "rect", "default_w": 12.0, "default_h": 12.0, "rooms": ["bathroom"]},
}

ICON_ALIAS_TO_CANONICAL = {
    "working_table": "desk",
    "working_chair": "chair",
}

ICON_DISPLAY_LABELS = {
    "single_bed": "single_bed",
    "double_bed": "double_bed",
    "closet": "closet",
    "dresser": "dresser",
    "desk": "working_table",
    "chair": "working_chair",
    "toilet": "toilet",
    "sink": "sink",
    "shower": "shower",
    "kitchen_sink": "kitchen_sink",
    "fridge": "fridge",
    "tv": "tv",
    "coffee_table": "coffee_table",
    "dining_table_rect": "dining_table_rect",
    "dining_table_round": "dining_table_round",
    "sofa": "sofa",
    "rug": "rug",
    "dining_chair": "dining_chair",
}


def _canonical_icon_id(icon_id):
    if not icon_id:
        return icon_id
    return ICON_ALIAS_TO_CANONICAL.get(icon_id, icon_id)


def _icon_display_label(icon_id):
    canonical = _canonical_icon_id(icon_id)
    return ICON_DISPLAY_LABELS.get(canonical, canonical)


def _normalize_manual_item_icon(item):
    if not isinstance(item, dict):
        return item
    canonical = _canonical_icon_id(item.get("icon"))
    if canonical:
        item["icon"] = canonical
        item["display_label"] = _icon_display_label(canonical)
    return item


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


def _to_shapely(geom_data):
    if isinstance(geom_data, BaseGeometry):
        return geom_data
    if isinstance(geom_data, dict) and 'type' in geom_data and 'coordinates' in geom_data:
        return shape(geom_data)
    return None


def _extract_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    if hasattr(geom, 'geoms'):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _editor_room_polygons(playground):
    rooms = {}
    for key in ("living", "bedroom", "bathroom", "kitchen"):
        rooms[key] = _extract_polygons(_to_shapely(playground.get(key)))
    return rooms


def _manual_item_to_footprint(item):
    if item["shape"] == "circle":
        return Point(float(item["x"]), float(item["y"])).buffer(float(item["r"]), resolution=24)

    x = float(item["x"])
    y = float(item["y"])
    w = float(item["w"])
    h = float(item["h"])
    hw = w / 2.0
    hh = h / 2.0
    return Polygon([
        (x - hw, y - hh),
        (x + hw, y - hh),
        (x + hw, y + hh),
        (x - hw, y + hh),
    ])


def _validate_manual_item(item, room_polygons):
    icon = _canonical_icon_id(item.get("icon"))
    item["icon"] = icon
    item["display_label"] = _icon_display_label(icon)
    cfg = MANUAL_FURNITURE_CATALOG.get(icon)
    if not cfg:
        return False, f"Unsupported icon '{icon}'."

    fp = _manual_item_to_footprint(item)
    if fp.is_empty or fp.area <= 1e-9:
        return False, f"{icon}: invalid footprint area."

    soft_tol = max(fp.length * 1e-9, 1e-4)
    for room_key in cfg["rooms"]:
        for room_poly in room_polygons.get(room_key, []):
            if room_poly.buffer(soft_tol).covers(fp):
                return True, None

    return False, f"{icon}: placement must be fully inside one of: {', '.join(cfg['rooms'])}."


def _prompt_icon_choice():
    keys = list(MANUAL_FURNITURE_CATALOG.keys())
    print("\nAvailable icons:")
    for i, k in enumerate(keys, start=1):
        print(f"  {i:>2}. {_icon_display_label(k)}")

    while True:
        raw = input("Choose icon number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print("Invalid icon selection.")


def _interactive_furniture_editor(playground, starting_items, room_polygons):
    """
    Interactive manual furniture editor using PySide6 clicks.
    Returns tuple: ("confirm"|"cancel", items, artifacts).
    """
    if not HAS_QT:
        raise RuntimeError("PySide6 is not installed. Install with: pip install PySide6")

    items = copy.deepcopy(starting_items)
    for it in items:
        _normalize_manual_item_icon(it)

    icon_ids = list(MANUAL_FURNITURE_CATALOG.keys())

    # Build world-space bounds for drawing transform.
    world_polys = []
    for k in ("wall", "living", "bedroom", "bathroom", "kitchen", "balcony"):
        world_polys.extend(_extract_polygons(_to_shapely(playground.get(k))))
    if not world_polys:
        raise RuntimeError("No plan geometry found for editor canvas.")

    minx = min(p.bounds[0] for p in world_polys)
    miny = min(p.bounds[1] for p in world_polys)
    maxx = max(p.bounds[2] for p in world_polys)
    maxy = max(p.bounds[3] for p in world_polys)
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)

    # Keep the scene compact enough for common laptop screens.
    canvas_w = 920.0
    canvas_h = 450
    margin = 20.0
    draw_w = canvas_w - 2.0 * margin
    draw_h = canvas_h - 2.0 * margin
    scale = min(draw_w / span_x, draw_h / span_y)

    state = {
        "active_icon": icon_ids[0],
        "mode": "point",
        "rect_start": None,
        "move_idx": None,
        "status": None,
        "items": items,
        "artifacts": {},
        "message": "Click on the plan to place furniture.",
    }

    def world_to_scene(wx, wy):
        sx = margin + (wx - minx) * scale
        sy = margin + (maxy - wy) * scale
        return sx, sy

    def scene_to_world(sx, sy):
        wx = minx + (sx - margin) / scale
        wy = maxy - (sy - margin) / scale
        return wx, wy

    class _PlanView(QGraphicsView):
        def __init__(self, scene, click_cb):
            super().__init__(scene)
            self._click_cb = click_cb
            self.setMouseTracking(True)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                p = self.mapToScene(event.pos())
                self._click_cb(float(p.x()), float(p.y()))
            super().mousePressEvent(event)

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dlg = QDialog()
    dlg.setWindowTitle("Manual Furniture Editor")
    dlg.resize(1200, 820)
    dlg.setStyleSheet(
        "QDialog { background-color: #1f1f1f; color: #f2f2f2; font-family: 'Times New Roman', 'Cambria', serif; }"
        "QLabel { color: #f2f2f2; font-size: 13px; }"
        "QPushButton { background-color: #343a40; color: #ffffff; border: 1px solid #555; border-radius: 4px; padding: 6px 10px; min-height: 30px; }"
        "QPushButton:hover { background-color: #495057; }"
        "QPushButton:checked { background-color: #0d6efd; border-color: #78a7ff; }"
        "QComboBox { background-color: #2a2a2a; color: #ffffff; border: 1px solid #555; padding: 4px 6px; min-height: 28px; }"
    )

    root_layout = QVBoxLayout(dlg)
    title_label = QLabel("Manual Furniture Editor")
    title_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #ffffff;")
    root_layout.addWidget(title_label)

    intro_label = QLabel(
        "Use the action buttons below to choose what a click does.\n"
        "- Place (Point): one click adds an item at that position.\n"
        "- Place (Rectangle): click two corners to set item size.\n"
        "- Move Item: click an item, then click its new location.\n"
        "- Delete Item: click an item to remove it."
    )
    intro_label.setWordWrap(True)
    intro_label.setStyleSheet("color: #d0d6dc;")
    root_layout.addWidget(intro_label)

    controls = QHBoxLayout()
    icon_combo = QComboBox()
    icon_combo.setMaxVisibleItems(20)
    for icon_id in icon_ids:
        icon_combo.addItem(_icon_display_label(icon_id), icon_id)
    icon_combo.setToolTip("Choose which furniture icon to place.")

    point_btn = QPushButton("Place (Point)")
    rect_btn = QPushButton("Place (Rectangle)")
    move_btn = QPushButton("Move Item")
    delete_btn = QPushButton("Delete Item")

    for btn in (point_btn, rect_btn, move_btn, delete_btn):
        btn.setCheckable(True)

    point_btn.setToolTip("Place one item with a single click.")
    rect_btn.setToolTip("Place by dragging two corners (two clicks).")
    move_btn.setToolTip("Select a nearby item, then click its new position.")
    delete_btn.setToolTip("Delete the nearest item where you click.")

    undo_btn = QPushButton("Undo")
    clear_btn = QPushButton("Clear")
    done_btn = QPushButton("Done")
    cancel_btn = QPushButton("Cancel")

    controls.addWidget(QLabel("Icon:"))
    controls.addWidget(icon_combo)
    controls.addWidget(point_btn)
    controls.addWidget(rect_btn)
    controls.addWidget(move_btn)
    controls.addWidget(delete_btn)
    controls.addWidget(undo_btn)
    controls.addWidget(clear_btn)
    controls.addWidget(done_btn)
    controls.addWidget(cancel_btn)
    root_layout.addLayout(controls)

    mode_label = QLabel("Current action: Place (Point)")
    mode_label.setStyleSheet("color: #9bd3ff; font-weight: 600;")
    root_layout.addWidget(mode_label)

    msg_label = QLabel(state["message"])
    msg_label.setWordWrap(True)
    msg_label.setStyleSheet("color: #f8f9fa; background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; padding: 8px;")
    root_layout.addWidget(msg_label)

    scene = QGraphicsScene(0.0, 0.0, canvas_w, canvas_h)
    view = _PlanView(scene, lambda sx, sy: _on_click(sx, sy))
    view.setStyleSheet("border: 1px solid #555; background-color: #111;")
    view.setMinimumSize(760, 500)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    root_layout.addWidget(view)

    def _set_message(msg):
        state["message"] = msg
        msg_label.setText(msg)

    def _make_item_from_world(icon, mode, wx, wy, w=None, h=None):
        cfg = MANUAL_FURNITURE_CATALOG[icon]
        display_label = _icon_display_label(icon)
        if mode == "point":
            if cfg["shape"] == "circle":
                return {
                    "icon": icon,
                    "display_label": display_label,
                    "shape": "circle",
                    "placement_mode": "point",
                    "x": float(wx),
                    "y": float(wy),
                    "r": float(cfg["default_r"]),
                    "rotation": 0.0,
                }
            return {
                "icon": icon,
                "display_label": display_label,
                "shape": "rect",
                "placement_mode": "point",
                "x": float(wx),
                "y": float(wy),
                "w": float(cfg["default_w"]),
                "h": float(cfg["default_h"]),
                "rotation": 0.0,
            }

        # rect mode
        if w is None or h is None or w <= 1e-6 or h <= 1e-6:
            return None
        if cfg["shape"] == "circle":
            return {
                "icon": icon,
                "display_label": display_label,
                "shape": "circle",
                "placement_mode": "rect",
                "x": float(wx),
                "y": float(wy),
                "r": float(min(w, h) / 2.0),
                "rotation": 0.0,
            }
        return {
            "icon": icon,
            "display_label": display_label,
            "shape": "rect",
            "placement_mode": "rect",
            "x": float(wx),
            "y": float(wy),
            "w": float(w),
            "h": float(h),
            "rotation": 0.0,
        }

    def _draw_plan_base():
        # Room outlines
        for room_key in ("living", "bedroom", "bathroom", "kitchen", "balcony"):
            color = QColor(80, 80, 80)
            if room_key == "living":
                color = QColor(90, 120, 90)
            for poly in _extract_polygons(_to_shapely(playground.get(room_key))):
                pts = [QPointF(*world_to_scene(x, y)) for x, y in poly.exterior.coords]
                scene.addPolygon(QPolygonF(pts), QPen(color, 1.0), QBrush(Qt.NoBrush))

        # Walls/windows stronger line
        for k in ("wall", "window"):
            for poly in _extract_polygons(_to_shapely(playground.get(k))):
                pts = [QPointF(*world_to_scene(x, y)) for x, y in poly.exterior.coords]
                scene.addPolygon(QPolygonF(pts), QPen(QColor(0, 220, 0), 1.5), QBrush(Qt.NoBrush))

        # Doors
        for k in ("door", "front_door"):
            for poly in _extract_polygons(_to_shapely(playground.get(k))):
                pts = [QPointF(*world_to_scene(x, y)) for x, y in poly.exterior.coords]
                scene.addPolygon(QPolygonF(pts), QPen(QColor(210, 0, 210), 1.2), QBrush(Qt.NoBrush))

    def _redraw_all():
        scene.clear()
        _draw_plan_base()

        # Existing furniture overlays
        for item in state["items"]:
            icon = _canonical_icon_id(item.get("icon"))
            item_label = item.get("display_label", _icon_display_label(icon))
            if item["shape"] == "circle":
                cx, cy = world_to_scene(item["x"], item["y"])
                rr = item["r"] * scale
                scene.addEllipse(cx - rr, cy - rr, rr * 2.0, rr * 2.0, QPen(QColor(0, 255, 255), 1.5), QBrush(Qt.NoBrush))
                txt = scene.addSimpleText(item_label)
                txt.setBrush(QBrush(QColor(255, 255, 255)))
                txt.setPos(cx + 4, cy + 4)
            else:
                x0, y0 = world_to_scene(item["x"] - item["w"] / 2.0, item["y"] - item["h"] / 2.0)
                x1, y1 = world_to_scene(item["x"] + item["w"] / 2.0, item["y"] + item["h"] / 2.0)
                rx, ry = min(x0, x1), min(y0, y1)
                rw, rh = abs(x1 - x0), abs(y1 - y0)
                scene.addRect(rx, ry, rw, rh, QPen(QColor(0, 255, 255), 1.5), QBrush(Qt.NoBrush))
                txt = scene.addSimpleText(item_label)
                txt.setBrush(QBrush(QColor(255, 255, 255)))
                txt.setPos(rx + 3, ry + 3)

        # Move-mode selected item highlight
        if state["move_idx"] is not None and 0 <= state["move_idx"] < len(state["items"]):
            sel = state["items"][state["move_idx"]]
            if sel["shape"] == "circle":
                cx, cy = world_to_scene(sel["x"], sel["y"])
                rr = sel["r"] * scale
                scene.addEllipse(cx - rr - 3, cy - rr - 3, (rr + 3) * 2.0, (rr + 3) * 2.0, QPen(QColor(255, 180, 0), 2.0), QBrush(Qt.NoBrush))
            else:
                x0, y0 = world_to_scene(sel["x"] - sel["w"] / 2.0, sel["y"] - sel["h"] / 2.0)
                x1, y1 = world_to_scene(sel["x"] + sel["w"] / 2.0, sel["y"] + sel["h"] / 2.0)
                rx, ry = min(x0, x1), min(y0, y1)
                rw, rh = abs(x1 - x0), abs(y1 - y0)
                scene.addRect(rx - 3, ry - 3, rw + 6, rh + 6, QPen(QColor(255, 180, 0), 2.0), QBrush(Qt.NoBrush))

        # Rect-mode first click preview marker
        if state["rect_start"] is not None:
            sx, sy = state["rect_start"]
            scene.addEllipse(sx - 4, sy - 4, 8, 8, QPen(QColor(255, 180, 0), 1.0), QBrush(QColor(255, 180, 0)))

    def _on_icon_change(_):
        chosen = icon_combo.currentData()
        state["active_icon"] = _canonical_icon_id(chosen)

    action_buttons = {
        "point": point_btn,
        "rect": rect_btn,
        "move": move_btn,
        "delete": delete_btn,
    }

    action_names = {
        "point": "Place (Point)",
        "rect": "Place (Rectangle)",
        "move": "Move Item",
        "delete": "Delete Item",
    }

    def _on_mode_change(mode_value):
        state["mode"] = mode_value
        state["rect_start"] = None
        state["move_idx"] = None
        mode_label.setText(f"Current action: {action_names.get(mode_value, mode_value)}")
        for action_mode, action_btn in action_buttons.items():
            action_btn.setChecked(action_mode == mode_value)
        _set_message(f"Action changed to {action_names.get(mode_value, mode_value)}. Click on plan.")
        _redraw_all()

    def _nearest_item_index(world_x, world_y):
        if not state["items"]:
            return None
        best_idx = None
        best_d2 = None
        for idx, it in enumerate(state["items"]):
            dx = float(it.get("x", 0.0)) - world_x
            dy = float(it.get("y", 0.0)) - world_y
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = idx

        # selection threshold in world units (about 20 px)
        threshold = (20.0 / max(scale, 1e-9)) ** 2
        if best_d2 is not None and best_d2 <= threshold:
            return best_idx
        return None

    def _on_undo():
        if state["items"]:
            state["items"].pop()
            _set_message("Removed last item.")
            _redraw_all()

    def _on_clear():
        state["items"].clear()
        state["rect_start"] = None
        _set_message("Cleared all items.")
        _redraw_all()

    def _on_done():
        progress = None
        artifacts = {}
        apt_id = playground.get("apartment_id") or playground.get("selected_apartment_id")

        state["status"] = "confirm"
        try:
            progress = QProgressDialog("Generating apartment preview and render...", None, 0, 0, dlg)
            progress.setWindowTitle("Generating")
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.show()
            QApplication.processEvents()

            from snapshot_to_render import run_post_done_render

            artifacts = run_post_done_render(scene=scene, apartment_id=apt_id, playground=playground)
            _set_message(
                "Done confirmed. Snapshot and render saved to dxf_output/snapshots."
            )
        except Exception as exc:
            artifacts = {"error": str(exc)}
            _set_message(f"Done confirmed. Render step failed (non-fatal): {exc}")
        finally:
            state["artifacts"] = artifacts
            if progress is not None:
                progress.close()
            dlg.accept()

    def _on_cancel():
        state["status"] = "cancel"
        dlg.reject()

    def _on_click(mouse_x, mouse_y):
        if not (margin <= mouse_x <= canvas_w - margin and margin <= mouse_y <= canvas_h - margin):
            return

        icon = state["active_icon"]
        icon_label = _icon_display_label(icon)
        mode = state["mode"]
        cfg = MANUAL_FURNITURE_CATALOG[icon]

        wx, wy = scene_to_world(mouse_x, mouse_y)

        if mode == "delete":
            idx = _nearest_item_index(wx, wy)
            if idx is None:
                _set_message("Delete mode: no nearby item.")
                return
            removed = state["items"].pop(idx)
            state["move_idx"] = None
            _set_message(f"Deleted {removed.get('icon', 'item')}.")
            _redraw_all()
            return

        if mode == "move":
            if state["move_idx"] is None:
                idx = _nearest_item_index(wx, wy)
                if idx is None:
                    _set_message("Move mode: click near an item to select it.")
                    return
                state["move_idx"] = idx
                _set_message("Move mode: item selected, click destination point.")
                _redraw_all()
                return

            idx = state["move_idx"]
            if not (0 <= idx < len(state["items"])):
                state["move_idx"] = None
                _set_message("Move mode: selection reset.")
                _redraw_all()
                return

            moved = copy.deepcopy(state["items"][idx])
            moved["x"] = float(wx)
            moved["y"] = float(wy)
            ok, reason = _validate_manual_item(moved, room_polygons)
            if not ok:
                _set_message(f"Move rejected: {reason}")
                state["move_idx"] = None
                _redraw_all()
                return

            state["items"][idx] = moved
            state["move_idx"] = None
            _set_message(f"Moved {moved.get('icon', 'item')}.")
            _redraw_all()
            return

        if mode == "point":
            item = _make_item_from_world(icon, "point", wx, wy)
            ok, reason = _validate_manual_item(item, room_polygons)
            if not ok:
                _set_message(f"Rejected: {reason}")
                return
            state["items"].append(item)
            _set_message(f"Added {icon_label}.")
            _redraw_all()
            return

        # rect mode: 2-click rectangle
        if state["rect_start"] is None:
            state["rect_start"] = (mouse_x, mouse_y)
            _set_message("Rect mode: click second corner.")
            _redraw_all()
            return

        sx0, sy0 = state["rect_start"]
        sx1, sy1 = mouse_x, mouse_y
        state["rect_start"] = None

        wx0, wy0 = scene_to_world(sx0, sy0)
        wx1, wy1 = scene_to_world(sx1, sy1)
        w = abs(wx1 - wx0)
        h = abs(wy1 - wy0)
        if w <= 1e-6 or h <= 1e-6:
            _set_message("Rejected: rectangle too small.")
            _redraw_all()
            return

        wcx = (wx0 + wx1) / 2.0
        wcy = (wy0 + wy1) / 2.0
        item = _make_item_from_world(icon, "rect", wcx, wcy, w=w, h=h)
        if item is None:
            _set_message("Rejected: invalid rectangle.")
            _redraw_all()
            return

        ok, reason = _validate_manual_item(item, room_polygons)
        if not ok:
            _set_message(f"Rejected: {reason}")
            _redraw_all()
            return

        state["items"].append(item)
        _set_message(f"Added {cfg['shape']} {icon_label}.")
        _redraw_all()

    icon_combo.currentIndexChanged.connect(_on_icon_change)
    point_btn.clicked.connect(lambda: _on_mode_change("point"))
    rect_btn.clicked.connect(lambda: _on_mode_change("rect"))
    move_btn.clicked.connect(lambda: _on_mode_change("move"))
    delete_btn.clicked.connect(lambda: _on_mode_change("delete"))
    undo_btn.clicked.connect(_on_undo)
    clear_btn.clicked.connect(_on_clear)
    done_btn.clicked.connect(_on_done)
    cancel_btn.clicked.connect(_on_cancel)

    _on_mode_change("point")

    _redraw_all()

    dlg.exec()

    status = state["status"] or "cancel"
    out_items = copy.deepcopy(state["items"])
    out_artifacts = copy.deepcopy(state.get("artifacts", {}))


    if status == "confirm":
        return "confirm", out_items, out_artifacts
    return "cancel", None, {}


def _cli_furniture_editor(playground, starting_items, room_polygons):
    """
    Temporary fallback editor when plot backend is unavailable.
    Returns tuple: ("confirm"|"cancel", items, artifacts).
    """
    _ = playground  # kept for interface parity
    items = copy.deepcopy(starting_items)

    print("\n--- CLI Furniture Editor Fallback ---")
    print("Commands: add, list, undo, clear, done, cancel, help")

    while True:
        cmd = input("[CLI Editor] command: ").strip().lower()

        if cmd in {"help", "h"}:
            print("add -> choose icon/mode and enter coordinates")
            print("done -> confirm")
            print("cancel -> reopen editor with previous confirmed items")
            continue

        if cmd in {"list", "ls"}:
            if not items:
                print("  (no items)")
            for i, it in enumerate(items, start=1):
                print(f"  {i:>2}. {it}")
            continue

        if cmd == "undo":
            if items:
                items.pop()
            continue

        if cmd == "clear":
            items = []
            continue

        if cmd == "done":
            return "confirm", items, {}

        if cmd == "cancel":
            return "cancel", None, {}

        if cmd != "add":
            print("Unknown command.")
            continue

        icon = _prompt_icon_choice()
        cfg = MANUAL_FURNITURE_CATALOG[icon]
        mode = input("Mode [point/rect]: ").strip().lower()
        if mode not in {"point", "rect"}:
            print("Invalid mode.")
            continue

        try:
            if mode == "point":
                x = float(input("x: ").strip())
                y = float(input("y: ").strip())
                if cfg["shape"] == "circle":
                    item = {
                        "icon": icon, "shape": "circle", "placement_mode": "point",
                        "x": x, "y": y, "r": float(cfg["default_r"]), "rotation": 0.0,
                    }
                else:
                    item = {
                        "icon": icon, "shape": "rect", "placement_mode": "point",
                        "x": x, "y": y, "w": float(cfg["default_w"]), "h": float(cfg["default_h"]), "rotation": 0.0,
                    }
            else:
                x0 = float(input("x0: ").strip())
                y0 = float(input("y0: ").strip())
                x1 = float(input("x1: ").strip())
                y1 = float(input("y1: ").strip())
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                if w <= 1e-6 or h <= 1e-6:
                    print("Rectangle dimensions are too small.")
                    continue
                if cfg["shape"] == "circle":
                    item = {
                        "icon": icon, "shape": "circle", "placement_mode": "rect",
                        "x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0, "r": min(w, h) / 2.0, "rotation": 0.0,
                    }
                else:
                    item = {
                        "icon": icon, "shape": "rect", "placement_mode": "rect",
                        "x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0, "w": w, "h": h, "rotation": 0.0,
                    }
        except ValueError:
            print("Invalid numeric input.")
            continue

        ok, reason = _validate_manual_item(item, room_polygons)
        if not ok:
            print(f"[CLI Editor][Rejected] {reason}")
            continue
        items.append(item)


def _run_mandatory_manual_editor(playground):
    """
    Mandatory editor loop. Cancel always reopens with previous confirmed items.
    Empty confirmed item set is valid.
    """
    existing = playground.get("manual_furniture", {})
    confirmed_items = copy.deepcopy(existing.get("items", []))
    room_polygons = _editor_room_polygons(playground)

    while True:
        try:
            status, edited_items, artifacts = _interactive_furniture_editor(playground, confirmed_items, room_polygons)
        except Exception as exc:
            print(f"[Editor] Interactive editor unavailable ({exc}). Falling back to CLI editor.")
            status, edited_items, artifacts = _cli_furniture_editor(playground, confirmed_items, room_polygons)

        if status == "confirm":
            confirmed_items = copy.deepcopy(edited_items)
            playground["manual_furniture"] = {
                "enabled": True,
                "confirmed": True,
                "items": confirmed_items,
            }
            if artifacts:
                playground.setdefault("artifacts", {})
                playground["artifacts"].update(artifacts)
            print(f"[Editor] Confirmed {len(confirmed_items)} manual furniture item(s).")
            return playground

        print("[Editor] Cancel selected. Reopening with previous confirmed items.")


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