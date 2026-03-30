import copy
import json
import os
from datetime import datetime


import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry

from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporter import export_to_dxf
from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine import (
    prepare_selected_playground_for_web,
    search_candidates_for_web,
)
from ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine_helpers import (
    build_candidate_preview_image,
    get_playground_from_sqlite,
)
from ResPlan_RAG.pipeline_code.retrieval_engine.interactive_cli import (
    ICON_DISPLAY_LABELS,
    MANUAL_FURNITURE_CATALOG,
    _canonical_icon_id,
    _icon_display_label,
    _validate_manual_item,
)
from ResPlan_RAG.pipeline_code.snapshot_pipeline.snapshot_to_render import (
    create_canny_control_map,
    generate_hf_render,
    process_cad_to_seg_map,
)


CANVAS_W = 1200
CANVAS_H = 680
ZOOM_MIN = 0.6
ZOOM_MAX = 2.8
ZOOM_STEP = 0.2
PAGE_CSS = """
.gradio-container {max-width: 1700px !important; margin: 0 auto;}
class="md svelte-1hf8a14 prose"
.section-card {background: #141821; border: 1px solid #2b3242; border-radius: 12px; padding: 12px;}
.muted {color: #b8c0cc;}
#editor-canvas {border: 1px solid #2b3242; border-radius: 8px;}
#zoom-display input {font-weight: 700;}
#overlay-box {border: 1px solid #3b82f6; border-radius: 10px; background: rgba(10, 15, 28, 0.95);}
"""


def _to_shapely(geom_data):
    if isinstance(geom_data, BaseGeometry):
        return geom_data
    if (
        isinstance(geom_data, dict)
        and "type" in geom_data
        and "coordinates" in geom_data
    ):
        return shape(geom_data)
    return None


def _extract_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _editor_room_polygons(playground):
    rooms = {}
    for key in ("living", "bedroom", "bathroom", "kitchen"):
        rooms[key] = _extract_polygons(_to_shapely(playground.get(key)))
    return rooms


def _world_bounds(playground):
    polys = []
    for key in ("wall", "living", "bedroom", "bathroom", "kitchen", "balcony"):
        polys.extend(_extract_polygons(_to_shapely(playground.get(key))))
    if not polys:
        return 0.0, 0.0, 100.0, 100.0
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return float(minx), float(miny), float(maxx), float(maxy)


def _view_bounds(world_bounds, zoom):
    minx, miny, maxx, maxy = world_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    spanx = (maxx - minx) / max(zoom, 1e-6)
    spany = (maxy - miny) / max(zoom, 1e-6)
    return cx - spanx / 2.0, cy - spany / 2.0, cx + spanx / 2.0, cy + spany / 2.0


def _world_to_px(x, y, view):
    vx0, vy0, vx1, vy1 = view
    w = max(vx1 - vx0, 1e-6)
    h = max(vy1 - vy0, 1e-6)
    px = int(round((x - vx0) / w * (CANVAS_W - 1)))
    py = int(round((vy1 - y) / h * (CANVAS_H - 1)))
    return px, py


def _px_to_world(px, py, view):
    vx0, vy0, vx1, vy1 = view
    w = max(vx1 - vx0, 1e-6)
    h = max(vy1 - vy0, 1e-6)
    x = vx0 + (float(px) / max(CANVAS_W - 1, 1)) * w
    y = vy1 - (float(py) / max(CANVAS_H - 1, 1)) * h
    return float(x), float(y)


def _manual_item_to_footprint(item):
    if item["shape"] == "circle":
        return Point(float(item["x"]), float(item["y"])).buffer(
            float(item["r"]), resolution=24
        )

    x = float(item["x"])
    y = float(item["y"])
    w = float(item["w"])
    h = float(item["h"])
    hw = w / 2.0
    hh = h / 2.0
    return Polygon(
        [
            (x - hw, y - hh),
            (x + hw, y - hh),
            (x + hw, y + hh),
            (x - hw, y + hh),
        ]
    )


def _render_editor_image(playground, items, zoom=1.0):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (17, 17, 17))
    draw = ImageDraw.Draw(img)

    world = _world_bounds(playground)
    view = _view_bounds(world, zoom)

    room_colors = {
        "living": (100, 140, 100),
        "bedroom": (120, 120, 160),
        "bathroom": (130, 110, 110),
        "kitchen": (130, 130, 90),
        "balcony": (90, 120, 120),
    }

    for key, color in room_colors.items():
        for poly in _extract_polygons(_to_shapely(playground.get(key))):
            pts = [_world_to_px(x, y, view) for x, y in poly.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, outline=color)

    for key in ("wall", "window"):
        for poly in _extract_polygons(_to_shapely(playground.get(key))):
            pts = [_world_to_px(x, y, view) for x, y in poly.exterior.coords]
            if len(pts) >= 3:
                draw.line(pts, fill=(0, 225, 0), width=2)

    for key in ("door", "front_door"):
        for poly in _extract_polygons(_to_shapely(playground.get(key))):
            pts = [_world_to_px(x, y, view) for x, y in poly.exterior.coords]
            if len(pts) >= 3:
                draw.line(pts, fill=(225, 0, 225), width=2)

    for item in items:
        icon = _canonical_icon_id(item.get("icon"))
        label = ICON_DISPLAY_LABELS.get(icon, icon)
        if item.get("shape") == "circle":
            cx, cy = _world_to_px(float(item["x"]), float(item["y"]), view)
            edge = _manual_item_to_footprint(item).bounds
            ex0, ey0 = _world_to_px(edge[0], edge[1], view)
            ex1, ey1 = _world_to_px(edge[2], edge[3], view)
            draw.ellipse(
                [min(ex0, ex1), min(ey0, ey1), max(ex0, ex1), max(ey0, ey1)],
                outline=(0, 255, 255),
                width=2,
            )
            draw.text((cx + 3, cy + 3), label, fill=(255, 255, 255))
        else:
            x = float(item["x"])
            y = float(item["y"])
            w = float(item["w"])
            h = float(item["h"])
            p0 = _world_to_px(x - w / 2.0, y - h / 2.0, view)
            p1 = _world_to_px(x + w / 2.0, y + h / 2.0, view)
            x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
            x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=2)
            draw.text((x0 + 3, y0 + 3), label, fill=(255, 255, 255))

    return np.array(img), view


def _nearest_item_index(items, wx, wy, world_bounds):
    if not items:
        return None
    minx, miny, maxx, maxy = world_bounds
    span = max(maxx - minx, maxy - miny, 1e-6)
    threshold = (span * 0.04) ** 2
    best_idx = None
    best_d2 = None
    for idx, it in enumerate(items):
        dx = float(it.get("x", 0.0)) - wx
        dy = float(it.get("y", 0.0)) - wy
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_idx = idx
    if best_d2 is not None and best_d2 <= threshold:
        return best_idx
    return None


def _new_editor_state():
    return {
        "cards": [],
        "selected_index": None,
        "pending_index": None,
        "playground": None,
        "items": [],
        "history": [],
        "mode": "point",
        "active_icon": list(MANUAL_FURNITURE_CATALOG.keys())[0],
        "rect_start": None,
        "move_idx": None,
        "zoom": 1.0,
        "view": (0.0, 0.0, 100.0, 100.0),
    }


def _card_caption(card):
    area = card.get("area")
    area_str = f"{area:.1f} sqm" if isinstance(area, float) else "N/A"
    return (
        f"#{card['rank']} ID {card['apartment_id']} | {area_str} | "
        f"bed {card['bedrooms']} bath {card['bathrooms']} balcony {card['balconies']}"
    )


def _search(query, bedrooms, bathrooms, balconies, state):
    # Check if the query is empty or just whitespace
    if not query or not str(query).strip():
        # Use gr.update() instead of None to prevent Gradio from
        # attempting to access directory paths for image components
        return state, [], "Enter a search query.", gr.update(), "", "", gr.update()

    # Perform the search using the retrieval engine
    cards = search_candidates_for_web(
        query=str(query).strip(),
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        balconies=balconies,
        n_results=10,
    )

    # Reset the editor state for a fresh search session
    state = _new_editor_state()

    # Create a lightweight, serializable version of the card data for the Gradio state
    safe_cards = []
    for card in cards:
        safe_cards.append(
            {
                "rank": card.get("rank"),
                "apartment_id": card.get("apartment_id"),
                "description": card.get("description", ""),
                "area": card.get("area"),
                "bedrooms": card.get("bedrooms", 0),
                "bathrooms": card.get("bathrooms", 0),
                "balconies": card.get("balconies", 0),
            }
        )

    state["cards"] = safe_cards

    # Map search results to the Gallery component format (Image, Caption)
    gallery = [
        (card.get("preview_image"), _card_caption(meta))
        for card, meta in zip(cards, safe_cards)
    ]

    status = f"Found {len(cards)} candidate(s). Click a card, review details, then press Select this plan."

    # Return gr.update() for the 'editor', 'candidate_preview', and 'render_output'
    # to avoid triggering the PermissionError on directory paths
    return state, gallery, status, gr.update(), "", gr.update(), gr.update()


def _format_candidate_metadata(card):
    desc = card.get("description", "")
    if len(desc) > 1200:
        desc = desc[:1197] + "..."
    area = card.get("area")
    area_str = f"{float(area):.2f} sqm" if isinstance(area, (int, float)) else "N/A"
    return (
        f"### Candidate #{card.get('rank')} (ID {card.get('apartment_id')})\n"
        f"- **Area:** {area_str}\n"
        f"- **Bedrooms:** {card.get('bedrooms', 0)}\n"
        f"- **Bathrooms:** {card.get('bathrooms', 0)}\n"
        f"- **Balconies:** {card.get('balconies', 0)}\n\n"
        f"**Description**\n\n{desc}"
    )


def _preview_candidate(evt: gr.SelectData, state):
    idx = int(evt.index) if evt and evt.index is not None else None
    cards = state.get("cards", [])
    if idx is None or idx < 0 or idx >= len(cards):
        return state, "Invalid selection.", "", None

    state["pending_index"] = idx
    card = cards[idx]
    preview_img = None
    try:
        apt_id = int(card["apartment_id"])
        pg = get_playground_from_sqlite(apt_id)
        if pg is not None:
            preview_img = build_candidate_preview_image(
                card["rank"], apt_id, card.get("description", ""), pg
            )
    except Exception:
        preview_img = None
    msg = f"Previewing candidate #{card['rank']} (ID {card['apartment_id']}). Click Select this plan to continue."
    return state, msg, _format_candidate_metadata(card), preview_img


def _select_candidate(target_area, state):
    idx = state.get("pending_index")
    cards = state.get("cards", [])
    if idx is None or idx < 0 or idx >= len(cards):
        return state, None, "Select a candidate card first.", "", 1.0

    selected = cards[idx]
    area = None
    if target_area not in (None, ""):
        try:
            area = float(target_area)
        except Exception:
            area = None

    playground = get_playground_from_sqlite(int(selected["apartment_id"]))
    if playground is None:
        return (
            state,
            None,
            "Failed to load selected apartment geometry from SQLite.",
            "",
            1.0,
        )

    playground["apartment_id"] = int(selected["apartment_id"])
    selected_for_prepare = copy.deepcopy(selected)
    selected_for_prepare["playground"] = playground

    playground = prepare_selected_playground_for_web(
        selected_for_prepare, target_area_sqm=area
    )
    items = copy.deepcopy(playground.get("manual_furniture", {}).get("items", []))

    state["selected_index"] = idx
    state["playground"] = playground
    state["items"] = items
    state["history"] = [copy.deepcopy(items)]
    state["zoom"] = 1.0

    image, view = _render_editor_image(playground, items, zoom=state["zoom"])
    state["view"] = view

    card = cards[idx]
    metadata = _format_candidate_metadata(card)

    return (
        state,
        image,
        "Candidate selected. Edit furniture and click Done.",
        metadata,
        state["zoom"],
    )


def _set_mode(mode, state):
    state["mode"] = str(mode)
    state["rect_start"] = None
    state["move_idx"] = None
    return state, f"Action changed to {mode}."


def _set_icon(icon_label, state):
    icon_id = icon_label
    for k, v in ICON_DISPLAY_LABELS.items():
        if v == icon_label:
            icon_id = k
            break
    state["active_icon"] = _canonical_icon_id(icon_id)
    return state, f"Icon set to {_icon_display_label(state['active_icon'])}."


def _update_zoom(delta, state):
    if state.get("playground") is None:
        return state, None, "Select a candidate first.", state.get("zoom", 1.0)

    state["zoom"] = float(np.clip(state.get("zoom", 1.0) + delta, ZOOM_MIN, ZOOM_MAX))
    image, view = _render_editor_image(
        state["playground"], state["items"], zoom=state["zoom"]
    )
    state["view"] = view
    return state, image, f"Zoom: {state['zoom']:.1f}x", state["zoom"]


def _apply_wheel_delta(wheel_payload, state):
    if not wheel_payload:
        return state, gr.update(), gr.update(), gr.update()
    try:
        data = json.loads(str(wheel_payload))
        delta_y = float(data.get("deltaY", 0.0))
    except Exception:
        return state, gr.update(), gr.update(), gr.update()

    # Browser wheel deltaY>0 means scroll down (zoom out), deltaY<0 zoom in.
    zoom_delta = -ZOOM_STEP if delta_y > 0 else ZOOM_STEP
    return _update_zoom(zoom_delta, state)


def _apply_canvas_click(evt: gr.SelectData, state):
    if state.get("playground") is None:
        return state, None, "Select a candidate first."

    if not evt or not evt.index:
        image, _ = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        return state, image, "Click ignored."

    px, py = evt.index
    wx, wy = _px_to_world(px, py, state["view"])
    mode = state.get("mode", "point")
    icon = state.get("active_icon")
    room_polygons = _editor_room_polygons(state["playground"])

    if mode == "delete":
        idx = _nearest_item_index(
            state["items"], wx, wy, _world_bounds(state["playground"])
        )
        if idx is None:
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, "Delete mode: no nearby item."
        removed = state["items"].pop(idx)
        state["history"].append(copy.deepcopy(state["items"]))
        image, view = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        state["view"] = view
        return state, image, f"Deleted {_icon_display_label(removed.get('icon'))}."

    if mode == "move":
        if state.get("move_idx") is None:
            idx = _nearest_item_index(
                state["items"], wx, wy, _world_bounds(state["playground"])
            )
            if idx is None:
                image, _ = _render_editor_image(
                    state["playground"], state["items"], zoom=state["zoom"]
                )
                return state, image, "Move mode: click near an item to select it."
            state["move_idx"] = idx
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, "Move mode: item selected. Click destination point."

        idx = state.get("move_idx")
        moved = copy.deepcopy(state["items"][idx])
        moved["x"] = float(wx)
        moved["y"] = float(wy)
        ok, reason = _validate_manual_item(moved, room_polygons)
        state["move_idx"] = None
        if not ok:
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, f"Move rejected: {reason}"
        state["items"][idx] = moved
        state["history"].append(copy.deepcopy(state["items"]))
        image, view = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        state["view"] = view
        return state, image, f"Moved {_icon_display_label(moved.get('icon'))}."

    cfg = MANUAL_FURNITURE_CATALOG[icon]
    if mode == "point":
        if cfg["shape"] == "circle":
            item = {
                "icon": icon,
                "display_label": _icon_display_label(icon),
                "shape": "circle",
                "placement_mode": "point",
                "x": float(wx),
                "y": float(wy),
                "r": float(cfg["default_r"]),
                "rotation": 0.0,
            }
        else:
            item = {
                "icon": icon,
                "display_label": _icon_display_label(icon),
                "shape": "rect",
                "placement_mode": "point",
                "x": float(wx),
                "y": float(wy),
                "w": float(cfg["default_w"]),
                "h": float(cfg["default_h"]),
                "rotation": 0.0,
            }
        ok, reason = _validate_manual_item(item, room_polygons)
        if not ok:
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, f"Rejected: {reason}"
        state["items"].append(item)
        state["history"].append(copy.deepcopy(state["items"]))
        image, view = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        state["view"] = view
        return state, image, f"Added {_icon_display_label(icon)}."

    if mode == "rect":
        if state.get("rect_start") is None:
            state["rect_start"] = (px, py)
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, "Rect mode: click second corner."

        px0, py0 = state["rect_start"]
        state["rect_start"] = None
        wx0, wy0 = _px_to_world(px0, py0, state["view"])
        wx1, wy1 = _px_to_world(px, py, state["view"])
        w = abs(wx1 - wx0)
        h = abs(wy1 - wy0)
        if w <= 1e-6 or h <= 1e-6:
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, "Rejected: rectangle too small."

        if cfg["shape"] == "circle":
            item = {
                "icon": icon,
                "display_label": _icon_display_label(icon),
                "shape": "circle",
                "placement_mode": "rect",
                "x": float((wx0 + wx1) / 2.0),
                "y": float((wy0 + wy1) / 2.0),
                "r": float(min(w, h) / 2.0),
                "rotation": 0.0,
            }
        else:
            item = {
                "icon": icon,
                "display_label": _icon_display_label(icon),
                "shape": "rect",
                "placement_mode": "rect",
                "x": float((wx0 + wx1) / 2.0),
                "y": float((wy0 + wy1) / 2.0),
                "w": float(w),
                "h": float(h),
                "rotation": 0.0,
            }
        ok, reason = _validate_manual_item(item, room_polygons)
        if not ok:
            image, _ = _render_editor_image(
                state["playground"], state["items"], zoom=state["zoom"]
            )
            return state, image, f"Rejected: {reason}"

        state["items"].append(item)
        state["history"].append(copy.deepcopy(state["items"]))
        image, view = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        state["view"] = view
        return state, image, f"Added {_icon_display_label(icon)} with rectangle mode."

    image, _ = _render_editor_image(
        state["playground"], state["items"], zoom=state["zoom"]
    )
    return state, image, f"Unsupported mode: {mode}"


def _undo(state):
    if state.get("playground") is None:
        return state, None, "Select a candidate first."
    history = state.get("history", [])
    if len(history) <= 1:
        image, _ = _render_editor_image(
            state["playground"], state["items"], zoom=state["zoom"]
        )
        return state, image, "Nothing to undo."
    history.pop()
    state["items"] = copy.deepcopy(history[-1])
    image, view = _render_editor_image(
        state["playground"], state["items"], zoom=state["zoom"]
    )
    state["view"] = view
    return state, image, "Undo applied."


def _clear(state):
    if state.get("playground") is None:
        return state, None, "Select a candidate first."
    state["items"] = []
    state["history"].append([])
    state["rect_start"] = None
    state["move_idx"] = None
    image, view = _render_editor_image(
        state["playground"], state["items"], zoom=state["zoom"]
    )
    state["view"] = view
    return state, image, "Cleared all items."


def _run_render_from_editor_image(playground, editor_image_np):
    apt_id = playground.get("apartment_id", "web")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.SNAPSHOT_OUTPUT_DIR, f"apt_{apt_id}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    raw_path = os.path.join(run_dir, f"apt_{apt_id}_{ts}_raw_web.png")
    Image.fromarray(editor_image_np).save(raw_path)

    seg_path = os.path.join(run_dir, f"apt_{apt_id}_{ts}_seg_web.png")
    canny_path = os.path.join(run_dir, f"apt_{apt_id}_{ts}_canny_web.png")
    render_path = os.path.join(run_dir, f"apt_{apt_id}_{ts}_render_web.png")

    mode = str(config.HF_CONTROL_MODE).strip().lower()
    control_images = []
    control_models = []
    conditioning_scale = None

    if mode == "seg":
        process_cad_to_seg_map(raw_path, seg_path, playground=playground)
        control_images = [seg_path]
        control_models = [config.HF_CONTROLNET_MODEL_ID_SEG]
        conditioning_scale = 1.0
    elif mode == "canny":
        create_canny_control_map(raw_path, canny_path)
        control_images = [canny_path]
        control_models = [config.HF_CONTROLNET_MODEL_ID_CANNY]
        conditioning_scale = 1.0
    else:
        process_cad_to_seg_map(raw_path, seg_path, playground=playground)
        create_canny_control_map(raw_path, canny_path)
        control_images = [seg_path, canny_path]
        control_models = [
            config.HF_CONTROLNET_MODEL_ID_SEG,
            config.HF_CONTROLNET_MODEL_ID_CANNY,
        ]
        conditioning_scale = list(config.HF_MULTI_CONTROL_SCALES)

    generate_hf_render(
        control_image_paths=control_images,
        controlnet_model_ids=control_models,
        output_image_path=render_path,
        controlnet_conditioning_scale=conditioning_scale,
    )
    return raw_path, seg_path, canny_path, render_path


def _done(editor_image, state):
    if state.get("playground") is None:
        return "Select a candidate first.", None, None

    playground = copy.deepcopy(state["playground"])
    playground["manual_furniture"] = {
        "enabled": True,
        "confirmed": True,
        "items": copy.deepcopy(state["items"]),
    }

    apt_id = playground.get("apartment_id", "web")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dxf_dir = os.path.join(config.PIPELINE_CODE_DIR, "dxf_output")
    os.makedirs(dxf_dir, exist_ok=True)
    dxf_path = os.path.join(dxf_dir, f"apt_{apt_id}_{ts}_web.dxf")
    export_to_dxf(playground, output_path=dxf_path)

    render_path = None
    status = f"DXF exported: {dxf_path}"

    if bool(config.ENABLE_POST_DONE_RENDER):
        try:
            _, _, _, render_path = _run_render_from_editor_image(
                playground, editor_image
            )
            status += f"\nRender exported: {render_path}"
        except Exception as exc:
            status += f"\nRender failed (non-fatal): {exc}"

    return status, dxf_path, render_path


def _done_with_progress(editor_image, state, progress=gr.Progress()):
    if state.get("playground") is None:
        yield "Select a candidate first.", None, None, gr.update(visible=False)
        return

    yield "Generating...", None, None, gr.update(visible=True)
    progress(0.05, desc="Preparing export")

    playground = copy.deepcopy(state["playground"])
    playground["manual_furniture"] = {
        "enabled": True,
        "confirmed": True,
        "items": copy.deepcopy(state["items"]),
    }

    apt_id = playground.get("apartment_id", "web")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dxf_dir = os.path.join(config.PROJECT_ROOT_DIR, "dxf_output")
    os.makedirs(dxf_dir, exist_ok=True)
    dxf_path = os.path.join(dxf_dir, f"apt_{apt_id}_{ts}_web.dxf")

    progress(0.30, desc="Exporting DXF")
    export_to_dxf(playground, output_path=dxf_path)
    status = f"DXF exported: {dxf_path}"
    yield status + "\nGenerating render...", dxf_path, None, gr.update(visible=True)

    render_path = None
    if bool(config.ENABLE_POST_DONE_RENDER):
        try:
            progress(0.55, desc="Creating control maps")
            _, _, _, render_path = _run_render_from_editor_image(
                playground, editor_image
            )
            progress(1.0, desc="Done")
            status += f"\nRender exported: {render_path}"
        except Exception as exc:
            progress(1.0, desc="Done with render warning")
            status += f"\nRender failed (non-fatal): {exc}"
    else:
        progress(1.0, desc="Done")

    yield status, dxf_path, render_path, gr.update(visible=False)


def create_app():
    icon_labels = sorted(set(ICON_DISPLAY_LABELS.values()))
    default_icon_label = _icon_display_label(list(MANUAL_FURNITURE_CATALOG.keys())[0])

    with gr.Blocks(title="Cursor For Architects") as demo:
        state = gr.State(_new_editor_state())
        wheel_payload = gr.Textbox(visible=False, elem_id="wheel-payload")
        wheel_dispatch = gr.Button(
            "wheel-dispatch", visible=False, elem_id="wheel-dispatch"
        )

        gr.Markdown(
            "## Cursor For Architects\n"
            "A user-friendly end-to-end pipeline: search -> compare top plans -> select -> edit -> export DXF + render."
        )
        gr.Markdown(
            "Use the candidate preview panel before committing to a plan. Mouse wheel and +/- buttons both control zoom.",
            elem_classes=["muted"],
        )

        with gr.Row():
            query = gr.Textbox(
                label="Apartment request",
                placeholder="e.g., 4 room apartment with 2 bathrooms and open livingroom",
            )
            bedrooms = gr.Number(label="Exact bedrooms", precision=0)
            bathrooms = gr.Number(label="Exact bathrooms", precision=0)
            balconies = gr.Number(label="Exact balconies", precision=0)
            search_btn = gr.Button("Search")

        with gr.Row():
            cards = gr.Gallery(
                label="Top candidate cards", columns=2, height=460, object_fit="contain"
            )
            with gr.Column(elem_classes=["section-card"]):
                candidate_preview = gr.Image(
                    label="Candidate preview", type="numpy", height=320
                )
                metadata = gr.Markdown("Select a card to inspect details.")
                select_plan_btn = gr.Button("Select this plan", variant="primary")

        with gr.Row():
            target_area = gr.Number(label="Target area sqm (optional)", value=None)
            gr.Markdown("Zoom with mouse wheel over the plan, or use +/-.")

        gr.Markdown("### Web Furniture Editor")
        with gr.Row():
            icon_dropdown = gr.Dropdown(
                icon_labels, value=default_icon_label, label="Icon"
            )
            mode = gr.Radio(
                ["point", "rect", "move", "delete"], value="point", label="Action"
            )
            zoom_out = gr.Button("-")
            zoom_in = gr.Button("+")
            zoom_value = gr.Number(
                label="Zoom", value=1.0, precision=1, elem_id="zoom-display"
            )
            undo_btn = gr.Button("Undo")
            clear_btn = gr.Button("Clear")
            done_btn = gr.Button("Done", variant="primary")

        editor = gr.Image(
            label="Plan editor canvas",
            type="numpy",
            height=CANVAS_H,
            elem_id="editor-canvas",
        )
        status = gr.Textbox(label="Status", lines=4)

        dxf_output = gr.File(label="DXF output")
        render_output = gr.Image(label="Rendered image", type="filepath")
        with gr.Row(visible=False, elem_id="overlay-box") as overlay_row:
            gr.Markdown(
                "### Generating...\nPlease wait while DXF/render files are generated."
            )

        search_btn.click(
            _search,
            inputs=[query, bedrooms, bathrooms, balconies, state],
            outputs=[
                state,
                cards,
                status,
                editor,
                metadata,
                candidate_preview,
                render_output,
            ],
        )

        cards.select(
            _preview_candidate,
            inputs=[state],
            outputs=[state, status, metadata, candidate_preview],
        )

        select_plan_btn.click(
            _select_candidate,
            inputs=[target_area, state],
            outputs=[state, editor, status, metadata, zoom_value],
        )

        icon_dropdown.change(
            _set_icon, inputs=[icon_dropdown, state], outputs=[state, status]
        )
        mode.change(_set_mode, inputs=[mode, state], outputs=[state, status])

        zoom_in.click(
            lambda s: _update_zoom(+ZOOM_STEP, s),
            inputs=[state],
            outputs=[state, editor, status, zoom_value],
        )
        zoom_out.click(
            lambda s: _update_zoom(-ZOOM_STEP, s),
            inputs=[state],
            outputs=[state, editor, status, zoom_value],
        )
        wheel_dispatch.click(
            _apply_wheel_delta,
            inputs=[wheel_payload, state],
            outputs=[state, editor, status, zoom_value],
        )

        editor.select(
            _apply_canvas_click, inputs=[state], outputs=[state, editor, status]
        )
        undo_btn.click(_undo, inputs=[state], outputs=[state, editor, status])
        clear_btn.click(_clear, inputs=[state], outputs=[state, editor, status])
        done_btn.click(
            _done_with_progress,
            inputs=[editor, state],
            outputs=[status, dxf_output, render_output, overlay_row],
        )

        # JS bridge: convert mouse wheel on the editor image into backend zoom events.
        gr.HTML(
            """
            <script>
            (() => {
              const bindWheel = () => {
                const root = document.querySelector('#editor-canvas');
                const payload = document.querySelector('#wheel-payload textarea, #wheel-payload input');
                const dispatchBtn = document.querySelector('#wheel-dispatch button');
                if (!root || !payload || !dispatchBtn) return false;
                const target = root.querySelector('img, canvas') || root;
                if (target.dataset.wheelBound === '1') return true;
                target.dataset.wheelBound = '1';
                target.addEventListener('wheel', (e) => {
                  e.preventDefault();
                  payload.value = JSON.stringify({deltaY: e.deltaY});
                  payload.dispatchEvent(new Event('input', { bubbles: true }));
                  dispatchBtn.click();
                }, { passive: false });
                return true;
              };
              if (!bindWheel()) {
                const obs = new MutationObserver(() => { bindWheel(); });
                obs.observe(document.body, { childList: true, subtree: true });
                setTimeout(() => obs.disconnect(), 15000);
              }
            })();
            </script>
            """
        )

    return demo


def main():
    app = create_app()
    app.queue().launch(
        allowed_paths=[config.PROJECT_ROOT_DIR],
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
        css=PAGE_CSS,
    )


if __name__ == "__main__":
    main()
