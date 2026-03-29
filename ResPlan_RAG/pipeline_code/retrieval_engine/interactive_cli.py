import os
import copy
from openai import OpenAI
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point

from ResPlan_RAG.pipeline_code import config

# Initialize openai API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))
logger = config.logger

def _prompt_icon_choice():
    keys = list(MANUAL_FURNITURE_CATALOG.keys())
    logger.info("Available icons:")
    for i, k in enumerate(keys, start=1):
        logger.info("  %2d. %s", i, _icon_display_label(k))

    while True:
        raw = input("Choose icon number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        logger.warning("Invalid icon selection.")


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


def _editor_room_polygons(playground):
    rooms = {}
    for key in ("living", "bedroom", "bathroom", "kitchen"):
        rooms[key] = _extract_polygons(_to_shapely(playground.get(key)))
    return rooms


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

            from ResPlan_RAG.pipeline_code.snapshot_pipeline.snapshot_to_render import run_post_done_render

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

    logger.info("--- CLI Furniture Editor Fallback ---")
    logger.info("Commands: add, list, undo, clear, done, cancel, help")

    while True:
        cmd = input("[CLI Editor] command: ").strip().lower()

        if cmd in {"help", "h"}:
            logger.info("add -> choose icon/mode and enter coordinates")
            logger.info("done -> confirm")
            logger.info("cancel -> reopen editor with previous confirmed items")
            continue

        if cmd in {"list", "ls"}:
            if not items:
                logger.info("  (no items)")
            for i, it in enumerate(items, start=1):
                logger.info("  %2d. %s", i, it)
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
            logger.warning("Unknown command.")
            continue

        icon = _prompt_icon_choice()
        cfg = MANUAL_FURNITURE_CATALOG[icon]
        mode = input("Mode [point/rect]: ").strip().lower()
        if mode not in {"point", "rect"}:
            logger.warning("Invalid mode.")
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
                    logger.warning("Rectangle dimensions are too small.")
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
            logger.warning("Invalid numeric input.")
            continue

        ok, reason = _validate_manual_item(item, room_polygons)
        if not ok:
            logger.warning("[CLI Editor][Rejected] %s", reason)
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
            logger.warning("[Editor] Interactive editor unavailable (%s). Falling back to CLI editor.", exc)
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
            logger.info("[Editor] Confirmed %s manual furniture item(s).", len(confirmed_items))
            return playground

        logger.info("[Editor] Cancel selected. Reopening with previous confirmed items.")
