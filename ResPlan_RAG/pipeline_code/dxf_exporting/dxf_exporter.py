import ezdxf
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry

from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporting_drawing_functions import _draw_icon_by_bbox, \
    _draw_dining_table_round, _canonical_icon_id
from dxf_exporting_drawing_functions import *
from room_level_placement_functions import (_place_bedroom, _place_kitchen, _place_bathroom, _place_living_room)


def _footprint_to_manual_item(icon, fp):
    """Convert a shapely footprint into editor item format."""
    canonical = _canonical_icon_id(icon)
    display_label = "working_table" if canonical == "desk" else ("working_chair" if canonical == "chair" else canonical)
    if fp.is_empty:
        return None

    if canonical == "dining_table_round":
        c = fp.centroid
        r = math.sqrt(max(fp.area, 1e-9) / math.pi)
        return {
            "icon": canonical,
            "display_label": display_label,
            "shape": "circle",
            "placement_mode": "auto",
            "x": float(c.x),
            "y": float(c.y),
            "r": float(r),
            "rotation": 0.0,
        }

    minx, miny, maxx, maxy = fp.bounds
    return {
        "icon": canonical,
        "display_label": display_label,
        "shape": "rect",
        "placement_mode": "auto",
        "x": float((minx + maxx) / 2.0),
        "y": float((miny + maxy) / 2.0),
        "w": float(maxx - minx),
        "h": float(maxy - miny),
        "rotation": 0.0,
    }


def _draw_manual_furniture_item(msp, item):
    """Draw a single manual furniture item saved by the editor."""
    icon = _canonical_icon_id(item.get("icon"))
    item["icon"] = icon
    shape_type = item.get("shape", "rect")

    if shape_type == "circle":
        x = float(item.get("x", 0.0))
        y = float(item.get("y", 0.0))
        r = float(item.get("r", 0.0))
        if r <= 0:
            return
        if icon != "dining_table_round":
            raise ValueError(f"Unsupported circular icon for DXF drawing: {icon}")
        _draw_dining_table_round(msp, x, y, r)
        return

    x = float(item.get("x", 0.0))
    y = float(item.get("y", 0.0))
    w = float(item.get("w", 0.0))
    h = float(item.get("h", 0.0))
    if w <= 0 or h <= 0:
        return

    x0 = x - w / 2.0
    y0 = y - h / 2.0
    x1 = x + w / 2.0
    y1 = y + h / 2.0

    _draw_icon_by_bbox(msp, icon, x0, y0, x1, y1, item=item)


def _draw_manual_furniture(msp, manual_furniture):
    items = manual_furniture.get("items", []) if isinstance(manual_furniture, dict) else []
    for item in items:
        _draw_manual_furniture_item(msp, item)
    return len(items)


# ═══════════════════════════════════════════════════════════════════
#  FURNITURE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def place_all_furniture(msp, playground):
    """
    Automatically place all furniture and fixtures:
      • Bedrooms  – largest → double bed (master); rest → single beds
      • Bathrooms – toilet + sink + shower in each
      • Kitchen   – fridge + double sink in each
      • Living    – sofa + coffee table + rug + TV + dining set (6 chairs)
    """

    manual_furniture = playground.get("manual_furniture")
    if isinstance(manual_furniture, dict) and manual_furniture.get("enabled"):
        count = _draw_manual_furniture(msp, manual_furniture)
        print(f"  [Furniture] Manual mode enabled: drew {count} item(s); auto-placement skipped.")
        return

    def to_shapely(geom_data):
        if isinstance(geom_data, BaseGeometry): return geom_data
        if isinstance(geom_data, dict) and 'type' in geom_data: return shape(geom_data)
        return None

    def extract_polygons(geom):
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, MultiPolygon):
            return [p for p in geom.geoms if not p.is_empty]
        if isinstance(geom, Polygon):
            return [geom]
        return []

    # ── Bedrooms ──────────────────────────────────────────────────
    door_geoms = extract_polygons(to_shapely(playground.get('door')))
    door_geoms += extract_polygons(to_shapely(playground.get('front_door')))

    bedrooms = extract_polygons(to_shapely(playground.get('bedroom')))
    if bedrooms:
        master_idx = max(range(len(bedrooms)), key=lambda i: bedrooms[i].area)
        for i, poly in enumerate(bedrooms):
            is_master = (i == master_idx)
            nearest_door = min(door_geoms, key=lambda d: poly.distance(d)) if door_geoms else None
            print(f"  [Furniture] {'Double' if is_master else 'Single'} bed → "
                  f"{'master bedroom' if is_master else f'bedroom {i + 1}'}")
            _place_bedroom(msp, poly, is_master, nearest_door)
    else:
        print("  [Furniture] No bedroom geometry found.")

    # ── Bathrooms ─────────────────────────────────────────────────
    bathrooms = extract_polygons(to_shapely(playground.get('bathroom')))
    for i, poly in enumerate(bathrooms):
        print(f"  [Furniture] Toilet + sink + shower → bathroom {i + 1}")
        _place_bathroom(msp, poly)
    if not bathrooms:
        print("  [Furniture] No bathroom geometry found.")

    # ── Kitchens ──────────────────────────────────────────────────
    kitchens = extract_polygons(to_shapely(playground.get('kitchen')))
    for i, poly in enumerate(kitchens):
        print(f"  [Furniture] Fridge + sink → kitchen {i + 1}")
        _place_kitchen(msp, poly)
    if not kitchens:
        print("  [Furniture] No kitchen geometry found.")

    # ── Living rooms ──────────────────────────────────────────────
    livings = extract_polygons(to_shapely(playground.get('living')))
    for i, poly in enumerate(livings):
        placed = _place_living_room(msp, poly, kitchen_geoms=kitchens, opening_geoms=door_geoms)
        if placed:
            print(f"  [Furniture] Sofa + coffee table + rug + TV + dining set → living {i + 1}")
        else:
            print(f"  [Furniture] Living {i + 1}: skipped (insufficient free space for full set)")
    if not livings:
        print("  [Furniture] No living geometry found.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXPORT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def export_to_dxf(playground, output_path="final_plan.dxf"):
    """
    Export playground to DXF with two phases:
      1. Structural elements – walls, windows, door swings, room outlines, labels
      2. Auto furniture      – beds, bathroom fixtures, kitchen appliances
    """
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Layer table
    doc.layers.new(name=L_WALLS,  dxfattribs={'color': 3})   # green
    doc.layers.new(name=L_ROOMS,  dxfattribs={'color': 8})   # grey
    doc.layers.new(name=L_DOORS,  dxfattribs={'color': 6})   # magenta
    doc.layers.new(name=L_LABELS, dxfattribs={'color': 2})   # yellow
    doc.layers.new(name=L_FURN,   dxfattribs={'color': 4})   # cyan   – beds

    # ── Geometry helpers ──────────────────────────────────────────

    def to_shapely(geom_data):
        if isinstance(geom_data, BaseGeometry): return geom_data
        if isinstance(geom_data, dict) and 'type' in geom_data: return shape(geom_data)
        return None

    def draw_polygon(geom, layer):
        if isinstance(geom, Polygon):
            msp.add_lwpolyline(list(geom.exterior.coords), close=True, dxfattribs={'layer': layer})
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': layer})

    def draw_door_swing(geom):
        polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        swing_ratio = 0.55  # < 1.0 makes the quarter swing narrower (quarter-ellipse)

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            w = maxx - minx
            h = maxy - miny

            msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={"layer": L_DOORS})

            if w > h:
                r_major = w
                r_minor = r_major * swing_ratio
                msp.add_line((minx, miny), (minx, miny + r_minor), dxfattribs={"layer": L_DOORS})
                msp.add_ellipse(
                    center=(minx, miny, 0.0),
                    major_axis=(r_major, 0.0, 0.0),
                    ratio=swing_ratio,
                    start_param=0.0,
                    end_param=1.5707963267948966,  # pi/2
                    dxfattribs={"layer": L_DOORS},
                )
            else:
                r_major = h
                r_minor = r_major * swing_ratio
                msp.add_line((minx, miny), (minx + r_minor, miny), dxfattribs={"layer": L_DOORS})
                msp.add_ellipse(
                    center=(minx, miny, 0.0),
                    major_axis=(0.0, r_major, 0.0),
                    ratio=swing_ratio,
                    start_param=-1.5707963267948966,  # -pi/2
                    end_param=0.0,
                    dxfattribs={"layer": L_DOORS},
                )

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1 — Structural elements
    # ─────────────────────────────────────────────────────────────
    print("[DXF Exporter] Drawing structural elements...")

    for key, val in playground.items():
        geom = to_shapely(val)
        if not geom or geom.is_empty:
            continue

        if key in ('wall', 'window'):
            draw_polygon(geom, L_WALLS)

        elif key in ('door', 'front_door'):
            draw_door_swing(geom)

        elif key in ('living', 'bedroom', 'bathroom', 'kitchen', 'balcony'):
            draw_polygon(geom, L_ROOMS)

            polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
            for i, poly in enumerate(polygons):
                cx, cy   = poly.centroid.x, poly.centroid.y
                room_name = key.upper() + (f" {i + 1}" if len(polygons) > 1 else "")
                msp.add_text(
                    room_name,
                    dxfattribs={'layer': L_LABELS, 'height': ROOM_LABEL_HEIGHT, 'color': 2}
                ).set_placement(
                    (cx, cy),
                    align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
                )

    # ─────────────────────────────────────────────────────────────
    #  PHASE 2 — Auto furniture placement
    # ─────────────────────────────────────────────────────────────
    print("[DXF Exporter] Auto-placing furniture and fixtures...")
    place_all_furniture(msp, playground)

    doc.saveas(output_path)
    print(f"\n[Success] DXF file saved to: {output_path}")