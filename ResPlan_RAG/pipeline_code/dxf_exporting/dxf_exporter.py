import math

import ezdxf
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry

# ─────────────────────────────────────────────
#  TUNEABLE CONSTANTS
# ─────────────────────────────────────────────
ROOM_LABEL_HEIGHT = 5.0   # text height for room name labels (drawing units)

# Layer names & ACI colours
L_WALLS  = 'WALLS'        # 3  green
L_ROOMS  = 'ROOMS'        # 8  grey
L_DOORS  = 'DOORS'        # 6  magenta
L_LABELS = 'ROOM_LABELS'  # 2  yellow
L_FURN   = 'FURNITURE'    # 4  cyan   – beds

# Living-room placement tuning
LIVING_ITEM_AREA_RATIO = 0.02         # each major living item = 2% of living area
LIVING_MIN_CLEARANCE = 0.05             # fixed minimum spacing from doors and boundaries
LIVING_KITCHEN_SCORE_WEIGHT = 0      # soft preference for dining near kitchen

# New zoning behavior for living placement
LIVING_TV_ZONE_RATIO = 0.38            # fraction of long-axis depth used for TV corner end-zone
LIVING_DINING_ZONE_RATIO = 0.38        # fraction of long-axis depth used for dining end-zone
LIVING_ZONE_INSET_RATIO = 0.04         # inset from side walls to keep circulation edges cleaner
LIVING_X_GAP_RATIO = 0.035             # X unit for TV->coffee->sofa spacing from room minor span

ICON_ALIAS_TO_CANONICAL = {
    "working_table": "desk",
    "working_chair": "chair",
    "tv_unit": "tv",
}


def _canonical_icon_id(icon_id):
    if not icon_id:
        return icon_id
    if str(icon_id).startswith("dining_chair"):
        return "dining_chair"
    return ICON_ALIAS_TO_CANONICAL.get(icon_id, icon_id)


# ═══════════════════════════════════════════════════════════════════
#  LOW-LEVEL DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def _rect(msp, x0, y0, x1, y1, layer):
    """Draw a closed rectangle as a lightweight polyline."""
    msp.add_lwpolyline(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        close=True, dxfattribs={'layer': layer}
    )


def _line(msp, x0, y0, x1, y1, layer):
    msp.add_line((x0, y0), (x1, y1), dxfattribs={'layer': layer})


def _circle(msp, cx, cy, r, layer):
    msp.add_circle((cx, cy), radius=r, dxfattribs={'layer': layer})


def _ellipse(msp, cx, cy, rx, ry, layer):
    """
    Draw an axis-aligned ellipse with automatic major-axis selection.

    Parameters:
      rx, ry: semi-axis lengths along X and Y intent (can be in any order).
    """
    rx = float(abs(rx))
    ry = float(abs(ry))

    if rx <= 0.0 or ry <= 0.0:
        raise ValueError(f"Ellipse radii must be > 0. Got rx={rx}, ry={ry}")

    # ezdxf expects:
    # - major_axis: vector from center to end of major semi-axis
    # - ratio: minor / major, strictly (0, 1]
    if rx >= ry:
        major_axis = (rx, 0.0, 0.0)   # major along X
        ratio = ry / rx
    else:
        major_axis = (0.0, ry, 0.0)   # major along Y
        ratio = rx / ry

    # Numerical safety (should already be in (0,1], but clamp tiny noise)
    ratio = max(min(ratio, 1.0), 1e-9)

    msp.add_ellipse(
        center=(float(cx), float(cy), 0.0),
        major_axis=major_axis,
        ratio=ratio,
        dxfattribs={"layer": layer},
    )


def _inset_box(x0, y0, x1, y1, inset_x_ratio=0.08, inset_y_ratio=None):
    """Return an inset rectangle, clamped so it always stays valid."""
    if inset_y_ratio is None:
        inset_y_ratio = inset_x_ratio

    w = max(x1 - x0, 1e-9)
    h = max(y1 - y0, 1e-9)
    ix = min(w * max(inset_x_ratio, 0.0), w * 0.45)
    iy = min(h * max(inset_y_ratio, 0.0), h * 0.45)
    return x0 + ix, y0 + iy, x1 - ix, y1 - iy


def _draw_dining_table_round(msp, cx, cy, r):
    """Round dining table with inner ring and crossed accents."""
    r = max(float(r), 1e-9)
    _circle(msp, cx, cy, r, L_FURN)
    _circle(msp, cx, cy, r * 0.62, L_FURN)
    _line(msp, cx - r * 0.62, cy, cx + r * 0.62, cy, L_FURN)
    _line(msp, cx, cy - r * 0.62, cx, cy + r * 0.62, L_FURN)


def _draw_bed_symbol_in_bbox(msp, x0, y0, x1, y1, *, is_double, head_side="top"):
    """Draw a consistent bed symbol inside a fixed bbox."""
    bw = max(x1 - x0, 1e-9)
    bh = max(y1 - y0, 1e-9)
    _rect(msp, x0, y0, x1, y1, L_FURN)

    # Headboard band on the requested side.
    hb = bh * 0.11
    if head_side == "bottom":
        _rect(msp, x0, y0, x1, y0 + hb, L_FURN)
        usable_y0, usable_y1 = y0 + hb, y1
        pillow_y = y0 + bh * 0.12
    else:
        _rect(msp, x0, y1 - hb, x1, y1, L_FURN)
        usable_y0, usable_y1 = y0, y1 - hb
        pillow_y = y1 - bh * 0.25

    # Inner contour for a cleaner, professional look.
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.06)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)

    # Center seam for double beds.
    if is_double:
        mid_x = (x0 + x1) / 2.0
        _line(msp, mid_x, usable_y0, mid_x, usable_y1, L_FURN)

    # Pillow blocks.
    ph = bh * 0.12
    py = max(y0 + bh * 0.05, min(pillow_y, y1 - ph - bh * 0.05))
    if is_double:
        half_w = bw / 2.0
        pw = half_w * 0.58
        left_cx = x0 + half_w * 0.5
        right_cx = x0 + half_w * 1.5
        _rect(msp, left_cx - pw / 2.0, py, left_cx + pw / 2.0, py + ph, L_FURN)
        _rect(msp, right_cx - pw / 2.0, py, right_cx + pw / 2.0, py + ph, L_FURN)
    else:
        pw = bw * 0.58
        px = x0 + (bw - pw) / 2.0
        _rect(msp, px, py, px + pw, py + ph, L_FURN)


# ═══════════════════════════════════════════════════════════════════
#  FURNITURE DRAWING FUNCTIONS
#  Every function receives the room polygon bounds and sizes itself
#  proportionally — no hard-coded unit assumptions.
# ═══════════════════════════════════════════════════════════════════

def _draw_double_bed(msp, minx, miny, maxx, maxy, door_pt=None):
    """
    Double bed placed in the corner opposite to the door-nearest corner.
    Falls back to bottom-left if no door point is provided.
    """
    rw = maxx - minx
    rh = maxy - miny
    if rw <= 0 or rh <= 0:
        return

    m = min(rw, rh) * 0.05

    # Keep existing double-bed proportions, then clamp to available space.
    bed_w = min(rw * 0.62, rh * 0.90)
    bed_h = bed_w / 1.40

    avail_w = max(rw - 2 * m, 1e-6)
    avail_h = max(rh - 2 * m, 1e-6)
    fit = min(avail_w / bed_w, avail_h / bed_h, 1.0)
    bed_w *= fit
    bed_h *= fit

    nearest_corner = "bl"
    if door_pt is not None:
        dx, dy = door_pt
        corners = {
            "bl": (minx, miny),
            "br": (maxx, miny),
            "tl": (minx, maxy),
            "tr": (maxx, maxy),
        }
        nearest_corner = min(
            corners,
            key=lambda c: (corners[c][0] - dx) ** 2 + (corners[c][1] - dy) ** 2
        )

    opposite = {"bl": "tr", "br": "tl", "tl": "br", "tr": "bl"}[nearest_corner]
    head_side = "bottom" if opposite in ("bl", "br") else "top"

    if opposite == "bl":
        bx0, by0 = minx + m, miny + m
        bx1, by1 = bx0 + bed_w, by0 + bed_h
    elif opposite == "br":
        bx1, by0 = maxx - m, miny + m
        bx0, by1 = bx1 - bed_w, by0 + bed_h
    elif opposite == "tl":
        bx0, by1 = minx + m, maxy - m
        bx1, by0 = bx0 + bed_w, by1 - bed_h
    else:  # "tr"
        bx1, by1 = maxx - m, maxy - m
        bx0, by0 = bx1 - bed_w, by1 - bed_h

    _draw_bed_symbol_in_bbox(msp, bx0, by0, bx1, by1, is_double=True, head_side=head_side)


def _draw_single_bed(msp, minx, miny, maxx, maxy, door_pt=None):
    """
    Single bed placed in the corner opposite to the door-nearest corner.
    Falls back to bottom-left if no door point is provided.
    """
    rw = maxx - minx
    rh = maxy - miny
    if rw <= 0 or rh <= 0:
        return

    m = min(rw, rh) * 0.05

    # Scale from room area, keep ~1:2 width:length ratio.
    room_area = rw * rh
    target_bed_area = max(room_area * 0.16, 1e-6)
    bed_w = (target_bed_area / 2.0) ** 0.5
    bed_h = bed_w * 2.0

    avail_w = max(rw - 2 * m, 1e-6)
    avail_h = max(rh - 2 * m, 1e-6)
    fit = min(avail_w / bed_w, avail_h / bed_h, 1.0)
    bed_w *= fit
    bed_h *= fit

    # Find corner nearest to door point, then place bed in opposite corner.
    nearest_corner = "bl"
    if door_pt is not None:
        dx, dy = door_pt
        corners = {
            "bl": (minx, miny),
            "br": (maxx, miny),
            "tl": (minx, maxy),
            "tr": (maxx, maxy),
        }
        nearest_corner = min(
            corners,
            key=lambda c: (corners[c][0] - dx) ** 2 + (corners[c][1] - dy) ** 2
        )

    opposite = {"bl": "tr", "br": "tl", "tl": "br", "tr": "bl"}[nearest_corner]
    head_side = "bottom" if opposite in ("bl", "br") else "top"

    if opposite == "bl":
        bx0, by0 = minx + m, miny + m
        bx1, by1 = bx0 + bed_w, by0 + bed_h
    elif opposite == "br":
        bx1, by0 = maxx - m, miny + m
        bx0, by1 = bx1 - bed_w, by0 + bed_h
    elif opposite == "tl":
        bx0, by1 = minx + m, maxy - m
        bx1, by0 = bx0 + bed_w, by1 - bed_h
    else:  # "tr"
        bx1, by1 = maxx - m, maxy - m
        bx0, by0 = bx1 - bed_w, by1 - bed_h

    _draw_bed_symbol_in_bbox(msp, bx0, by0, bx1, by1, is_double=False, head_side=head_side)


def _draw_double_bed_in_bbox(msp, x0, y0, x1, y1, head_side="top"):
    """Draw a double-bed symbol strictly inside the provided bbox (no rescaling)."""
    _draw_bed_symbol_in_bbox(msp, x0, y0, x1, y1, is_double=True, head_side=head_side)


def _draw_single_bed_in_bbox(msp, x0, y0, x1, y1, head_side="top"):
    """Draw a single-bed symbol strictly inside the provided bbox (no rescaling)."""
    _draw_bed_symbol_in_bbox(msp, x0, y0, x1, y1, is_double=False, head_side=head_side)


def _draw_toilet(msp, x0, y0, w, h):
    """
    Toilet: outer rectangle + tank band at base + elliptical bowl above.
    """
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.10)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)

    tank_h = h * 0.24
    _rect(msp, x0 + w * 0.10, y0, x1 - w * 0.10, y0 + tank_h, L_FURN)
    bowl_cy = y0 + tank_h + (h - tank_h) * 0.48
    _ellipse(msp, x0 + w / 2.0, bowl_cy, w * 0.30, (h - tank_h) * 0.34, L_FURN)
    _circle(msp, x0 + w / 2.0, bowl_cy, min(w, h) * 0.07, L_FURN)


def _draw_sink(msp, x0, y0, w, h):
    """
    Bathroom sink: rectangle + circular basin + tap stub.
    """
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.12)
    _ellipse(msp, (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0, (ix1 - ix0) * 0.34, (iy1 - iy0) * 0.34, L_FURN)
    _circle(msp, (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0, min(w, h) * 0.05, L_FURN)
    tap_x = x0 + w * 0.50
    _line(msp, tap_x, y1 - h * 0.10, tap_x, y1 + h * 0.10, L_FURN)
    _line(msp, tap_x - w * 0.08, y1 + h * 0.10, tap_x + w * 0.08, y1 + h * 0.10, L_FURN)


def _draw_shower(msp, x0, y0, size):
    """
    Shower tray: square + X cross + central showerhead circle.
    """
    x1, y1 = x0 + size, y0 + size
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.10)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    _line(msp, ix0, iy0, ix1, iy1, L_FURN)
    _line(msp, ix1, iy0, ix0, iy1, L_FURN)
    _circle(msp, (x0 + x1) / 2.0, (y0 + y1) / 2.0, size * 0.10, L_FURN)
    _circle(msp, x0 + size * 0.18, y1 - size * 0.18, size * 0.06, L_FURN)


def _draw_kitchen_sink(msp, x0, y0, w, h):
    """
    Kitchen sink: outer rectangle + two basins side by side + tap divider.
    """
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)

    m = min(w, h) * 0.09
    hw = max((w - 3 * m) / 2.0, 1e-9)
    _rect(msp, x0 + m, y0 + m, x0 + m + hw, y1 - m, L_FURN)
    _rect(msp, x0 + 2 * m + hw, y0 + m, x1 - m, y1 - m, L_FURN)

    div_x = x0 + m + hw + m / 2.0
    _line(msp, div_x, y0 + m, div_x, y1 - m, L_FURN)
    _circle(msp, div_x, y1 + h * 0.08, min(w, h) * 0.05, L_FURN)


def _draw_fridge(msp, x0, y0, w, h):
    """
    Fridge: outer rectangle + lower freezer compartment + door handle.
    """
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.08)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)

    freeze_h = h * 0.30
    _line(msp, x0, y0 + freeze_h, x1, y0 + freeze_h, L_FURN)
    hx = x0 + w * 0.86
    _line(msp, hx, y0 + freeze_h + h * 0.08, hx, y1 - h * 0.12, L_FURN)
    _line(msp, hx, y0 + h * 0.05, hx, y0 + freeze_h - h * 0.05, L_FURN)


def _draw_tv(msp, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.10)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    _line(msp, x0 + w * 0.50, y0, x0 + w * 0.50, y0 - h * 0.12, L_FURN)
    _line(msp, x0 + w * 0.34, y0 - h * 0.12, x0 + w * 0.66, y0 - h * 0.12, L_FURN)


def _draw_coffee_table(msp, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.10)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    _line(msp, x0 + w * 0.50, y0 + h * 0.12, x0 + w * 0.50, y1 - h * 0.12, L_FURN)
    _line(msp, x0 + w * 0.15, y0 + h * 0.50, x1 - w * 0.15, y0 + h * 0.50, L_FURN)


def _draw_dining_table_rect(msp, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.09)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    leg_r = min(w, h) * 0.06
    _circle(msp, x0 + w * 0.18, y0 + h * 0.20, leg_r, L_FURN)
    _circle(msp, x1 - w * 0.18, y0 + h * 0.20, leg_r, L_FURN)
    _circle(msp, x0 + w * 0.18, y1 - h * 0.20, leg_r, L_FURN)
    _circle(msp, x1 - w * 0.18, y1 - h * 0.20, leg_r, L_FURN)


def _draw_sofa(msp, x0, y0, w, h):
    """Simple professional 3-seater sofa symbol."""
    if w <= 0 or h <= 0:
        return

    x1, y1 = x0 + w, y0 + h

    # Outer frame
    _rect(msp, x0, y0, x1, y1, L_FURN)

    # Armrests
    arm = min(w, h) * 0.10
    _rect(msp, x0, y0, x0 + arm, y1, L_FURN)
    _rect(msp, x1 - arm, y0, x1, y1, L_FURN)

    # Backrest
    back_h = h * 0.22
    _rect(msp, x0 + arm, y1 - back_h, x1 - arm, y1, L_FURN)

    # Seat block
    sx0, sx1 = x0 + arm, x1 - arm
    sy0, sy1 = y0 + h * 0.15, y1 - back_h
    _rect(msp, sx0, sy0, sx1, sy1, L_FURN)

    # 3-seat seams
    _line(msp, sx0 + (sx1 - sx0) / 3.0, sy0, sx0 + (sx1 - sx0) / 3.0, sy1, L_FURN)
    _line(msp, sx0 + 2.0 * (sx1 - sx0) / 3.0, sy0, sx0 + 2.0 * (sx1 - sx0) / 3.0, sy1, L_FURN)


def _draw_rug(msp, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.08)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    _line(msp, ix0, (iy0 + iy1) / 2.0, ix1, (iy0 + iy1) / 2.0, L_FURN)
    _line(msp, (ix0 + ix1) / 2.0, iy0, (ix0 + ix1) / 2.0, iy1, L_FURN)


def _draw_dining_chair(msp, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    _rect(msp, x0 + w * 0.14, y0 + h * 0.18, x1 - w * 0.14, y0 + h * 0.58, L_FURN)
    _line(msp, x0 + w * 0.18, y0 + h * 0.80, x1 - w * 0.18, y0 + h * 0.80, L_FURN)
    _line(msp, x0 + w * 0.20, y0 + h * 0.18, x0 + w * 0.20, y0 + h * 0.05, L_FURN)
    _line(msp, x1 - w * 0.20, y0 + h * 0.18, x1 - w * 0.20, y0 + h * 0.05, L_FURN)


def _draw_icon_by_bbox(msp, icon, x0, y0, x1, y1, item=None):
    icon = _canonical_icon_id(icon)
    w = x1 - x0
    h = y1 - y0

    if icon == "single_bed":
        _draw_single_bed_in_bbox(msp, x0, y0, x1, y1, head_side=(item or {}).get("head_side", "top"))
    elif icon == "double_bed":
        _draw_double_bed_in_bbox(msp, x0, y0, x1, y1, head_side=(item or {}).get("head_side", "top"))
    elif icon == "toilet":
        _draw_toilet(msp, x0, y0, w, h)
    elif icon == "sink":
        _draw_sink(msp, x0, y0, w, h)
    elif icon == "shower":
        _draw_shower(msp, x0, y0, min(w, h))
    elif icon == "kitchen_sink":
        _draw_kitchen_sink(msp, x0, y0, w, h)
    elif icon == "fridge":
        _draw_fridge(msp, x0, y0, w, h)
    elif icon == "closet":
        _draw_closet(msp, x0, y0, w, h)
    elif icon == "dresser":
        _draw_dresser(msp, x0, y0, w, h)
    elif icon == "desk":
        _draw_work_desk(msp, x0, y0, w, h)
    elif icon == "chair":
        _draw_office_chair(msp, x0, y0, w, h)
    elif icon == "tv":
        _draw_tv(msp, x0, y0, w, h)
    elif icon == "coffee_table":
        _draw_coffee_table(msp, x0, y0, w, h)
    elif icon == "dining_table_rect":
        _draw_dining_table_rect(msp, x0, y0, w, h)
    elif icon == "sofa":
        _draw_sofa(msp, x0, y0, w, h)
    elif icon == "rug":
        _draw_rug(msp, x0, y0, w, h)
    elif icon == "dining_chair":
        _draw_dining_chair(msp, x0, y0, w, h)
    else:
        raise ValueError(f"Unsupported furniture icon for DXF drawing: {icon}")


def _draw_icon_from_footprint(msp, icon, fp, item=None):
    icon = _canonical_icon_id(icon)
    if icon == "dining_table_round":
        c = fp.centroid
        r = math.sqrt(max(fp.area, 1e-9) / math.pi)
        _draw_dining_table_round(msp, float(c.x), float(c.y), float(r))
        return

    minx, miny, maxx, maxy = fp.bounds
    _draw_icon_by_bbox(msp, icon, minx, miny, maxx, maxy, item=item)


def _draw_closet(msp, x0, y0, w, h):
    """Closet: rectangle with vertical panel divisions."""
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.07)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    if w >= h:
        x1 = x0 + w / 3.0
        x2 = x0 + 2.0 * w / 3.0
        _line(msp, x1, iy0, x1, iy1, L_FURN)
        _line(msp, x2, iy0, x2, iy1, L_FURN)
        _circle(msp, x0 + w * 0.49, y0 + h * 0.50, min(w, h) * 0.03, L_FURN)
        _circle(msp, x0 + w * 0.51, y0 + h * 0.50, min(w, h) * 0.03, L_FURN)
    else:
        y1 = y0 + h / 3.0
        y2 = y0 + 2.0 * h / 3.0
        _line(msp, ix0, y1, ix1, y1, L_FURN)
        _line(msp, ix0, y2, ix1, y2, L_FURN)


def _draw_dresser(msp, x0, y0, w, h):
    """Small dresser table: rectangle with two drawer lines."""
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    ix0, iy0, ix1, iy1 = _inset_box(x0, y0, x1, y1, 0.08)
    _rect(msp, ix0, iy0, ix1, iy1, L_FURN)
    y_a = y0 + h * 0.33
    y_b = y0 + h * 0.66
    _line(msp, x0 + w * 0.10, y_a, x1 - w * 0.10, y_a, L_FURN)
    _line(msp, x0 + w * 0.10, y_b, x1 - w * 0.10, y_b, L_FURN)
    _circle(msp, x0 + w * 0.50, y0 + h * 0.16, min(w, h) * 0.03, L_FURN)
    _circle(msp, x0 + w * 0.50, y0 + h * 0.49, min(w, h) * 0.03, L_FURN)
    _circle(msp, x0 + w * 0.50, y0 + h * 0.82, min(w, h) * 0.03, L_FURN)


def _draw_work_desk(msp, x0, y0, w, h):
    """Work desk: rectangle with a narrow top edge accent."""
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0, y0, x1, y1, L_FURN)
    _line(msp, x0 + w * 0.08, y0 + h * 0.82, x1 - w * 0.08, y0 + h * 0.82, L_FURN)
    _line(msp, x0 + w * 0.18, y0, x0 + w * 0.18, y0 + h * 0.16, L_FURN)
    _line(msp, x1 - w * 0.18, y0, x1 - w * 0.18, y0 + h * 0.16, L_FURN)
    _rect(msp, x0 + w * 0.22, y0 + h * 0.30, x1 - w * 0.22, y0 + h * 0.62, L_FURN)


def _draw_office_chair(msp, x0, y0, w, h):
    """Office chair: seat rectangle + circular wheel base hint."""
    x1, y1 = x0 + w, y0 + h
    _rect(msp, x0 + w * 0.18, y0 + h * 0.18, x1 - w * 0.18, y0 + h * 0.56, L_FURN)
    _rect(msp, x0 + w * 0.24, y0 + h * 0.62, x1 - w * 0.24, y0 + h * 0.90, L_FURN)
    cx = x0 + w / 2.0
    cy = y0 + h * 0.16
    _circle(msp, cx, cy, min(w, h) * 0.14, L_FURN)
    _line(msp, cx, y0 + h * 0.18, cx, y0 + h * 0.56, L_FURN)
    _line(msp, cx, cy, cx + w * 0.18, cy, L_FURN)
    _line(msp, cx, cy, cx - w * 0.18, cy, L_FURN)


def _single_bed_reference_size(minx, miny, maxx, maxy):
    """Return single-bed reference size for scaling accessory furniture."""
    rw = maxx - minx
    rh = maxy - miny
    if rw <= 0 or rh <= 0:
        return 1.0, 2.0

    m = min(rw, rh) * 0.05
    room_area = rw * rh
    target_bed_area = max(room_area * 0.16, 1e-6)
    bed_w = (target_bed_area / 2.0) ** 0.5
    bed_h = bed_w * 2.0

    avail_w = max(rw - 2 * m, 1e-6)
    avail_h = max(rh - 2 * m, 1e-6)
    fit = min(avail_w / bed_w, avail_h / bed_h, 1.0)
    return bed_w * fit, bed_h * fit


def _compute_bed_box(minx, miny, maxx, maxy, is_master, door_pt=None):
    """Compute bed bounding box consistent with current bed draw logic."""
    rw = maxx - minx
    rh = maxy - miny
    if rw <= 0 or rh <= 0:
        return None

    m = min(rw, rh) * 0.05

    if is_master:
        bed_w = min(rw * 0.62, rh * 0.90)
        bed_h = bed_w / 1.40
    else:
        room_area = rw * rh
        target_bed_area = max(room_area * 0.16, 1e-6)
        bed_w = (target_bed_area / 2.0) ** 0.5
        bed_h = bed_w * 2.0

    avail_w = max(rw - 2 * m, 1e-6)
    avail_h = max(rh - 2 * m, 1e-6)
    fit = min(avail_w / bed_w, avail_h / bed_h, 1.0)
    bed_w *= fit
    bed_h *= fit

    nearest_corner = "bl"
    if door_pt is not None:
        dx, dy = door_pt
        corners = {
            "bl": (minx, miny),
            "br": (maxx, miny),
            "tl": (minx, maxy),
            "tr": (maxx, maxy),
        }
        nearest_corner = min(corners, key=lambda c: (corners[c][0] - dx) ** 2 + (corners[c][1] - dy) ** 2)

    opposite = {"bl": "tr", "br": "tl", "tl": "br", "tr": "bl"}[nearest_corner]

    if opposite == "bl":
        bx0, by0 = minx + m, miny + m
        bx1, by1 = bx0 + bed_w, by0 + bed_h
    elif opposite == "br":
        bx1, by0 = maxx - m, miny + m
        bx0, by1 = bx1 - bed_w, by0 + bed_h
    elif opposite == "tl":
        bx0, by1 = minx + m, maxy - m
        bx1, by0 = bx0 + bed_w, by1 - bed_h
    else:
        bx1, by1 = maxx - m, maxy - m
        bx0, by0 = bx1 - bed_w, by1 - bed_h

    # Keep head-side semantics consistent with DXF bed symbol orientation.
    head_side = "bottom" if opposite in ("bl", "br") else "top"

    return {
        "bbox": (bx0, by0, bx1, by1),
        "margin": m,
        "head_side": head_side,
    }


def _bed_wall_from_bbox(minx, miny, maxx, maxy, bed_bbox):
    bx0, by0, bx1, by1 = bed_bbox
    d = {
        "left": abs(bx0 - minx),
        "right": abs(maxx - bx1),
        "bottom": abs(by0 - miny),
        "top": abs(maxy - by1),
    }
    return min(d, key=d.get)


def _bbox_polygon(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _is_valid_piece(room_poly, fp, occupied, door_geom, door_clearance):
    if fp.is_empty or fp.area <= 1e-9:
        return False

    soft_tol = max(room_poly.length * 1e-9, 1e-4)
    if not room_poly.buffer(soft_tol).covers(fp):
        return False

    if door_geom is not None and not door_geom.is_empty:
        if fp.distance(door_geom) < door_clearance:
            return False

    for other in occupied:
        if fp.intersects(other) or fp.distance(other) < 0.05:
            return False

    return True


def _dims_from_area_ratio(area, ratio_wh):
    w = (max(area, 1e-9) * max(ratio_wh, 1e-9)) ** 0.5
    h = max(area, 1e-9) / w
    return w, h


def _draw_piece_by_name(msp, name, bbox):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    if name == "closet":
        _draw_closet(msp, x0, y0, w, h)
    elif name == "dresser":
        _draw_dresser(msp, x0, y0, w, h)
    elif name == "desk":
        _draw_work_desk(msp, x0, y0, w, h)
    elif name == "chair":
        _draw_office_chair(msp, x0, y0, w, h)


def _draw_poly_outline(msp, poly, layer):
    """Draw polygon exterior as a closed polyline."""
    if poly.is_empty:
        return
    msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': layer})


def _living_local_frame(polygon):
    """Build a local (u, v) frame from minimum_rotated_rectangle."""
    mrr = polygon.minimum_rotated_rectangle
    pts = list(mrr.exterior.coords)[:4]
    edges = []
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        edges.append((ln, dx, dy))

    _, ex, ey = max(edges, key=lambda e: e[0])
    u_len = max(math.hypot(ex, ey), 1e-9)
    ux, uy = ex / u_len, ey / u_len
    vx, vy = -uy, ux

    cx, cy = polygon.centroid.x, polygon.centroid.y

    def project_xy(x, y):
        px, py = x - cx, y - cy
        return px * ux + py * uy, px * vx + py * vy

    uv = [project_xy(x, y) for x, y in polygon.exterior.coords]
    u_vals = [u for u, _ in uv]
    v_vals = [v for _, v in uv]
    u_min, u_max = min(u_vals), max(u_vals)
    v_min, v_max = min(v_vals), max(v_vals)

    return {
        'cx': cx,
        'cy': cy,
        'ux': ux,
        'uy': uy,
        'vx': vx,
        'vy': vy,
        'u_min': u_min,
        'u_max': u_max,
        'v_min': v_min,
        'v_max': v_max,
        'diag': math.hypot(u_max - u_min, v_max - v_min),
    }


def _local_to_world(frame, u, v):
    x = frame['cx'] + u * frame['ux'] + v * frame['vx']
    y = frame['cy'] + u * frame['uy'] + v * frame['vy']
    return x, y


def _project_to_local(frame, x, y):
    px, py = x - frame['cx'], y - frame['cy']
    u = px * frame['ux'] + py * frame['uy']
    v = px * frame['vx'] + py * frame['vy']
    return u, v


def _local_rect_polygon(frame, u, v, w, h):
    hu, hv = w / 2.0, h / 2.0
    corners_local = [
        (u - hu, v - hv),
        (u + hu, v - hv),
        (u + hu, v + hv),
        (u - hu, v + hv),
    ]
    return Polygon([_local_to_world(frame, cu, cv) for cu, cv in corners_local])


def _dims_from_area(target_area, ratio_wh):
    w = math.sqrt(max(target_area, 1e-9) * max(ratio_wh, 1e-9))
    h = max(target_area, 1e-9) / w
    return w, h


def _fit_rect_inside(living_poly, frame, w, h, target_u, target_v, grid=13, avoid_geoms=None, min_sep=0.0):
    """Find a near-target local center where an oriented rectangle fits inside living polygon."""
    soft_tol = max(living_poly.length * 1e-9, 1e-4)
    avoid_geoms = avoid_geoms or []
    min_sep = max(float(min_sep), 0.0)

    for scale in (1.0, 0.95, 0.90, 0.85, 0.80):
        ws = w * scale
        hs = h * scale
        u_min = frame['u_min'] + ws / 2.0
        u_max = frame['u_max'] - ws / 2.0
        v_min = frame['v_min'] + hs / 2.0
        v_max = frame['v_max'] - hs / 2.0
        if u_min > u_max or v_min > v_max:
            continue

        tu = min(max(target_u, u_min), u_max)
        tv = min(max(target_v, v_min), v_max)

        u_vals = [u_min + (u_max - u_min) * (i / max(grid - 1, 1)) for i in range(grid)]
        v_vals = [v_min + (v_max - v_min) * (i / max(grid - 1, 1)) for i in range(grid)]
        u_vals.append(tu)
        v_vals.append(tv)

        best = None
        best_score = None
        for u in u_vals:
            for v in v_vals:
                fp = _local_rect_polygon(frame, u, v, ws, hs)
                if not living_poly.buffer(soft_tol).covers(fp):
                    continue

                blocked = False
                for g in avoid_geoms:
                    if fp.distance(g) < min_sep:
                        blocked = True
                        break
                if blocked:
                    continue

                score = (u - tu) ** 2 + (v - tv) ** 2
                if best_score is None or score < best_score:
                    best_score = score
                    best = {'score': score, 'u': u, 'v': v, 'w': ws, 'h': hs, 'footprint': fp}

        if best is not None:
            return best

    return None


def _fit_circle_inside(living_poly, frame, r, target_u, target_v, grid=13):
    """Find a near-target local center where a circle fits inside living polygon."""
    soft_tol = max(living_poly.length * 1e-9, 1e-4)

    for scale in (1.0, 0.95, 0.90, 0.85, 0.80):
        rs = r * scale
        u_min = frame['u_min'] + rs
        u_max = frame['u_max'] - rs
        v_min = frame['v_min'] + rs
        v_max = frame['v_max'] - rs
        if u_min > u_max or v_min > v_max:
            continue

        tu = min(max(target_u, u_min), u_max)
        tv = min(max(target_v, v_min), v_max)
        u_vals = [u_min + (u_max - u_min) * (i / max(grid - 1, 1)) for i in range(grid)]
        v_vals = [v_min + (v_max - v_min) * (i / max(grid - 1, 1)) for i in range(grid)]
        u_vals.append(tu)
        v_vals.append(tv)

        best = None
        best_score = None
        for u in u_vals:
            for v in v_vals:
                cx, cy = _local_to_world(frame, u, v)
                fp = Point(cx, cy).buffer(rs, resolution=24)
                if not living_poly.buffer(soft_tol).covers(fp):
                    continue
                score = (u - tu) ** 2 + (v - tv) ** 2
                if best_score is None or score < best_score:
                    best_score = score
                    best = {'score': score, 'u': u, 'v': v, 'r': rs, 'footprint': fp}

        if best is not None:
            return best

    return None


def _is_layout_valid(living_poly, items, opening_geoms, clearance):
    tol = 1e-6
    soft_tol = max(living_poly.length * 1e-9, 1e-4)
    major_for_door_clearance = {'sofa', 'dining_table_rect', 'dining_table_round'}

    for item in items:
        fp = item['footprint']
        if fp.is_empty:
            print(f"[LivingLayout][Invalid] empty footprint for item={item['name']} at _is_layout_valid:383")
            return False
        inner = living_poly.buffer(-tol)
        # Accept boundary-touching footprints; reject only true outside overflow.
        if not inner.is_empty:
            if not (inner.covers(fp) or living_poly.buffer(soft_tol).covers(fp)):
                print(f"[LivingLayout][Invalid] item outside living polygon item={item['name']} at _is_layout_valid:390")
                return False
        else:
            if not living_poly.buffer(soft_tol).covers(fp):
                print(f"[LivingLayout][Invalid] item outside living polygon (thin buffer path) item={item['name']} at _is_layout_valid:394")
                return False

        if item['name'] in major_for_door_clearance:
            for opening in opening_geoms:
                if fp.distance(opening) < clearance:
                    print(f"[LivingLayout][Invalid] door/opening clearance violation item={item['name']} dist={fp.distance(opening):.4f} clearance={clearance:.4f} at _is_layout_valid:393")
                    return False

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i]
            b = items[j]
            a_name = a['name']
            b_name = b['name']

            if 'rug' in {a_name, b_name}:
                continue
            if a_name.startswith('dining_chair') and b_name.startswith('dining_chair'):
                continue
            if (a_name.startswith('dining_chair') and b_name.startswith('dining_table')) or \
               (b_name.startswith('dining_chair') and a_name.startswith('dining_table')):
                continue

            if a['footprint'].distance(b['footprint']) < 0.05:
                print(f"[LivingLayout][Invalid] collision a={a_name} b={b_name} dist={a['footprint'].distance(b['footprint']):.4f} threshold=0.0500 at _is_layout_valid:411")
                return False

    return True


def _score_layout(items, kitchen_geoms, weight, diag):
    score = 0.0
    table = next((it for it in items if it['name'] in {'dining_table_rect', 'dining_table_round'}), None)
    if table and kitchen_geoms:
        d = min(table['footprint'].distance(k) for k in kitchen_geoms)
        score += weight * (1.0 - min(d / max(diag, 1e-6), 1.0))
    return score


def _draw_living_items(msp, items):
    for item in items:
        name = item['name']
        fp = item['footprint']
        _draw_icon_from_footprint(msp, name, fp)


def _build_dining_rect(frame, u_center, v_center, area_budget, clearance):
    table_area = area_budget * 0.60
    chair_area = max((area_budget - table_area) / 6.0, 1e-9)

    tw, th = _dims_from_area(table_area, ratio_wh=1.60)
    cw, ch = _dims_from_area(chair_area, ratio_wh=0.90)

    items = []
    table_fp = _local_rect_polygon(frame, u_center, v_center, tw, th)
    items.append({'name': 'dining_table_rect', 'footprint': table_fp})

    step_u = tw / 3.0
    side_gap = (th / 2.0) + (ch / 2.0) + clearance * 0.35
    chair_u_positions = [u_center - step_u, u_center, u_center + step_u]
    for idx, cu in enumerate(chair_u_positions, 1):
        items.append({'name': f'dining_chair_{idx}', 'footprint': _local_rect_polygon(frame, cu, v_center + side_gap, cw, ch)})
    for idx, cu in enumerate(chair_u_positions, 4):
        items.append({'name': f'dining_chair_{idx}', 'footprint': _local_rect_polygon(frame, cu, v_center - side_gap, cw, ch)})

    return items


def _build_dining_round(frame, u_center, v_center, area_budget, clearance):
    table_area = area_budget * 0.60
    chair_area = max((area_budget - table_area) / 6.0, 1e-9)

    r_table = math.sqrt(table_area / math.pi)
    chair_w, chair_h = _dims_from_area(chair_area, ratio_wh=1.0)
    r_ring = r_table + max(chair_w, chair_h) * 0.70 + clearance * 0.25

    items = []
    table_fp = Point(*_local_to_world(frame, u_center, v_center)).buffer(r_table, resolution=24)
    items.append({'name': 'dining_table_round', 'footprint': table_fp})

    for idx in range(6):
        a = (2.0 * math.pi / 6.0) * idx
        cu = u_center + r_ring * math.cos(a)
        cv = v_center + r_ring * math.sin(a)
        items.append({
            'name': f'dining_chair_{idx + 1}',
            'footprint': _local_rect_polygon(frame, cu, cv, chair_w, chair_h),
        })

    return items


def _local_bounds_of_geom(frame, geom):
    if geom.is_empty:
        return None

    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if not p.is_empty]
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]

    if not polys:
        return None

    u_vals = []
    v_vals = []
    for poly in polys:
        for x, y in poly.exterior.coords:
            u, v = _project_to_local(frame, x, y)
            u_vals.append(u)
            v_vals.append(v)

    return min(u_vals), min(v_vals), max(u_vals), max(v_vals)


def _end_zone_polygon(living_poly, frame, axis, side, depth, inset):
    """
    Create one end-zone rectangle (in local frame), then clip it by living polygon.
    side = -1 means min side of the split axis, +1 means max side.
    """
    total_u = frame['u_max'] - frame['u_min']
    total_v = frame['v_max'] - frame['v_min']
    if axis == 'u':
        zone_w = max(depth, 1e-6)
        zone_h = max(total_v - 2.0 * inset, 1e-6)
        center_v = (frame['v_min'] + frame['v_max']) / 2.0
        if side < 0:
            center_u = frame['u_min'] + inset + zone_w / 2.0
        else:
            center_u = frame['u_max'] - inset - zone_w / 2.0
        raw = _local_rect_polygon(frame, center_u, center_v, zone_w, zone_h)
    else:
        zone_h = max(depth, 1e-6)
        zone_w = max(total_u - 2.0 * inset, 1e-6)
        center_u = (frame['u_min'] + frame['u_max']) / 2.0
        if side < 0:
            center_v = frame['v_min'] + inset + zone_h / 2.0
        else:
            center_v = frame['v_max'] - inset - zone_h / 2.0
        raw = _local_rect_polygon(frame, center_u, center_v, zone_w, zone_h)
    return living_poly.intersection(raw)


def _all_items_inside(container_poly, items):
    if container_poly.is_empty:
        return False
    soft_tol = max(container_poly.length * 1e-9, 1e-4)
    grown = container_poly.buffer(soft_tol)
    return all(grown.covers(it['footprint']) for it in items)



def _place_living_room(msp, polygon, kitchen_geoms=None, opening_geoms=None, return_items=False):
    kitchen_geoms = kitchen_geoms or []
    opening_geoms = opening_geoms or []

    def _dbg(msg):
        print(f"[LivingLayout][Debug] {msg}")

    if polygon.is_empty or polygon.area <= 1e-6:
        print("[LivingLayout][Skip] invalid living geometry")
        _dbg(f"polygon_empty={polygon.is_empty} area={polygon.area if not polygon.is_empty else 'n/a'}")
        return [] if return_items else False

    frame = _living_local_frame(polygon)
    clearance = LIVING_MIN_CLEARANCE
    base_item_area = max(polygon.area * LIVING_ITEM_AREA_RATIO, 1e-6)

    # Base dimensions are used only for initial end-zone depth estimation.
    base_sofa_w, base_sofa_h = _dims_from_area(base_item_area, ratio_wh=2.80)
    base_tv_w, base_tv_h = _dims_from_area(base_item_area, ratio_wh=4.00)
    base_rect_tw, base_rect_th = _dims_from_area(base_item_area * 0.60, ratio_wh=1.60)
    base_round_r = math.sqrt((base_item_area * 0.60) / math.pi)

    span_u = frame['u_max'] - frame['u_min']
    span_v = frame['v_max'] - frame['v_min']
    split_axis = 'u' if span_u >= span_v else 'v'
    long_span = max(span_u, span_v)
    short_span = max(min(span_u, span_v), 1e-6)

    inset = max(clearance, short_span * LIVING_ZONE_INSET_RATIO)
    tv_depth = max(long_span * LIVING_TV_ZONE_RATIO, max(base_sofa_h, base_tv_h) + 4.0 * clearance)
    dining_depth = max(long_span * LIVING_DINING_ZONE_RATIO, max(base_rect_th, 2.0 * base_round_r) + 4.0 * clearance)
    x_gap = max(1.0, short_span * LIVING_X_GAP_RATIO)  # X

    _dbg(
        f"area={polygon.area:.3f} span_u={span_u:.3f} span_v={span_v:.3f} "
        f"split_axis={split_axis} base_item_area={base_item_area:.3f} inset={inset:.3f} "
        f"tv_depth={tv_depth:.3f} dining_depth={dining_depth:.3f} x_gap={x_gap:.3f}"
    )

    candidates = []

    for tv_side in (-1, 1):
        dining_side = -tv_side
        _dbg(f"trying tv_side={tv_side} dining_side={dining_side}")

        tv_zone = _end_zone_polygon(polygon, frame, split_axis, tv_side, tv_depth, inset)
        dining_zone = _end_zone_polygon(polygon, frame, split_axis, dining_side, dining_depth, inset)

        if tv_zone.is_empty or dining_zone.is_empty:
            _dbg(
                f"zone empty: tv_zone_empty={tv_zone.is_empty} dining_zone_empty={dining_zone.is_empty} "
                f"tv_zone_area={0.0 if tv_zone.is_empty else tv_zone.area:.3f} "
                f"dining_zone_area={0.0 if dining_zone.is_empty else dining_zone.area:.3f}"
            )
            continue

        _dbg(f"zone areas: tv={tv_zone.area:.3f} dining={dining_zone.area:.3f}")

        tv_bounds = _local_bounds_of_geom(frame, tv_zone)
        dining_bounds = _local_bounds_of_geom(frame, dining_zone)
        if tv_bounds is None or dining_bounds is None:
            _dbg(f"bounds unavailable: tv_bounds={tv_bounds} dining_bounds={dining_bounds}")
            continue

        tu_min, tv_min, tu_max, tv_max = tv_bounds
        du_min, dv_min, du_max, dv_max = dining_bounds
        _dbg(
            f"tv_bounds=(u:{tu_min:.3f}->{tu_max:.3f}, v:{tv_min:.3f}->{tv_max:.3f}) "
            f"dining_bounds=(u:{du_min:.3f}->{du_max:.3f}, v:{dv_min:.3f}->{dv_max:.3f})"
        )

        # Per-zone area caps: each group can only size from its own zone.
        tv_item_area = max(min(base_item_area, tv_zone.area * LIVING_ITEM_AREA_RATIO), 1e-6)
        dining_item_area = max(min(base_item_area, dining_zone.area * LIVING_ITEM_AREA_RATIO), 1e-6)

        sofa_w, sofa_h = _dims_from_area(tv_item_area, ratio_wh=2.80)
        coffee_w, coffee_h = _dims_from_area(tv_item_area, ratio_wh=1.35)
        rug_w, rug_h = _dims_from_area(tv_item_area, ratio_wh=1.80)
        tv_w, tv_h = _dims_from_area(tv_item_area, ratio_wh=4.00)

        _dbg(f"item_area_caps: tv={tv_item_area:.3f} dining={dining_item_area:.3f}")


        # ---------- TV CORNER ZONE ----------
        if split_axis == 'u':
            cross_v = (tv_min + tv_max) / 2.0

            tv_depth_u = tv_h
            tv_width_v = tv_w
            coffee_depth_u = coffee_h
            coffee_width_v = coffee_w
            sofa_depth_u = sofa_h
            sofa_width_v = sofa_w
            rug_depth_u = rug_h
            rug_width_v = rug_w

            if tv_side < 0:
                tv_u_target = tu_min + tv_depth_u / 2.0 + clearance
                coffee_u_target = tv_u_target + tv_depth_u / 2.0 + 3.0 * x_gap + coffee_depth_u / 2.0
                sofa_u_target = coffee_u_target + coffee_depth_u / 2.0 + x_gap + sofa_depth_u / 2.0
            else:
                tv_u_target = tu_max - tv_depth_u / 2.0 - clearance
                coffee_u_target = tv_u_target - tv_depth_u / 2.0 - 3.0 * x_gap - coffee_depth_u / 2.0
                sofa_u_target = coffee_u_target - coffee_depth_u / 2.0 - x_gap - sofa_depth_u / 2.0

            _dbg(
                f"tv_targets_u: tv={tv_u_target:.3f} coffee={coffee_u_target:.3f} sofa={sofa_u_target:.3f} "
                f"cross_v={cross_v:.3f}"
            )

            tv_fit = _fit_rect_inside(tv_zone, frame, tv_depth_u, tv_width_v, tv_u_target, cross_v)
            if tv_fit is None:
                _dbg("tv_fit failed in tv_zone")
                continue

            coffee_fit = _fit_rect_inside(
                tv_zone,
                frame,
                coffee_depth_u,
                coffee_width_v,
                coffee_u_target,
                cross_v,
                avoid_geoms=[tv_fit['footprint']],
                min_sep=0.05,
            )
            if coffee_fit is None:
                _dbg("coffee_fit failed in tv_zone")
                continue

            sofa_fit = _fit_rect_inside(
                tv_zone,
                frame,
                sofa_depth_u,
                sofa_width_v,
                sofa_u_target,
                cross_v,
                avoid_geoms=[tv_fit['footprint'], coffee_fit['footprint']],
                min_sep=0.05,
            )
            if sofa_fit is None:
                _dbg("sofa_fit failed in tv_zone")
                continue

            rug_u_target = (coffee_fit['u'] + sofa_fit['u']) / 2.0
            rug_fit = _fit_rect_inside(tv_zone, frame, rug_depth_u, rug_width_v, rug_u_target, cross_v)
            if rug_fit is None:
                _dbg("rug_fit failed in tv_zone")
                continue

        else:
            cross_u = (tu_min + tu_max) / 2.0

            tv_depth_v = tv_h
            tv_width_u = tv_w
            coffee_depth_v = coffee_h
            coffee_width_u = coffee_w
            sofa_depth_v = sofa_h
            sofa_width_u = sofa_w
            rug_depth_v = rug_h
            rug_width_u = rug_w

            if tv_side < 0:
                tv_v_target = tv_min + tv_depth_v / 2.0 + clearance
                coffee_v_target = tv_v_target + tv_depth_v / 2.0 + 3.0 * x_gap + coffee_depth_v / 2.0
                sofa_v_target = coffee_v_target + coffee_depth_v / 2.0 + x_gap + sofa_depth_v / 2.0
            else:
                tv_v_target = tv_max - tv_depth_v / 2.0 - clearance
                coffee_v_target = tv_v_target - tv_depth_v / 2.0 - 3.0 * x_gap - coffee_depth_v / 2.0
                sofa_v_target = coffee_v_target - coffee_depth_v / 2.0 - x_gap - sofa_depth_v / 2.0

            _dbg(
                f"tv_targets_v: tv={tv_v_target:.3f} coffee={coffee_v_target:.3f} sofa={sofa_v_target:.3f} "
                f"cross_u={cross_u:.3f}"
            )

            tv_fit = _fit_rect_inside(tv_zone, frame, tv_width_u, tv_depth_v, cross_u, tv_v_target)
            if tv_fit is None:
                _dbg("tv_fit failed in tv_zone")
                continue

            coffee_fit = _fit_rect_inside(
                tv_zone,
                frame,
                coffee_width_u,
                coffee_depth_v,
                cross_u,
                coffee_v_target,
                avoid_geoms=[tv_fit['footprint']],
                min_sep=0.05,
            )
            if coffee_fit is None:
                _dbg("coffee_fit failed in tv_zone")
                continue

            sofa_fit = _fit_rect_inside(
                tv_zone,
                frame,
                sofa_width_u,
                sofa_depth_v,
                cross_u,
                sofa_v_target,
                avoid_geoms=[tv_fit['footprint'], coffee_fit['footprint']],
                min_sep=0.05,
            )
            if sofa_fit is None:
                _dbg("sofa_fit failed in tv_zone")
                continue

            rug_v_target = (coffee_fit['v'] + sofa_fit['v']) / 2.0
            rug_fit = _fit_rect_inside(tv_zone, frame, rug_width_u, rug_depth_v, cross_u, rug_v_target)
            if rug_fit is None:
                _dbg("rug_fit failed in tv_zone")
                continue

        tv_items = [
            {'name': 'tv_unit', 'footprint': tv_fit['footprint']},
            {'name': 'coffee_table', 'footprint': coffee_fit['footprint']},
            {'name': 'sofa', 'footprint': sofa_fit['footprint']},
            {'name': 'rug', 'footprint': rug_fit['footprint']},
        ]
        if not _all_items_inside(tv_zone, tv_items):
            _dbg("tv_items not fully inside tv_zone")
            continue

        # ---------- DINING ZONE (opposite end) ----------
        d_center_u = (du_min + du_max) / 2.0
        d_center_v = (dv_min + dv_max) / 2.0
        dining_items = None

        for budget_scale in (1.0, 0.9, 0.8):
            budget = dining_item_area * budget_scale
            tw, th = _dims_from_area(budget * 0.60, ratio_wh=1.60)
            table_fit = _fit_rect_inside(dining_zone, frame, tw, th, d_center_u, d_center_v)
            if table_fit is None:
                _dbg(f"rect dining table_fit failed at budget_scale={budget_scale:.2f}")
                continue

            rect_items = _build_dining_rect(frame, table_fit['u'], table_fit['v'], budget, clearance)
            if _all_items_inside(dining_zone, rect_items):
                dining_items = rect_items
                _dbg(f"rect dining success at budget_scale={budget_scale:.2f}")
                break
            else:
                _dbg(f"rect dining items outside zone at budget_scale={budget_scale:.2f}")

        if dining_items is None:
            for budget_scale in (1.0, 0.9, 0.8):
                budget = dining_item_area * budget_scale
                rr = math.sqrt((budget * 0.60) / math.pi)
                circle_fit = _fit_circle_inside(dining_zone, frame, rr, d_center_u, d_center_v)
                if circle_fit is None:
                    _dbg(f"round dining circle_fit failed at budget_scale={budget_scale:.2f}")
                    continue

                round_items = _build_dining_round(frame, circle_fit['u'], circle_fit['v'], budget, clearance)
                if _all_items_inside(dining_zone, round_items):
                    dining_items = round_items
                    _dbg(f"round dining success at budget_scale={budget_scale:.2f}")
                    break
                else:
                    _dbg(f"round dining items outside zone at budget_scale={budget_scale:.2f}")

        if dining_items is None:
            _dbg("no dining configuration fit in dining_zone")
            continue

        candidate = tv_items + dining_items
        if _is_layout_valid(polygon, candidate, opening_geoms, clearance):
            candidates.append(candidate)
            _dbg(f"candidate accepted, total_candidates={len(candidates)}")
        else:
            _dbg("candidate rejected by _is_layout_valid (see detailed invalid reason above)")

    if not candidates:
        print("[LivingLayout][Skip] no valid layout candidate found after end-zone strategy")
        _dbg("all tv_side attempts failed")
        return [] if return_items else False

    scored = [
        (_score_layout(it, kitchen_geoms, LIVING_KITCHEN_SCORE_WEIGHT, frame['diag']), idx, it)
        for idx, it in enumerate(candidates)
    ]
    best_score, best_idx, best = max(scored, key=lambda x: x[0])
    _dbg(f"selected candidate index={best_idx} score={best_score:.4f} from {len(candidates)} candidates")

    if return_items:
        return best

    _draw_living_items(msp, best)
    return True


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


def generate_auto_furniture_items(playground):
    """
    Build phase-1 automatic furniture as editable item records for the GUI editor.
    Phase 1 seeds bedroom, bathroom, and kitchen furniture.
    Living-room furniture is intentionally not auto-seeded here.
    """
    def to_shapely(geom_data):
        if isinstance(geom_data, BaseGeometry):
            return geom_data
        if isinstance(geom_data, dict) and 'type' in geom_data:
            return shape(geom_data)
        return None

    def extract_polygons(geom):
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, MultiPolygon):
            return [p for p in geom.geoms if not p.is_empty]
        if isinstance(geom, Polygon):
            return [geom]
        return []

    def add_rect_item(icon, x0, y0, x1, y1, extra_fields=None):
        canonical = _canonical_icon_id(icon)
        out = {
            "icon": canonical,
            "display_label": "working_table" if canonical == "desk" else ("working_chair" if canonical == "chair" else canonical),
            "shape": "rect",
            "placement_mode": "auto",
            "x": float((x0 + x1) / 2.0),
            "y": float((y0 + y1) / 2.0),
            "w": float(x1 - x0),
            "h": float(y1 - y0),
            "rotation": 0.0,
        }
        if extra_fields:
            out.update(extra_fields)
        items.append(out)

    items = []

    # doors for room-aware heuristics
    door_geoms = extract_polygons(to_shapely(playground.get('door')))
    door_geoms += extract_polygons(to_shapely(playground.get('front_door')))

    # Bedrooms
    bedrooms = extract_polygons(to_shapely(playground.get('bedroom')))
    if bedrooms:
        master_idx = max(range(len(bedrooms)), key=lambda i: bedrooms[i].area)
        for i, poly in enumerate(bedrooms):
            is_master = (i == master_idx)
            nearest_door = min(door_geoms, key=lambda d: poly.distance(d)) if door_geoms else None
            door_pt = None
            if nearest_door is not None and not nearest_door.is_empty:
                c = nearest_door.centroid
                door_pt = (c.x, c.y)

            minx, miny, maxx, maxy = poly.bounds
            bed_layout = _compute_bed_box(minx, miny, maxx, maxy, is_master=is_master, door_pt=door_pt)
            if bed_layout is None:
                continue
            bx0, by0, bx1, by1 = bed_layout["bbox"]
            add_rect_item(
                "double_bed" if is_master else "single_bed",
                bx0,
                by0,
                bx1,
                by1,
                extra_fields={"head_side": bed_layout.get("head_side", "top")},
            )

            # Mirror bedroom extra-furniture heuristics from _place_bedroom.
            bed_bbox = bed_layout["bbox"]
            bed_head_side = bed_layout.get("head_side", "top")
            bed_wall = _bed_wall_from_bbox(minx, miny, maxx, maxy, bed_bbox)
            bx0, by0, bx1, by1 = bed_bbox

            ref_w, ref_h = _single_bed_reference_size(minx, miny, maxx, maxy)
            ref_bed_area = max(ref_w * ref_h, 1e-6)

            closet_area = ref_bed_area * 0.5
            dresser_area = ref_bed_area * 0.2
            desk_area = ref_bed_area * 0.4
            chair_area = ref_bed_area * 0.2

            room_poly = poly
            room_unit = min(maxx - minx, maxy - miny)
            edge_margin = max(room_unit * 0.05, 0.5)
            piece_gap = max(room_unit * 0.03, 0.35)
            door_clearance = max(room_unit * 0.12, 0.8)
            occupied = [_bbox_polygon(*bed_bbox)]

            closet_len, _ = _dims_from_area_ratio(closet_area, ratio_wh=4.0)
            wall_order = ["left", "right", "bottom", "top"]
            if bed_wall in wall_order:
                wall_order.remove(bed_wall)

            placed_closet = False
            for wall in wall_order:
                if wall in ("left", "right"):
                    usable = max((maxy - miny) - 2.0 * edge_margin, 1e-6)
                    length = min(closet_len, usable)
                    depth = max(closet_area / max(length, 1e-6), 1e-6)
                    for frac in (0.5, 0.25, 0.75):
                        cy = miny + edge_margin + usable * frac
                        y0 = cy - length / 2.0
                        y1 = cy + length / 2.0
                        if wall == "left":
                            x0 = minx + edge_margin
                            x1 = x0 + depth
                        else:
                            x1 = maxx - edge_margin
                            x0 = x1 - depth
                        fp = _bbox_polygon(x0, y0, x1, y1)
                        if _is_valid_piece(room_poly, fp, occupied, nearest_door, door_clearance):
                            add_rect_item("closet", x0, y0, x1, y1)
                            occupied.append(fp)
                            placed_closet = True
                            break
                else:
                    usable = max((maxx - minx) - 2.0 * edge_margin, 1e-6)
                    length = min(closet_len, usable)
                    depth = max(closet_area / max(length, 1e-6), 1e-6)
                    for frac in (0.5, 0.25, 0.75):
                        cx = minx + edge_margin + usable * frac
                        x0 = cx - length / 2.0
                        x1 = cx + length / 2.0
                        if wall == "bottom":
                            y0 = miny + edge_margin
                            y1 = y0 + depth
                        else:
                            y1 = maxy - edge_margin
                            y0 = y1 - depth
                        fp = _bbox_polygon(x0, y0, x1, y1)
                        if _is_valid_piece(room_poly, fp, occupied, nearest_door, door_clearance):
                            add_rect_item("closet", x0, y0, x1, y1)
                            occupied.append(fp)
                            placed_closet = True
                            break
                if placed_closet:
                    break

            dr_w, dr_h = _dims_from_area_ratio(dresser_area, ratio_wh=1.4)
            dr_w *= 0.5
            dr_h *= 0.5

            if bed_head_side == "top":
                dresser_candidates = [
                    (bx0 - piece_gap - dr_w, by1 - dr_h, bx0 - piece_gap, by1),
                    (bx1 + piece_gap, by1 - dr_h, bx1 + piece_gap + dr_w, by1),
                ]
            else:
                dresser_candidates = [
                    (bx0 - piece_gap - dr_w, by0, bx0 - piece_gap, by0 + dr_h),
                    (bx1 + piece_gap, by0, bx1 + piece_gap + dr_w, by0 + dr_h),
                ]

            for x0, y0, x1, y1 in dresser_candidates:
                fp = _bbox_polygon(x0, y0, x1, y1)
                if _is_valid_piece(room_poly, fp, occupied, nearest_door, door_clearance):
                    add_rect_item("dresser", x0, y0, x1, y1)
                    occupied.append(fp)
                    break

            desk_w, desk_h = _dims_from_area_ratio(desk_area, ratio_wh=1.8)
            chair_s = max(chair_area, 1e-6) ** 0.5
            chair_s *= 0.5
            office_gap = piece_gap

            corner_variants = []
            corner_variants.append(((minx + edge_margin, miny + edge_margin, minx + edge_margin + desk_w, miny + edge_margin + desk_h),
                                    (minx + edge_margin + desk_w * 0.5 - chair_s / 2.0, miny + edge_margin + desk_h + office_gap,
                                     minx + edge_margin + desk_w * 0.5 + chair_s / 2.0, miny + edge_margin + desk_h + office_gap + chair_s)))
            corner_variants.append(((minx + edge_margin, miny + edge_margin, minx + edge_margin + desk_h, miny + edge_margin + desk_w),
                                    (minx + edge_margin + desk_h + office_gap, miny + edge_margin + desk_w * 0.5 - chair_s / 2.0,
                                     minx + edge_margin + desk_h + office_gap + chair_s, miny + edge_margin + desk_w * 0.5 + chair_s / 2.0)))
            corner_variants.append(((maxx - edge_margin - desk_w, miny + edge_margin, maxx - edge_margin, miny + edge_margin + desk_h),
                                    (maxx - edge_margin - desk_w * 0.5 - chair_s / 2.0, miny + edge_margin + desk_h + office_gap,
                                     maxx - edge_margin - desk_w * 0.5 + chair_s / 2.0, miny + edge_margin + desk_h + office_gap + chair_s)))
            corner_variants.append(((maxx - edge_margin - desk_h, miny + edge_margin, maxx - edge_margin, miny + edge_margin + desk_w),
                                    (maxx - edge_margin - desk_h - office_gap - chair_s, miny + edge_margin + desk_w * 0.5 - chair_s / 2.0,
                                     maxx - edge_margin - desk_h - office_gap, miny + edge_margin + desk_w * 0.5 + chair_s / 2.0)))
            corner_variants.append(((minx + edge_margin, maxy - edge_margin - desk_h, minx + edge_margin + desk_w, maxy - edge_margin),
                                    (minx + edge_margin + desk_w * 0.5 - chair_s / 2.0, maxy - edge_margin - desk_h - office_gap - chair_s,
                                     minx + edge_margin + desk_w * 0.5 + chair_s / 2.0, maxy - edge_margin - desk_h - office_gap)))
            corner_variants.append(((minx + edge_margin, maxy - edge_margin - desk_w, minx + edge_margin + desk_h, maxy - edge_margin),
                                    (minx + edge_margin + desk_h + office_gap, maxy - edge_margin - desk_w * 0.5 - chair_s / 2.0,
                                     minx + edge_margin + desk_h + office_gap + chair_s, maxy - edge_margin - desk_w * 0.5 + chair_s / 2.0)))
            corner_variants.append(((maxx - edge_margin - desk_w, maxy - edge_margin - desk_h, maxx - edge_margin, maxy - edge_margin),
                                    (maxx - edge_margin - desk_w * 0.5 - chair_s / 2.0, maxy - edge_margin - desk_h - office_gap - chair_s,
                                     maxx - edge_margin - desk_w * 0.5 + chair_s / 2.0, maxy - edge_margin - desk_h - office_gap)))
            corner_variants.append(((maxx - edge_margin - desk_h, maxy - edge_margin - desk_w, maxx - edge_margin, maxy - edge_margin),
                                    (maxx - edge_margin - desk_h - office_gap - chair_s, maxy - edge_margin - desk_w * 0.5 - chair_s / 2.0,
                                     maxx - edge_margin - desk_h - office_gap, maxy - edge_margin - desk_w * 0.5 + chair_s / 2.0)))

            for desk_bbox, chair_bbox in corner_variants:
                d_fp = _bbox_polygon(*desk_bbox)
                c_fp = _bbox_polygon(*chair_bbox)
                if not _is_valid_piece(room_poly, d_fp, occupied, nearest_door, door_clearance):
                    continue
                if not _is_valid_piece(room_poly, c_fp, occupied + [d_fp], nearest_door, door_clearance):
                    continue
                add_rect_item("desk", *desk_bbox)
                add_rect_item("chair", *chair_bbox)
                occupied.append(d_fp)
                occupied.append(c_fp)
                break

    # Bathrooms
    bathrooms = extract_polygons(to_shapely(playground.get('bathroom')))
    for poly in bathrooms:
        minx, miny, maxx, maxy = poly.bounds
        rw = maxx - minx
        rh = maxy - miny
        unit = min(rw, rh)
        m = unit * 0.05
        room_area = max(rw * rh, 1e-6)

        shw_area = room_area * 0.20
        toi_area = room_area * 0.10
        snk_area = room_area * 0.05
        toi_ratio = 0.38 / 0.55
        snk_ratio = 0.34 / 0.32

        toi_w = (toi_area * toi_ratio) ** 0.5
        toi_h = toi_area / toi_w
        snk_w = (snk_area * snk_ratio) ** 0.5
        snk_h = snk_area / snk_w
        shw_s = shw_area ** 0.5

        if rw >= rh:
            tox, toy = minx + m, miny + m
            six, siy = minx + m + toi_w + m, miny + m
            shx, shy = maxx - m - shw_s, maxy - m - shw_s
        else:
            tox, toy = minx + m, miny + m
            six, siy = minx + m, miny + m + toi_h + m
            shx, shy = maxx - m - shw_s, maxy - m - shw_s

        items.append({"icon": "toilet", "shape": "rect", "placement_mode": "auto", "x": tox + toi_w / 2.0, "y": toy + toi_h / 2.0, "w": toi_w, "h": toi_h, "rotation": 0.0})
        items.append({"icon": "sink", "shape": "rect", "placement_mode": "auto", "x": six + snk_w / 2.0, "y": siy + snk_h / 2.0, "w": snk_w, "h": snk_h, "rotation": 0.0})
        items.append({"icon": "shower", "shape": "rect", "placement_mode": "auto", "x": shx + shw_s / 2.0, "y": shy + shw_s / 2.0, "w": shw_s, "h": shw_s, "rotation": 0.0})

    # Kitchens
    kitchens = extract_polygons(to_shapely(playground.get('kitchen')))
    for poly in kitchens:
        minx, miny, maxx, maxy = poly.bounds
        rw = maxx - minx
        rh = maxy - miny
        unit = min(rw, rh)
        m = unit * 0.05
        frg_w = unit * 0.40
        frg_h = unit * 0.55
        snk_w = unit * 0.58
        snk_h = unit * 0.32

        if rw >= rh:
            fx, fy = minx + m, miny + m
            sx, sy = maxx - m - snk_w, miny + m
        else:
            fx, fy = minx + m, miny + m
            sx, sy = minx + m, miny + m + frg_h + m

        items.append({"icon": "fridge", "shape": "rect", "placement_mode": "auto", "x": fx + frg_w / 2.0, "y": fy + frg_h / 2.0, "w": frg_w, "h": frg_h, "rotation": 0.0})
        items.append({"icon": "kitchen_sink", "shape": "rect", "placement_mode": "auto", "x": sx + snk_w / 2.0, "y": sy + snk_h / 2.0, "w": snk_w, "h": snk_h, "rotation": 0.0})


    return items


# ═══════════════════════════════════════════════════════════════════
#  ROOM-LEVEL PLACEMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _place_bedroom(msp, polygon, is_master, door_geom=None):
    minx, miny, maxx, maxy = polygon.bounds

    door_pt = None
    if door_geom is not None and not door_geom.is_empty:
        c = door_geom.centroid
        door_pt = (c.x, c.y)

    if is_master:
        _draw_double_bed(msp, minx, miny, maxx, maxy, door_pt=door_pt)
    else:
        _draw_single_bed(msp, minx, miny, maxx, maxy, door_pt=door_pt)

    # ---- Extra bedroom furniture ----
    bed_layout = _compute_bed_box(minx, miny, maxx, maxy, is_master=is_master, door_pt=door_pt)
    if bed_layout is None:
        return

    bed_bbox = bed_layout["bbox"]
    bed_head_side = bed_layout.get("head_side", "top")
    bed_wall = _bed_wall_from_bbox(minx, miny, maxx, maxy, bed_bbox)
    bx0, by0, bx1, by1 = bed_bbox
    bw = bx1 - bx0
    bh = by1 - by0

    ref_w, ref_h = _single_bed_reference_size(minx, miny, maxx, maxy)
    ref_bed_area = max(ref_w * ref_h, 1e-6)

    closet_area = ref_bed_area * 0.5
    dresser_area = ref_bed_area * 0.2
    desk_area = ref_bed_area * 0.4
    chair_area = ref_bed_area * 0.2

    room_poly = polygon
    room_unit = min(maxx - minx, maxy - miny)
    edge_margin = max(room_unit * 0.05, 0.5)
    piece_gap = max(room_unit * 0.03, 0.35)
    door_clearance = max(room_unit * 0.12, 0.8)

    occupied = [_bbox_polygon(*bed_bbox)]

    # 1) Closet: place on a wall different from bed wall and away from door.
    closet_len, closet_dep = _dims_from_area_ratio(closet_area, ratio_wh=4.0)
    wall_order = ["left", "right", "bottom", "top"]
    if bed_wall in wall_order:
        wall_order.remove(bed_wall)

    placed_closet = False
    for wall in wall_order:
        if wall in ("left", "right"):
            usable = max((maxy - miny) - 2.0 * edge_margin, 1e-6)
            length = min(closet_len, usable)
            depth = max(closet_area / max(length, 1e-6), 1e-6)
            for frac in (0.5, 0.25, 0.75):
                cy = miny + edge_margin + usable * frac
                y0 = cy - length / 2.0
                y1 = cy + length / 2.0
                if wall == "left":
                    x0 = minx + edge_margin
                    x1 = x0 + depth
                else:
                    x1 = maxx - edge_margin
                    x0 = x1 - depth
                fp = _bbox_polygon(x0, y0, x1, y1)
                if _is_valid_piece(room_poly, fp, occupied, door_geom, door_clearance):
                    _draw_piece_by_name(msp, "closet", (x0, y0, x1, y1))
                    occupied.append(fp)
                    placed_closet = True
                    break
        else:
            usable = max((maxx - minx) - 2.0 * edge_margin, 1e-6)
            length = min(closet_len, usable)
            depth = max(closet_area / max(length, 1e-6), 1e-6)
            for frac in (0.5, 0.25, 0.75):
                cx = minx + edge_margin + usable * frac
                x0 = cx - length / 2.0
                x1 = cx + length / 2.0
                if wall == "bottom":
                    y0 = miny + edge_margin
                    y1 = y0 + depth
                else:
                    y1 = maxy - edge_margin
                    y0 = y1 - depth
                fp = _bbox_polygon(x0, y0, x1, y1)
                if _is_valid_piece(room_poly, fp, occupied, door_geom, door_clearance):
                    _draw_piece_by_name(msp, "closet", (x0, y0, x1, y1))
                    occupied.append(fp)
                    placed_closet = True
                    break

        if placed_closet:
            break

    # 2) Dresser: only on the two bedside positions near the bed head; else skip.
    dr_w, dr_h = _dims_from_area_ratio(dresser_area, ratio_wh=1.4)
    # reduce dresser by 50% (linear dimensions)
    dr_w *= 0.5
    dr_h *= 0.5

    if bed_head_side == "top":
        dresser_candidates = [
            (bx0 - piece_gap - dr_w, by1 - dr_h, bx0 - piece_gap, by1),
            (bx1 + piece_gap, by1 - dr_h, bx1 + piece_gap + dr_w, by1),
        ]
    else:
        dresser_candidates = [
            (bx0 - piece_gap - dr_w, by0, bx0 - piece_gap, by0 + dr_h),
            (bx1 + piece_gap, by0, bx1 + piece_gap + dr_w, by0 + dr_h),
        ]

    for x0, y0, x1, y1 in dresser_candidates:
        fp = _bbox_polygon(x0, y0, x1, y1)
        if _is_valid_piece(room_poly, fp, occupied, door_geom, door_clearance):
            _draw_piece_by_name(msp, "dresser", (x0, y0, x1, y1))
            occupied.append(fp)
            break

    # 3) Corner office (desk + chair): fallback to none if no clear corner.
    desk_w, desk_h = _dims_from_area_ratio(desk_area, ratio_wh=1.8)
    chair_s = max(chair_area, 1e-6) ** 0.5
    # reduce working chair by 75% (to 1/4 of current linear size)
    chair_s *= 0.5
    office_gap = piece_gap

    corner_variants = []
    # bl
    corner_variants.append(((minx + edge_margin, miny + edge_margin, minx + edge_margin + desk_w, miny + edge_margin + desk_h),
                            (minx + edge_margin + desk_w * 0.5 - chair_s / 2.0, miny + edge_margin + desk_h + office_gap,
                             minx + edge_margin + desk_w * 0.5 + chair_s / 2.0, miny + edge_margin + desk_h + office_gap + chair_s)))
    corner_variants.append(((minx + edge_margin, miny + edge_margin, minx + edge_margin + desk_h, miny + edge_margin + desk_w),
                            (minx + edge_margin + desk_h + office_gap, miny + edge_margin + desk_w * 0.5 - chair_s / 2.0,
                             minx + edge_margin + desk_h + office_gap + chair_s, miny + edge_margin + desk_w * 0.5 + chair_s / 2.0)))
    # br
    corner_variants.append(((maxx - edge_margin - desk_w, miny + edge_margin, maxx - edge_margin, miny + edge_margin + desk_h),
                            (maxx - edge_margin - desk_w * 0.5 - chair_s / 2.0, miny + edge_margin + desk_h + office_gap,
                             maxx - edge_margin - desk_w * 0.5 + chair_s / 2.0, miny + edge_margin + desk_h + office_gap + chair_s)))
    corner_variants.append(((maxx - edge_margin - desk_h, miny + edge_margin, maxx - edge_margin, miny + edge_margin + desk_w),
                            (maxx - edge_margin - desk_h - office_gap - chair_s, miny + edge_margin + desk_w * 0.5 - chair_s / 2.0,
                             maxx - edge_margin - desk_h - office_gap, miny + edge_margin + desk_w * 0.5 + chair_s / 2.0)))
    # tl
    corner_variants.append(((minx + edge_margin, maxy - edge_margin - desk_h, minx + edge_margin + desk_w, maxy - edge_margin),
                            (minx + edge_margin + desk_w * 0.5 - chair_s / 2.0, maxy - edge_margin - desk_h - office_gap - chair_s,
                             minx + edge_margin + desk_w * 0.5 + chair_s / 2.0, maxy - edge_margin - desk_h - office_gap)))
    corner_variants.append(((minx + edge_margin, maxy - edge_margin - desk_w, minx + edge_margin + desk_h, maxy - edge_margin),
                            (minx + edge_margin + desk_h + office_gap, maxy - edge_margin - desk_w * 0.5 - chair_s / 2.0,
                             minx + edge_margin + desk_h + office_gap + chair_s, maxy - edge_margin - desk_w * 0.5 + chair_s / 2.0)))
    # tr
    corner_variants.append(((maxx - edge_margin - desk_w, maxy - edge_margin - desk_h, maxx - edge_margin, maxy - edge_margin),
                            (maxx - edge_margin - desk_w * 0.5 - chair_s / 2.0, maxy - edge_margin - desk_h - office_gap - chair_s,
                             maxx - edge_margin - desk_w * 0.5 + chair_s / 2.0, maxy - edge_margin - desk_h - office_gap)))
    corner_variants.append(((maxx - edge_margin - desk_h, maxy - edge_margin - desk_w, maxx - edge_margin, maxy - edge_margin),
                            (maxx - edge_margin - desk_h - office_gap - chair_s, maxy - edge_margin - desk_w * 0.5 - chair_s / 2.0,
                             maxx - edge_margin - desk_h - office_gap, maxy - edge_margin - desk_w * 0.5 + chair_s / 2.0)))

    for desk_bbox, chair_bbox in corner_variants:
        d_fp = _bbox_polygon(*desk_bbox)
        c_fp = _bbox_polygon(*chair_bbox)
        if not _is_valid_piece(room_poly, d_fp, occupied, door_geom, door_clearance):
            continue
        if not _is_valid_piece(room_poly, c_fp, occupied + [d_fp], door_geom, door_clearance):
            continue
        _draw_piece_by_name(msp, "desk", desk_bbox)
        _draw_piece_by_name(msp, "chair", chair_bbox)
        occupied.append(d_fp)
        occupied.append(c_fp)
        break


def _place_bathroom(msp, polygon):
    """
    Toilet (bottom-left) + sink (beside / above toilet) + shower (opposite corner).
    Layout flips depending on room orientation.
    """
    minx, miny, maxx, maxy = polygon.bounds
    rw   = maxx - minx
    rh   = maxy - miny
    unit = min(rw, rh)
    m    = unit * 0.05

    room_area = max(rw * rh, 1e-6)

    # Target footprint by bathroom area
    shw_area = room_area * 0.20
    toi_area = room_area * 0.10
    snk_area = room_area * 0.05

    # Keep existing shape proportions for toilet/sink
    toi_ratio = 0.38 / 0.55  # width / height
    snk_ratio = 0.34 / 0.32  # width / height

    toi_w = (toi_area * toi_ratio) ** 0.5
    toi_h = toi_area / toi_w

    snk_w = (snk_area * snk_ratio) ** 0.5
    snk_h = snk_area / snk_w

    # Shower remains square
    shw_s = shw_area ** 0.5

    if rw >= rh:
        # Wide bathroom – fixtures along bottom, shower top-right
        _draw_toilet(msp,  minx + m,                   miny + m, toi_w, toi_h)
        _draw_sink(msp,    minx + m + toi_w + m,        miny + m, snk_w, snk_h)
        _draw_shower(msp,  maxx - m - shw_s,             maxy - m - shw_s, shw_s)
    else:
        # Portrait bathroom – stack vertically, shower top-right
        _draw_toilet(msp,  minx + m,  miny + m,                      toi_w, toi_h)
        _draw_sink(msp,    minx + m,  miny + m + toi_h + m,          snk_w, snk_h)
        _draw_shower(msp,  maxx - m - shw_s, maxy - m - shw_s,       shw_s)


def _place_kitchen(msp, polygon):
    """
    Fridge (one corner) + double kitchen sink (opposite end).
    Layout adapts to room orientation.
    """
    minx, miny, maxx, maxy = polygon.bounds
    rw   = maxx - minx
    rh   = maxy - miny
    unit = min(rw, rh)
    m    = unit * 0.05

    frg_w = unit * 0.40
    frg_h = unit * 0.55
    snk_w = unit * 0.58
    snk_h = unit * 0.32

    if rw >= rh:
        # Wide kitchen – fridge left, sink right
        _draw_fridge(msp,        minx + m,              miny + m, frg_w, frg_h)
        _draw_kitchen_sink(msp,  maxx - m - snk_w,       miny + m, snk_w, snk_h)
    else:
        # Narrow kitchen – fridge bottom, sink above
        _draw_fridge(msp,        minx + m,  miny + m,              frg_w, frg_h)
        _draw_kitchen_sink(msp,  minx + m,  miny + m + frg_h + m,  snk_w, snk_h)


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