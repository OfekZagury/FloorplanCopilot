# ─────────────────────────────────────────────
#  TUNEABLE CONSTANTS
# ─────────────────────────────────────────────
import math

from shapely.geometry.polygon import Polygon

from ResPlan_RAG.pipeline_code import config

logger = config.logger

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
        logger.error("Ellipse radii must be > 0. Got rx=%s ry=%s", rx, ry)
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



def _canonical_icon_id(icon_id):
    if not icon_id:
        return icon_id
    if str(icon_id).startswith("dining_chair"):
        return "dining_chair"
    return ICON_ALIAS_TO_CANONICAL.get(icon_id, icon_id)



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
        logger.error("Unsupported furniture icon for DXF drawing: %s", icon)
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