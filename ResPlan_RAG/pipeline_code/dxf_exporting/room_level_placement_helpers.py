from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
import math

from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporting_drawing_functions import _draw_icon_from_footprint


# ═══════════════════════════════════════════════════════════════════
#  ROOM-LEVEL PLACEMENT FUNCTIONS HELPERS
# ═══════════════════════════════════════════════════════════════════


def _local_rect_polygon(frame, u, v, w, h):
    hu, hv = w / 2.0, h / 2.0
    corners_local = [
        (u - hu, v - hv),
        (u + hu, v - hv),
        (u + hu, v + hv),
        (u - hu, v + hv),
    ]
    return Polygon([_local_to_world(frame, cu, cv) for cu, cv in corners_local])


def _draw_living_items(msp, items):
    for item in items:
        name = item['name']
        fp = item['footprint']
        _draw_icon_from_footprint(msp, name, fp)


def _all_items_inside(container_poly, items):
    if container_poly.is_empty:
        return False
    soft_tol = max(container_poly.length * 1e-9, 1e-4)
    grown = container_poly.buffer(soft_tol)
    return all(grown.covers(it['footprint']) for it in items)


def _local_to_world(frame, u, v):
    x = frame['cx'] + u * frame['ux'] + v * frame['vx']
    y = frame['cy'] + u * frame['uy'] + v * frame['vy']
    return x, y


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


def _project_to_local(frame, x, y):
    px, py = x - frame['cx'], y - frame['cy']
    u = px * frame['ux'] + py * frame['uy']
    v = px * frame['vx'] + py * frame['vy']
    return u, v


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

def _dims_from_area(target_area, ratio_wh):
    w = math.sqrt(max(target_area, 1e-9) * max(ratio_wh, 1e-9))
    h = max(target_area, 1e-9) / w
    return w, h


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