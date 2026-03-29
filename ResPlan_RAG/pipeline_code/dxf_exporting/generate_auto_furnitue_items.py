from shapely.geometry.base import BaseGeometry
from shapely.geometry.geo import shape
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporting_drawing_functions import _bbox_polygon, _is_valid_piece, \
    _dims_from_area_ratio, _single_bed_reference_size, _bed_wall_from_bbox, _compute_bed_box, _canonical_icon_id


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