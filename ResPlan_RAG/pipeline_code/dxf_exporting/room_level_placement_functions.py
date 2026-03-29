import math

from ResPlan_RAG.pipeline_code import config

from ResPlan_RAG.pipeline_code.dxf_exporting.room_level_placement_helpers import  _draw_living_items, _all_items_inside

from ResPlan_RAG.pipeline_code.dxf_exporting.dxf_exporting_drawing_functions import _compute_bed_box, \
    _bed_wall_from_bbox, _single_bed_reference_size, _bbox_polygon, _dims_from_area_ratio, _is_valid_piece, \
    _draw_double_bed, _draw_single_bed, _draw_piece_by_name, _draw_fridge, _draw_kitchen_sink, \
    _draw_toilet, _draw_sink, _draw_shower, LIVING_KITCHEN_SCORE_WEIGHT, LIVING_ITEM_AREA_RATIO, \
    LIVING_ZONE_INSET_RATIO, LIVING_TV_ZONE_RATIO, LIVING_DINING_ZONE_RATIO, LIVING_X_GAP_RATIO, LIVING_MIN_CLEARANCE
from ResPlan_RAG.pipeline_code.dxf_exporting.room_level_placement_helpers import _living_local_frame, _dims_from_area, \
    _local_bounds_of_geom, _end_zone_polygon, _fit_rect_inside, _fit_circle_inside, _build_dining_round, \
    _build_dining_rect, _score_layout, _is_layout_valid

logger = config.logger


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



def _place_living_room(msp, polygon, kitchen_geoms=None, opening_geoms=None, return_items=False):
    kitchen_geoms = kitchen_geoms or []
    opening_geoms = opening_geoms or []

    def _dbg(msg):
        logger.debug("[LivingLayout][Debug] %s", msg)

    if polygon.is_empty or polygon.area <= 1e-6:
        logger.warning("[LivingLayout][Skip] invalid living geometry")
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
        logger.warning("[LivingLayout][Skip] no valid layout candidate found after end-zone strategy")
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

