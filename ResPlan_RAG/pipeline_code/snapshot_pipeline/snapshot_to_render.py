import math
import os
import importlib
from datetime import datetime
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

from ResPlan_RAG.pipeline_code import config


_PIPE_CACHE = {}


def apply_isometric_warp(image_path):
    """
    Rotates and squashes the image while expanding the canvas to prevent clipping.
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # Calculate the new canvas dimensions needed for a 45-degree rotation
    angle = 45
    theta = math.radians(angle)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))

    new_w = int(w * cos_t + h * sin_t)
    new_h = int(w * sin_t + h * cos_t)

    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Move the image to the center of the new canvas (so it won't be clipped)
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]

    # Rotate with the expanded canvas
    rotated = cv2.warpAffine(img, M_rot, (new_w, new_h), borderValue=(0, 0, 0))

    # Isometric squash to create 3D depth
    warped = cv2.resize(rotated, (new_w, int(new_h * 0.577)))

    cv2.imwrite(image_path, warped)


def _safe_apartment_label(apartment_id):
    if apartment_id is None:
        return "unknown"
    return str(apartment_id).replace(" ", "_")


def _build_artifact_paths(apartment_id):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    apt_label = _safe_apartment_label(apartment_id)

    base_name = f"apt_{apt_label}_{ts}"
    run_dir = os.path.join(config.SNAPSHOT_OUTPUT_DIR, base_name)
    os.makedirs(run_dir, exist_ok=True)

    return {
        "raw_snapshot": os.path.join(run_dir, f"{base_name}_raw.png"),
        "segmentation_map": os.path.join(run_dir, f"{base_name}_seg.png"),
        "canny_map": os.path.join(run_dir, f"{base_name}_canny.png"),
        "rendered_image": os.path.join(run_dir, f"{base_name}_render.png"),
    }


def capture_scene_snapshot(scene, output_path):
    """Render the full Qt scene into a PNG file."""
    try:
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QImage, QPainter
    except Exception as exc:
        raise RuntimeError(f"PySide6 is required for scene snapshot capture: {exc}")

    rect = scene.sceneRect()
    if rect.isNull() or rect.width() <= 1 or rect.height() <= 1:
        rect = scene.itemsBoundingRect()

    width = max(1, int(round(rect.width())))
    height = max(1, int(round(rect.height())))

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.black)

    painter = QPainter(image)
    scene.render(painter, target=QRectF(0.0, 0.0, float(width), float(height)), source=rect)
    painter.end()

    ok = image.save(output_path)
    if not ok:
        raise RuntimeError(f"Failed to save scene snapshot to {output_path}")
    return output_path


def get_hsv_mask(image_hsv, color_name):
    """Create a binary mask for a specific color family in HSV space."""
    if color_name == "cyan":
        lower = np.array([85, 100, 100])
        upper = np.array([105, 255, 255])
    elif color_name == "green":
        lower = np.array([40, 40, 40])
        upper = np.array([80, 255, 255])
    else:
        return None
    return cv2.inRange(image_hsv, lower, upper)


def _to_shapely(geom_data):
    if isinstance(geom_data, BaseGeometry):
        return geom_data
    if isinstance(geom_data, dict) and "type" in geom_data and "coordinates" in geom_data:
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


def _legacy_hsv_seg_map(input_path, output_path):
    """Legacy fallback that infers coarse seg classes from editor colors."""
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Could not load input image: {input_path}")

    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cyan_mask = get_hsv_mask(hsv, "cyan")
    green_mask = get_hsv_mask(hsv, "green")

    seg_map = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(seg_map)

    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    furniture_color = (0, 192, 192)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15:
            draw.rectangle([x, y, x + w, y + h], fill=furniture_color)

    wall_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_color = (255, 255, 255)
    for cnt in wall_contours:
        if cv2.contourArea(cnt) > 10:
            points = [tuple(p[0]) for p in cnt]
            if len(points) > 1:
                draw.line(points, fill=wall_color, width=5)

    seg_map.save(output_path)
    return output_path


def _geometry_room_seg_map(playground, output_path, width, height):
    """Rasterize room polygons into a model-friendly segmentation control map."""
    if not isinstance(playground, dict):
        raise RuntimeError("playground is not a dictionary")

    room_keys = list(config.HF_SEG_ROOM_PRIORITY)
    room_polygons = {
        key: _extract_polygons(_to_shapely(playground.get(key)))
        for key in room_keys
    }
    wall_polygons = _extract_polygons(_to_shapely(playground.get("wall")))

    all_polygons = []
    for key in room_keys:
        all_polygons.extend(room_polygons.get(key, []))
    all_polygons.extend(wall_polygons)
    if not all_polygons:
        raise RuntimeError("No room/wall polygons available for geometry segmentation")

    minx = min(p.bounds[0] for p in all_polygons)
    miny = min(p.bounds[1] for p in all_polygons)
    maxx = max(p.bounds[2] for p in all_polygons)
    maxy = max(p.bounds[3] for p in all_polygons)
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)

    margin = float(config.HF_SEG_CANVAS_MARGIN_PX)
    draw_w = max(1.0, float(width) - 2.0 * margin)
    draw_h = max(1.0, float(height) - 2.0 * margin)
    scale = min(draw_w / span_x, draw_h / span_y)

    seg = np.zeros((height, width, 3), dtype=np.uint8)

    def world_to_px(x, y):
        px = int(round(margin + (x - minx) * scale))
        py = int(round(margin + (maxy - y) * scale))
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        return px, py

    def draw_poly_fill(poly, rgb_color):
        ext = np.array([world_to_px(x, y) for x, y in poly.exterior.coords], dtype=np.int32)
        if len(ext) >= 3:
            cv2.fillPoly(seg, [ext], color=tuple(int(v) for v in rgb_color))
        for interior in poly.interiors:
            hole = np.array([world_to_px(x, y) for x, y in interior.coords], dtype=np.int32)
            if len(hole) >= 3:
                cv2.fillPoly(seg, [hole], color=(0, 0, 0))

    for room_key in room_keys:
        rgb = config.HF_SEG_ROOM_COLORS.get(room_key)
        if rgb is None:
            continue
        for poly in room_polygons.get(room_key, []):
            draw_poly_fill(poly, rgb)

    wall_rgb = config.HF_SEG_ROOM_COLORS.get("wall")
    wall_width = int(config.HF_SEG_WALL_LINE_WIDTH)
    if wall_rgb is not None and wall_width > 0:
        for poly in wall_polygons:
            ext = np.array([world_to_px(x, y) for x, y in poly.exterior.coords], dtype=np.int32)
            if len(ext) >= 2:
                cv2.polylines(seg, [ext], isClosed=True, color=tuple(int(v) for v in wall_rgb), thickness=wall_width)

    Image.fromarray(seg).save(output_path)
    return output_path


def process_cad_to_seg_map(input_path, output_path, playground=None):
    """
    Build segmentation control map.
    Uses geometry-driven room labeling when playground is provided, with HSV fallback.
    """
    if config.HF_SEG_USE_PLAYGROUND_GEOMETRY and playground is not None:
        img = cv2.imread(input_path)
        if img is not None:
            h, w = img.shape[:2]
            try:
                return _geometry_room_seg_map(playground, output_path, width=w, height=h)
            except Exception:
                if not config.HF_SEG_ALLOW_FALLBACK_TO_HSV:
                    raise

    return _legacy_hsv_seg_map(input_path, output_path)


def create_canny_control_map(input_path, output_path):
    """Create a 3-channel canny edge image for ControlNet conditioning."""
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Could not load input image for canny map: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        threshold1=int(config.HF_CANNY_LOW_THRESHOLD),
        threshold2=int(config.HF_CANNY_HIGH_THRESHOLD),
    )
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    Image.fromarray(edges_rgb).save(output_path)
    return output_path


def _torch_dtype_from_name(dtype_name, torch_module):
    if str(dtype_name).lower() == "float16":
        return torch_module.float16
    return torch_module.float32


def _get_or_create_pipeline(controlnet_model_ids):
    model_ids = tuple(controlnet_model_ids)
    cache_key = (
        model_ids,
        str(config.HF_BASE_MODEL_ID),
        str(config.HF_DEVICE),
        str(config.HF_TORCH_DTYPE),
    )
    if cache_key in _PIPE_CACHE:
        return _PIPE_CACHE[cache_key]

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for rendering. Details: {exc}")

    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        MultiControlNetModel = None
        if len(model_ids) > 1:
            from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
    except Exception as exc:
        raise RuntimeError(
            "Diffusion dependencies are missing. Install torch/diffusers/transformers/accelerate/safetensors. "
            f"Details: {exc}"
        )

    torch_dtype = _torch_dtype_from_name(config.HF_TORCH_DTYPE, torch)

    if len(model_ids) == 1:
        controlnet = ControlNetModel.from_pretrained(
            model_ids[0],
            torch_dtype=torch_dtype,
        )
    else:
        if MultiControlNetModel is None:
            raise RuntimeError("MultiControlNetModel is unavailable in the installed diffusers package")
        controlnets = [
            ControlNetModel.from_pretrained(mid, torch_dtype=torch_dtype)
            for mid in model_ids
        ]
        controlnet = MultiControlNetModel(controlnets)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.HF_BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.HF_DEVICE)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    _PIPE_CACHE[cache_key] = pipe
    return _PIPE_CACHE[cache_key]


def generate_hf_render(control_image_paths, controlnet_model_ids, output_image_path, controlnet_conditioning_scale=None, prompt=None, negative_prompt=None):
    """Generate a photorealistic image from one or multiple control images."""
    try:
        import torch
        diffusers_utils = importlib.import_module("diffusers.utils")
        load_image = getattr(diffusers_utils, "load_image")
    except Exception as exc:
        raise RuntimeError(f"Could not import torch/diffusers utilities: {exc}")

    if len(control_image_paths) != len(controlnet_model_ids):
        raise RuntimeError("control_image_paths and controlnet_model_ids must have the same length")

    pipe = _get_or_create_pipeline(controlnet_model_ids)
    loaded_images = [load_image(p) for p in control_image_paths]
    image_input = loaded_images[0] if len(loaded_images) == 1 else loaded_images

    if str(config.HF_DEVICE).lower() == "cpu":
        generator = torch.Generator().manual_seed(int(config.HF_SEED))
    else:
        generator = torch.Generator(device=config.HF_DEVICE).manual_seed(int(config.HF_SEED))

    # --- Handling dynamic prompts ---
    # If a specific prompt (like 3D) was passed, we will use it, otherwise we will take the 2D from the config
    final_prompt = prompt if prompt is not None else config.HF_PROMPT
    final_negative = negative_prompt if negative_prompt is not None else config.HF_NEGATIVE_PROMPT

    output = pipe(
        final_prompt,
        negative_prompt=final_negative,
        image=image_input,
        num_inference_steps=int(config.HF_NUM_INFERENCE_STEPS),
        guidance_scale=float(config.HF_GUIDANCE_SCALE),
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    output.save(output_image_path)
    return output_image_path


def _resolve_control_inputs(raw_snapshot_path, artifacts, playground=None):
    mode = str(config.HF_CONTROL_MODE).strip().lower()

    if mode == "seg":
        seg_path = process_cad_to_seg_map(raw_snapshot_path, artifacts["segmentation_map"], playground=playground)
        return {
            "mode": mode,
            "control_images": [seg_path],
            "controlnet_models": [config.HF_CONTROLNET_MODEL_ID_SEG],
            "conditioning_scale": None,
        }

    if mode == "canny":
        canny_path = create_canny_control_map(raw_snapshot_path, artifacts["canny_map"])
        return {
            "mode": mode,
            "control_images": [canny_path],
            "controlnet_models": [config.HF_CONTROLNET_MODEL_ID_CANNY],
            "conditioning_scale": None,
        }

    if mode == "multi":
        seg_path = process_cad_to_seg_map(raw_snapshot_path, artifacts["segmentation_map"], playground=playground)
        canny_path = create_canny_control_map(raw_snapshot_path, artifacts["canny_map"])
        scales = list(config.HF_MULTI_CONTROL_SCALES)
        if len(scales) != 2:
            raise RuntimeError("HF_MULTI_CONTROL_SCALES must contain exactly two values: [seg, canny]")
        return {
            "mode": mode,
            "control_images": [seg_path, canny_path],
            "controlnet_models": [config.HF_CONTROLNET_MODEL_ID_SEG, config.HF_CONTROLNET_MODEL_ID_CANNY],
            "conditioning_scale": scales,
        }

    raise RuntimeError(
        f"Unsupported HF_CONTROL_MODE='{config.HF_CONTROL_MODE}'. Use one of: seg, canny, multi."
    )


def run_post_done_render(scene, apartment_id=None, playground=None):
    """
    Runs the pipeline twice: once for 2D and a second time for 3D.
    """
    artifacts = _build_artifact_paths(apartment_id)

    raw_2d = artifacts["raw_snapshot"].replace("_raw.png", "_raw_2d.png")
    raw_3d = artifacts["raw_snapshot"].replace("_raw.png", "_raw_3d.png")
    render_2d_path = artifacts["rendered_image"].replace("_render.png", "_render_2d.png")
    render_3d_path = artifacts["rendered_image"].replace("_render.png", "_render_3d.png")

    # ==========================================
    # First run: 2D Top-Down
    # ==========================================
    capture_scene_snapshot(scene, raw_2d)
    art_2d = {
        "segmentation_map": artifacts["segmentation_map"].replace("_seg.png", "_seg_2d.png"),
        "canny_map": artifacts["canny_map"].replace("_canny.png", "_canny_2d.png")
    }
    control_2d = _resolve_control_inputs(raw_2d, art_2d, playground=playground)

    if config.ENABLE_POST_DONE_RENDER:
        generate_hf_render(
            control_image_paths=control_2d["control_images"],
            controlnet_model_ids=control_2d["controlnet_models"],
            output_image_path=render_2d_path,
            controlnet_conditioning_scale=control_2d["conditioning_scale"],
            prompt=config.HF_PROMPT,
            negative_prompt=config.HF_NEGATIVE_PROMPT
        )
        print(f"[Render Engine] 2D Render complete: {render_2d_path}")

    # ==========================================
    # Second run: 3D Isometric Dollhouse
    # ==========================================
    shutil.copy(raw_2d, raw_3d)
    apply_isometric_warp(raw_3d)

    # The Fix: Copy the successful 2D maps and squash them!
    # This ensures Canny and Seg are 100% aligned and won't cause hallucinations.
    control_3d_images = []
    for img_2d_path in control_2d["control_images"]:
        img_3d_path = img_2d_path.replace("_2d.png", "_3d.png")
        shutil.copy(img_2d_path, img_3d_path)
        apply_isometric_warp(img_3d_path)  # Squash the map itself
        control_3d_images.append(img_3d_path)

    if config.ENABLE_POST_DONE_RENDER:
        generate_hf_render(
            control_image_paths=control_3d_images,
            controlnet_model_ids=control_2d["controlnet_models"],  # Use the same models
            output_image_path=render_3d_path,
            controlnet_conditioning_scale=config.HF_MULTI_CONTROL_SCALES_3D if control_2d["mode"] == "multi" else None,
            prompt=config.HF_PROMPT_3D,
            negative_prompt=config.HF_NEGATIVE_PROMPT_3D
        )
        print(f"[Render Engine] 3D Render complete: {render_3d_path}")

    return {"render_2d": render_2d_path, "render_3d": render_3d_path}

