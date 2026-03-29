import argparse
import os
from datetime import datetime

from ResPlan_RAG.pipeline_code import config
from ResPlan_RAG.pipeline_code.snapshot_pipeline.snapshot_to_render import create_canny_control_map, generate_hf_render, process_cad_to_seg_map


def main():
    parser = argparse.ArgumentParser(
        description="Run snapshot preprocessing and optional HF render from an existing raw image."
    )
    parser.add_argument("--input", required=True, help="Path to raw snapshot PNG/JPG")
    parser.add_argument(
        "--apartment-id",
        default="manual",
        help="Apartment id label used in output filename",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only create segmentation map, skip diffusion render",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    os.makedirs(config.SNAPSHOT_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"apt_{args.apartment_id}_{ts}"

    seg_path = os.path.join(config.SNAPSHOT_OUTPUT_DIR, f"{prefix}_seg.png")
    canny_path = os.path.join(config.SNAPSHOT_OUTPUT_DIR, f"{prefix}_canny.png")
    render_path = os.path.join(config.SNAPSHOT_OUTPUT_DIR, f"{prefix}_render.png")

    mode = str(config.HF_CONTROL_MODE).strip().lower()
    control_images = []
    control_models = []
    conditioning_scale = None

    if mode == "seg":
        process_cad_to_seg_map(args.input, seg_path)
        print(f"Segmentation map: {seg_path}")
        control_images = [seg_path]
        control_models = [config.HF_CONTROLNET_MODEL_ID_SEG]
    elif mode == "canny":
        create_canny_control_map(args.input, canny_path)
        print(f"Canny map: {canny_path}")
        control_images = [canny_path]
        control_models = [config.HF_CONTROLNET_MODEL_ID_CANNY]
    elif mode == "multi":
        process_cad_to_seg_map(args.input, seg_path)
        create_canny_control_map(args.input, canny_path)
        print(f"Segmentation map: {seg_path}")
        print(f"Canny map: {canny_path}")
        control_images = [seg_path, canny_path]
        control_models = [config.HF_CONTROLNET_MODEL_ID_SEG, config.HF_CONTROLNET_MODEL_ID_CANNY]
        conditioning_scale = list(config.HF_MULTI_CONTROL_SCALES)
    else:
        raise RuntimeError(f"Unsupported HF_CONTROL_MODE='{config.HF_CONTROL_MODE}'")

    if not args.skip_render:
        out = generate_hf_render(
            control_image_paths=control_images,
            controlnet_model_ids=control_models,
            output_image_path=render_path,
            controlnet_conditioning_scale=conditioning_scale,
        )
        print(f"Rendered image: {out}")


if __name__ == "__main__":
    main()

