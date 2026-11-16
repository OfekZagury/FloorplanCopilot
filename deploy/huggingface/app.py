"""Hugging Face Space entry point for the FloorplanCopilot live demo.

On a ZeroGPU Space only the Stable Diffusion render runs on a GPU slot; search,
scaling, the furniture editor and the DXF export run on CPU. Locally (no `spaces`
package) the same file runs the app without the render step.
"""
import os
import threading
import time

try:
    import spaces  # preinstalled on Hugging Face Spaces
except ImportError:
    spaces = None

ON_ZEROGPU = spaces is not None and bool(os.getenv("SPACE_ID"))

# These must be set before the pipeline modules read their config.
os.environ.setdefault("ENABLE_WEB_EDITOR", "true")
os.environ.setdefault("GRADIO_HOST", "0.0.0.0")
os.environ.setdefault("GRADIO_PORT", os.getenv("PORT", "7860"))
if ON_ZEROGPU:
    os.environ.setdefault("ENABLE_POST_DONE_RENDER", "true")
    os.environ.setdefault("HF_DEVICE", "cuda")
    os.environ.setdefault("HF_TORCH_DTYPE", "float16")

from ResPlan_RAG.pipeline_code import config  # noqa: E402
from ResPlan_RAG.pipeline_code.gradio_GUI import gradio_app  # noqa: E402

OUTPUT_MAX_AGE_SECONDS = 60 * 60


def _cleanup_old_outputs():
    """Delete exported DXF files and renders older than an hour."""
    while True:
        cutoff = time.time() - OUTPUT_MAX_AGE_SECONDS
        for folder in gradio_app.output_dirs():
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    path = os.path.join(root, name)
                    try:
                        if os.path.getmtime(path) < cutoff:
                            os.remove(path)
                    except OSError:
                        pass
        time.sleep(10 * 60)


if ON_ZEROGPU:
    # Allocate a GPU only while the diffusion render runs.
    gradio_app._run_render_from_editor_image = spaces.GPU(duration=120)(
        gradio_app._run_render_from_editor_image
    )

    # Load the ControlNet pipeline once at startup (ZeroGPU defers the .to("cuda")),
    # so a visitor's first render doesn't wait for multi-GB downloads.
    try:
        from ResPlan_RAG.pipeline_code.snapshot_pipeline import snapshot_to_render

        snapshot_to_render._get_or_create_pipeline(
            [config.HF_CONTROLNET_MODEL_ID_SEG, config.HF_CONTROLNET_MODEL_ID_CANNY]
        )
        print("[space] Render pipeline loaded")
    except Exception as exc:  # the rest of the app still works without renders
        print(f"[space] Render pipeline not available: {exc}")

threading.Thread(target=_cleanup_old_outputs, daemon=True).start()
gradio_app.main()
