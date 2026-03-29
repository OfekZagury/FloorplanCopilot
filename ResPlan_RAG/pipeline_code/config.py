import os

# ==========================================
# Database & File Paths
# ==========================================
# The path to the downloaded ResPlan dataset
RESPLAN_DATA_PATH = '../resplan_dataset_files/ResPlan.pkl'

# The path where the SQLite database will be saved
SQLITE_DB_PATH = 'databases/apartments_raw.db'

# The path where the ChromaDB vector database will be saved
CHROMA_DB_PATH = 'databases/chroma_db'

# ==========================================
# API Keys
# ==========================================
# Gemini API Key for Embeddings and LLM generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# ==========================================
# Application Settings
# ==========================================
# Collection name for ChromaDB
CHROMA_COLLECTION_NAME = 'floorplans'

# ==========================================
# Snapshot + Render Settings (post-editor Done)
# ==========================================
PIPELINE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_CODE_DIR, '..', '..'))
SNAPSHOT_OUTPUT_DIR = os.path.join(PIPELINE_CODE_DIR, 'snapshots')

# Toggle for generating a photorealistic render after GUI Done.
ENABLE_POST_DONE_RENDER = True

# Hugging Face model settings
HF_BASE_MODEL_ID = 'runwayml/stable-diffusion-v1-5'

# Control mode options: seg | canny | multi
HF_CONTROL_MODE = 'multi'

# Single-control model IDs
HF_CONTROLNET_MODEL_ID_SEG = 'lllyasviel/control_v11p_sd15_seg'
HF_CONTROLNET_MODEL_ID_CANNY = 'lllyasviel/control_v11p_sd15_canny'

# Multi-control order is fixed: [seg, canny]
HF_MULTI_CONTROL_SCALES = [1.0, 0.6]

# Geometry-driven room segmentation (ADE20K-style palette values commonly used by ControlNet seg)
HF_SEG_USE_PLAYGROUND_GEOMETRY = True
HF_SEG_ALLOW_FALLBACK_TO_HSV = True
HF_SEG_CANVAS_MARGIN_PX = 12
HF_SEG_WALL_LINE_WIDTH = 3

# Room paint order and semantic colors (RGB)
HF_SEG_ROOM_PRIORITY = ["living", "bedroom", "bathroom", "kitchen"]
HF_SEG_ROOM_COLORS = {
	"living": (80, 50, 50),      # floor-like indoor region
	"bedroom": (204, 5, 255),    # bed-like class color
	"bathroom": (0, 160, 255),   # fixture/wet-area-like class color
	"kitchen": (224, 5, 255),    # cabinet-like class color
	"wall": (120, 120, 120),     # wall class color
}

HF_DEVICE = 'cpu'
HF_TORCH_DTYPE = 'float32'

# Canny preprocessing settings
HF_CANNY_LOW_THRESHOLD = 100
HF_CANNY_HIGH_THRESHOLD = 200

# Generation settings
HF_NUM_INFERENCE_STEPS = 12
HF_GUIDANCE_SCALE = 7.5
HF_SEED = 42
HF_PROMPT = (
	'A photorealistic top-down architectural floor plan of a modern apartment. '
	'High-end interior design, wooden flooring, realistic furniture textures, '
	'bright natural lighting, 8k resolution, cinematic render, architectural photography.'
)
HF_NEGATIVE_PROMPT = 'text, watermark, blurry, low quality, distorted, messy, labels'


# --- 3D Isometric Render Settings ---
HF_PROMPT_3D = "3D isometric dollhouse view, angled camera, high quality architectural render, extruded walls, realistic lighting, unreal engine 5, detailed interior design"
HF_NEGATIVE_PROMPT_3D = "top-down, flat, 2D plan, blueprint, drawing, wireframe, ugly, blurry"
HF_MULTI_CONTROL_SCALES_3D = [0.85, 0.35]