import os

# about habitat scene
INVALID_SCENE_ID = []

# about chatgpt api
END_POINT = os.getenv("END_POINT", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")

# HMGE Agent API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://opencode.ai/zen/go/v1/chat/completions",
)
MODEL_NAME = os.getenv("MODEL_NAME", "mimo-v2.5")

# Planner API (mimo-v2.5 via opencode.ai; defaults to HMGE Agent API values)
# LEGACY (Task 9): These import-time constants are only used by legacy
# backup code paths (call_vlm, legacy Planner).  Runtime runners use
# ProviderConfig with env-var names for lazy resolution — see
# src/tiernav_runtime/config.py.  Do not add new dependencies on these
# constants.
QWEN_PLANNER_API_KEY = os.getenv("QWEN_PLANNER_API_KEY", OPENAI_API_KEY)
QWEN_PLANNER_BASE_URL = os.getenv("QWEN_PLANNER_BASE_URL", OPENAI_BASE_URL)
QWEN_PLANNER_MODEL = os.getenv("QWEN_PLANNER_MODEL", MODEL_NAME)

# GroundingDINO
GROUNDINGDINO_DIR = os.getenv("GROUNDINGDINO_DIR", "/root/ContextNav")
GROUNDINGDINO_CONFIG = os.getenv(
    "GROUNDINGDINO_CONFIG",
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
)
GROUNDINGDINO_WEIGHTS = os.getenv(
    "GROUNDINGDINO_WEIGHTS",
    "data/groundingdino_swint_ogc.pth",
)
