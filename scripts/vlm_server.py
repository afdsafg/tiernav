"""FastAPI server exposing AstraNav-Memory's Qwen2.5-VL-3B + DINOv3 context-
compression VLM as an OpenAI-compatible HTTP API.

This server replaces the `mimo-v2.5` cloud API used by the tiernav project's
`call_vlm` (src/agent_workflow.py) and `Planner` (src/agent_planner.py) classes.
It loads the fused DS16-Context100 checkpoint (Qwen2.5-VL-3B with DINOv3 +
conv compression fused into the vision encoder) ONCE at startup and serves:

    POST /v1/chat/completions  — OpenAI-compatible chat (images via data: URLs)
    GET  /v1/models            — list loaded model
    GET  /health               — health check (503 until model is loaded)

Deployment (server 8.147.163.63):
    Conda env:  /root/miniconda3/envs/tiernav/bin/python
                (custom transformers-4.57.1 installed here — stock transformers
                 lacks use_compression/dino_* config and DINOv3ViTModel module)
    Model:      /media/user_datadisk/models/DS16-Context100  (~8GB bf16)
    GPU:        RTX5880-Ada-48Q (48GB)

    Start:
      nohup /root/miniconda3/envs/tiernav/bin/python scripts/vlm_server.py \\
          --host 127.0.0.1 --port 12220 \\
          > /root/vlm_server.log 2>&1 &

    The tiernav project sets `no_proxy=localhost,127.0.0.1` before requests —
    this is critical, otherwise the xray HTTP proxy intercepts localhost.

    Then in tiernav .env:
      OPENAI_API_KEY=any
      OPENAI_BASE_URL=http://127.0.0.1:12220/v1/chat/completions
      MODEL_NAME=DS16-Context100
"""

import os

# --- Offline / robustness flags (must be set before transformers import) ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import base64
import binascii
import io
import logging
import re
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("vlm_server")

# ─── CLI ───────────────────────────────────────────────────────────────────


def _parse_pixels(s: str) -> int:
    """Parse a 'WxH' or 'W*H' pixel spec into total pixel count."""
    m = re.fullmatch(r"\s*(\d+)\s*[*xX]\s*(\d+)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid pixels spec: {s!r} (use e.g. 4480*4480)"
        )
    return int(m.group(1)) * int(m.group(2))


_parser = argparse.ArgumentParser(
    description="OpenAI-compatible VLM server (AstraNav-Memory Qwen2.5-VL + DINOv3)."
)
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=12220)
_parser.add_argument(
    "--model-path",
    default="/media/user_datadisk/models/DS16-Context100",
)
_parser.add_argument(
    "--max-pixels",
    type=_parse_pixels,
    default=4480 * 4480,
    help="Processor max_pixels as WxH or W*H (default 4480*4480).",
)
_parser.add_argument(
    "--model-name",
    default=None,
    help="Public model name returned by /v1/models (default: basename of --model-path).",
)
CLI = _parser.parse_args()

MODEL_PATH = CLI.model_path
MODEL_NAME = CLI.model_name or os.path.basename(MODEL_PATH.rstrip("/"))
MIN_PIXELS = 56 * 56
MAX_PIXELS = CLI.max_pixels

# ─── Model state (loaded once at startup) ─────────────────────────────────

model = None
processor = None
load_error: Optional[str] = None
generate_lock = asyncio.Lock()


def load_models() -> None:
    """Load Qwen2.5-VL + custom processor.

    Sets module-level `model`/`processor`. On failure, sets `load_error` so
    /health and request handlers can report it.
    """
    global model, processor, load_error
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(
            "Loading model from %s (bf16, sdpa, device_map=auto) ...", MODEL_PATH
        )
        t0 = time.time()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        logger.info(
            "Model loaded in %.1fs. device=%s",
            time.time() - t0,
            next(model.parameters()).device,
        )
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"
        logger.error("Model load FAILED: %s\n%s", load_error, traceback.format_exc())


# ─── Request/response schemas ─────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: Any  # str | list[dict]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    stop: Optional[Any] = None  # str | list[str]
    stream: Optional[bool] = False  # accepted but ignored (no streaming)


# ─── Helpers ───────────────────────────────────────────────────────────────

_DATA_URL_RE = re.compile(r"^data:[^;]*;base64,(.*)$", re.DOTALL)


def decode_data_url(url: str) -> Image.Image:
    """Decode a `data:image/...;base64,...` URL into a PIL RGB image."""
    if not isinstance(url, str):
        raise ValueError("image_url must be a string")
    m = _DATA_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            "image_url must be a data: URL (data:image/...;base64,...)"
        )
    raw = m.group(1)
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 image: {e}")
    if not data:
        raise ValueError("empty image data")
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot open image: {e}")


def build_chat_messages(req_messages: list[ChatMessage]):
    """Convert OpenAI-style messages into Qwen2.5-VL chat-template format.

    Returns (chat_messages, images):
      - chat_messages: list of {"role","content"} dicts. String content is kept
        as a plain string (system/assistant messages). List content is rewritten
        to Qwen format: {"type":"image","image":<PIL>} placeholders +
        {"type":"text","text":...} parts.
      - images: flat ordered list of PIL images for processor(images=...).
    """
    chat_messages: list[dict] = []
    images: list[Image.Image] = []
    for msg in req_messages:
        if isinstance(msg.content, str):
            chat_messages.append({"role": msg.role, "content": msg.content})
            continue
        if not isinstance(msg.content, list):
            raise ValueError(
                f"message content must be str or list, got {type(msg.content)}"
            )
        content_parts: list[dict] = []
        for part in msg.content:
            if not isinstance(part, dict):
                raise ValueError(
                    f"content part must be dict, got {type(part)}"
                )
            ptype = part.get("type")
            if ptype == "text":
                content_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url_obj = part.get("image_url") or {}
                url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                img = decode_data_url(url)
                images.append(img)
                content_parts.append({"type": "image", "image": img})
            else:
                # Pass through anything else as text.
                content_parts.append(
                    {"type": "text", "text": str(part.get("text", ""))}
                )
        chat_messages.append({"role": msg.role, "content": content_parts})
    return chat_messages, images


def truncate_at_stop(text: str, stop: Any) -> str:
    """Cut `text` at the first occurrence of any stop string."""
    if not stop:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = len(text)
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    return text[:cut] if cut < len(text) else text


def run_generate(
    req: ChatCompletionRequest, chat_messages, images
) -> dict:
    """Blocking: run model.generate and return an OpenAI-style response dict."""
    text = processor.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    proc_kwargs = dict(
        text=[text],
        padding=True,
        return_tensors="pt",
        use_compression=True,
        compression_times=2,
        dino_size="vitb16",
    )
    if images:
        proc_kwargs["images"] = images
    inputs = processor(**proc_kwargs)
    inputs = inputs.to(model.device)

    max_new_tokens = req.max_tokens or 4096
    do_sample = bool(req.temperature and req.temperature > 0)
    gen_kwargs: dict = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p if req.top_p is not None else 0.9

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs.input_ids.shape[1]
    new_ids = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    completion_tokens = int(new_ids.shape[1])

    if req.stop:
        output_text = truncate_at_stop(output_text, req.stop)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": completion_tokens,
            "total_tokens": int(prompt_len) + completion_tokens,
        },
    }


def _error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status == 400 else "api_error",
                "code": None,
            }
        },
    )


# ─── FastAPI app ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loads in a worker thread so the event loop stays responsive.
    await asyncio.to_thread(load_models)
    yield


app = FastAPI(title="AstraNav-Memory VLM Server", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading" if load_error is None else "error",
                "error": load_error,
            },
        )
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "astranav",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return _error_response(400, f"invalid JSON body: {e}")
    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        return _error_response(400, f"invalid request: {e}")

    if model is None or processor is None:
        return _error_response(
            503, f"model not loaded: {load_error or 'still loading'}"
        )

    try:
        chat_messages, images = build_chat_messages(req.messages)
    except ValueError as e:
        logger.warning("bad request: %s", e)
        return _error_response(400, str(e))

    prompt_chars = 0
    for m in req.messages:
        if isinstance(m.content, str):
            prompt_chars += len(m.content)
        elif isinstance(m.content, list):
            for p in m.content:
                if isinstance(p, dict) and p.get("type") == "text":
                    prompt_chars += len(p.get("text", ""))
    logger.info(
        "request: messages=%d images=%d prompt_chars=%d max_tokens=%s temp=%s",
        len(req.messages), len(images), prompt_chars, req.max_tokens, req.temperature,
    )

    t0 = time.time()
    try:
        async with generate_lock:
            result = await asyncio.to_thread(
                run_generate, req, chat_messages, images
            )
    except Exception as e:
        logger.error("generate failed: %s\n%s", e, traceback.format_exc())
        return _error_response(500, f"model error: {e}")

    latency = time.time() - t0
    out_text = result["choices"][0]["message"]["content"]
    logger.info(
        "response: latency=%.2fs out_chars=%d completion_tokens=%d",
        latency, len(out_text), result["usage"]["completion_tokens"],
    )
    return result


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting server on %s:%d (model=%s)", CLI.host, CLI.port, MODEL_PATH
    )
    uvicorn.run(app, host=CLI.host, port=CLI.port, workers=1, log_level="info")
