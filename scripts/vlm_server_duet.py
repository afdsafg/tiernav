#!/usr/bin/env python3
"""DUET-VLM powered OpenAI-compatible VLM server for TierNav.

Loads Qwen2.5-VL-7B-Instruct with DUET-VLM compression (VisionZip + PyramidDrop)
and exposes a standard /v1/chat/completions endpoint.

DUET-VLM: Dual-stage Efficient Token reduction
  - Stage 1 (VisionZip): Token clustering at vision encoder → ~67% token reduction
  - Stage 2 (PyramidDrop): Context-aware progressive token pruning in LLM layers
  - Accuracy: 99% of baseline at 67% token reduction

Model: Qwen2.5-VL-7B-Instruct (7B params, ~16GB bf16)
Compression: VisionZip (dominant=170, contextual=35) + PyramidDrop (50%/25%)
Port: 12221 (default, separate from AstraNav 3B on 12220)

Start: nohup /root/miniconda3/envs/tiernav/bin/python scripts/vlm_server_duet.py \
         --host 127.0.0.1 --port 12221 > /root/vlm_duet.log 2>&1 &

Requirements:
  - tiernav conda env with transformers, fastapi, uvicorn, DUET-VLM installed
  - Qwen2.5-VL-7B-Instruct model (symlinked to /media/user_datadisk/models/)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vlm_duet")

# ─── CLI ────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="DUET-VLM Qwen2.5-VL server")
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=12221)
_parser.add_argument(
    "--model-path",
    default="/media/user_datadisk/models/Qwen2.5-VL-7B-Instruct",
)
_parser.add_argument(
    "--visionzip-dominant", type=int, default=170,
    help="VisionZip dominant tokens (keeps top-k by attention score).",
)
_parser.add_argument(
    "--visionzip-contextual", type=int, default=35,
    help="VisionZip contextual tokens (cluster-aggregated).",
)
_parser.add_argument(
    "--pdrop-enabled", action="store_true", default=True,
    help="Enable PyramidDrop stage-2 compression.",
)
_parser.add_argument(
    "--pdrop-layers", type=int, nargs="+", default=[14, 21],
    help="LLM layers where PyramidDrop applies.",
)
_parser.add_argument(
    "--pdrop-ratios", type=float, nargs="+", default=[0.5, 0.25],
    help="Drop ratios per PyramidDrop layer.",
)
CLI = _parser.parse_args()

MODEL_PATH = CLI.model_path
MODEL_NAME = os.path.basename(MODEL_PATH.rstrip("/"))

# ─── Model state ────────────────────────────────────────────────────────────

model = None
processor = None
load_error: Optional[str] = None
generate_lock = asyncio.Lock()


def load_models() -> None:
    global model, processor, load_error
    try:
        # Use DUET-VLM's Qwen2.5-VL (standalone implementation with VisionZip + PyramidDrop)
        from qwen2_5_vl.modeling_qwen2_5vl_duet import Qwen2_5_VLForConditionalGeneration
        from transformers import AutoProcessor

        logger.info("Loading model from %s (bf16, sdpa, device_map=auto) ...", MODEL_PATH)
        t0 = time.time()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()

        # Apply DUET compression
        model.configure_duet(
            visionzip_enabled=True,
            dominant_tokens=CLI.visionzip_dominant,
            contextual_tokens=CLI.visionzip_contextual,
            pdrop_enabled=CLI.pdrop_enabled,
            layer_list=CLI.pdrop_layers,
            ratio_list=CLI.pdrop_ratios,
        )
        logger.info(
            "DUET configured: visionzip=%d/%d pdrop=%s layers=%s ratios=%s",
            CLI.visionzip_dominant, CLI.visionzip_contextual,
            CLI.pdrop_enabled, CLI.pdrop_layers, CLI.pdrop_ratios,
        )

        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        # Ensure chat template is loaded
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and not getattr(tok, "chat_template", None):
            jinja_path = os.path.join(MODEL_PATH, "chat_template.jinja")
            if os.path.exists(jinja_path):
                with open(jinja_path, encoding="utf-8") as fh:
                    tok.chat_template = fh.read()
                logger.info("Loaded chat_template.jinja.")
        logger.info(
            "Model loaded in %.1fs. device=%s",
            time.time() - t0, next(model.parameters()).device,
        )
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"
        logger.error("Model load FAILED: %s\n%s", load_error, traceback.format_exc())


# ─── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="DUET-VLM Server", version="1.0")


@app.on_event("startup")
async def _startup():
    await asyncio.to_thread(load_models)


@app.get("/health")
async def health():
    if model is not None and processor is not None:
        return {"status": "ok", "model": MODEL_NAME, "duet": True}
    return JSONResponse(
        {"status": "error", "detail": load_error or "model not loaded"},
        status_code=503,
    )


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


# ─── Request/response schemas ───────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None


# ─── Image utilities ────────────────────────────────────────────────────────

TARGET_SIZE = (720, 640)

def decode_data_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, b64_part = url.split(",", 1)
    else:
        b64_part = url
    data = base64.b64decode(b64_part)
    return Image.open(io.BytesIO(data)).convert("RGB")


def convert_messages(msgs: list[ChatMessage]) -> tuple[list[dict], list[Image.Image]]:
    chat_messages = []
    images = []
    for msg in msgs:
        content = msg.content
        if isinstance(content, str):
            chat_messages.append({"role": msg.role, "content": content})
            continue
        content_parts = []
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                img = decode_data_url(url)
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                images.append(img)
                content_parts.append({"type": "image", "image": img})
            else:
                content_parts.append({"type": "text", "text": str(part.get("text", ""))})
        chat_messages.append({"role": msg.role, "content": content_parts})
    return chat_messages, images


def truncate_at_stop(text: str, stop: Any) -> str:
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


def run_generate(req: ChatCompletionRequest, chat_messages, images) -> dict:
    tok = getattr(processor, "tokenizer", processor)
    text = tok.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if images:
        proc_kwargs["images"] = images
    inputs = processor(**proc_kwargs)
    inputs = inputs.to(model.device)

    max_new_tokens = req.max_tokens or 4096
    do_sample = bool(req.temperature and req.temperature > 0)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p or 0.9

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs.input_ids.shape[1]
    new_ids = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    output_text = truncate_at_stop(output_text, req.stop)
    return {
        "id": f"chatcmpl-{os.urandom(12).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": new_ids.shape[1],
            "total_tokens": prompt_len + new_ids.shape[1],
        },
    }


# ─── Endpoint ───────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None or processor is None:
        return JSONResponse(
            {"error": {"message": "Model not loaded", "type": "api_error", "code": None}},
            status_code=503,
        )
    try:
        chat_messages, images = convert_messages(req.messages)
        t0 = time.time()
        prompt_chars = len(json.dumps(chat_messages))
        logger.info(
            "request: messages=%d images=%d prompt_chars=%d max_tokens=%d temp=%.1f",
            len(req.messages), len(images), prompt_chars,
            req.max_tokens or 4096, req.temperature or 0.3,
        )
        result = await asyncio.to_thread(run_generate, req, chat_messages, images)
        logger.info(
            "response: latency=%.2fs out_chars=%d completion_tokens=%d",
            time.time() - t0, len(result["choices"][0]["message"]["content"]),
            result["usage"]["completion_tokens"],
        )
        return result
    except Exception:
        logger.error("generate failed:\n%s", traceback.format_exc())
        return JSONResponse(
            {"error": {"message": f"model error: {traceback.format_exc()[-500:]}",
                       "type": "api_error", "code": None}},
            status_code=500,
        )


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("DUET-VLM server starting on %s:%d (model=%s)", CLI.host, CLI.port, MODEL_PATH)
    uvicorn.run(app, host=CLI.host, port=CLI.port, log_level="info", workers=1)
