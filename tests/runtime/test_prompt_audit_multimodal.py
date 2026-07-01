import json

from src.tiernav_runtime.recorder import PromptAuditRecorder, sanitize_multimodal_messages


def test_sanitize_multimodal_messages_replaces_base64_with_metadata():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("a" * 80)},
                },
            ],
        }
    ]

    sanitized = sanitize_multimodal_messages(messages)

    image_url = sanitized[0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,<omitted chars=80 sha256=")
    assert "aaaaaaaaaa" not in image_url


def test_prompt_audit_records_multimodal_without_raw_base64(tmp_path):
    recorder = PromptAuditRecorder(tmp_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("b" * 64)},
                },
            ],
        }
    ]

    recorder.record_multimodal(
        episode_id="ep-1",
        round_index=2,
        step_index=3,
        label="answerer",
        messages=messages,
    )

    path = tmp_path / "prompt_audit" / "ep-1.multimodal.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    url = payload["messages"][0]["content"][1]["image_url"]["url"]

    assert payload["label"] == "answerer"
    assert payload["round"] == 2
    assert payload["step"] == 3
    assert "<omitted chars=64 sha256=" in url
    assert "bbbbbbbbbb" not in json.dumps(payload)
