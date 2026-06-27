#!/usr/bin/env python3
"""Score AEQA results using 3D-Mem-AEQA-Eval.

Usage:
    python scripts/score_aeqa.py --result-dir <dir> --eval-tool <path-to-3D-Mem-AEQA-Eval>

Produces: LLM Match (%) and LLM Match SPL (%) printed to stdout + saved to
<result-dir>/scores.json.
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True, help="Dir containing gpt_answer.json + path_length_list.pkl")
    ap.add_argument("--eval-tool", required=True, help="Path to 3D-Mem-AEQA-Eval repo")
    ap.add_argument("--dataset", default="open-eqa-41")
    ap.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = ap.parse_args()

    gpt_answer = os.path.join(args.result_dir, "gpt_answer.json")
    assert os.path.exists(gpt_answer), f"Missing {gpt_answer}"

    env = {**os.environ, "OPENAI_API_KEY": args.openai_api_key}

    # Step 1: LLM Match scoring
    print("=== Step 1: evaluate-predictions.py (LLM Match) ===")
    cmd1 = [
        sys.executable, os.path.join(args.eval_tool, "evaluate-predictions.py"),
        "--dataset", os.path.join(args.eval_tool, "data", f"{args.dataset}.json"),
        gpt_answer,
    ]
    print(" ".join(cmd1))
    r1 = subprocess.run(cmd1, cwd=args.eval_tool, env=env, capture_output=True, text=True)
    print(r1.stdout)
    if r1.returncode != 0:
        print("STDERR:", r1.stderr)
        sys.exit(1)

    # Step 2: SPL + accuracy over 41 questions
    print("=== Step 2: get-scores-41.py (SPL) ===")
    cmd2 = [
        sys.executable, os.path.join(args.eval_tool, "get-scores-41.py"),
        "--result-path", args.result_dir,
        "--dataset", args.dataset,
    ]
    print(" ".join(cmd2))
    r2 = subprocess.run(cmd2, cwd=args.eval_tool, env=env, capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print("STDERR:", r2.stderr)
        sys.exit(1)

    # Parse + save scores
    scores = {"llm_match_output": r1.stdout, "spl_output": r2.stdout}
    with open(os.path.join(args.result_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {os.path.join(args.result_dir, 'scores.json')}")


if __name__ == "__main__":
    main()
