from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a grid-world planning agent.\n"
    "Return valid JSON only. No markdown.\n"
    "No extra text."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_layers(spec: str, n_hidden_state_layers: int) -> list[int]:
    """
    hidden_states contains embedding layer + transformer layers.
    For Qwen2.5-7B, this is usually 29 entries: 0..28.
    """
    spec = spec.strip().lower()
    last = n_hidden_state_layers - 1

    if spec == "all":
        return list(range(n_hidden_state_layers))

    if spec == "auto":
        candidates = [0, last // 4, last // 2, (3 * last) // 4, last]
        out: list[int] = []
        for x in candidates:
            if x not in out:
                out.append(x)
        return out

    layers: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = n_hidden_state_layers + idx
        if not (0 <= idx < n_hidden_state_layers):
            raise ValueError(
                f"Layer index {item} resolved to {idx}, "
                f"but valid range is 0..{n_hidden_state_layers - 1}"
            )
        layers.append(idx)

    if not layers:
        raise ValueError("No valid layer selected.")

    return layers


def build_prompt_chat(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_chat(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory, e.g. outputs/logs/strategy_A_qwen_100")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", default="auto", help='auto, all, or comma list, e.g. "0,8,16,24,28"')
    ap.add_argument("--repr-key", default="prompt_last", choices=["prompt_last", "response_last"])
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows; use 50 for a quick test")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}. Run Strategy A first.")

    rows = load_jsonl(steps_path)
    rows = [r for r in rows if not r.get("parse_error", False)]

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    if not rows:
        raise RuntimeError("No valid non-parse-error rows found.")

    print(f"[extract_repr_A] rows={len(rows)}")
    print(f"[extract_repr_A] loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[extract_repr_A] loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_transformer_layers = int(getattr(model.config, "num_hidden_layers", 0))
    n_hidden_state_layers = n_transformer_layers + 1
    selected_layers = parse_layers(args.layers, n_hidden_state_layers)

    print(f"[extract_repr_A] hidden_state_layers={n_hidden_state_layers}")
    print(f"[extract_repr_A] selected_layers={selected_layers}")
    print(f"[extract_repr_A] repr_key={args.repr_key}")

    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        prompt = row["prompt_text"]
        response = row["raw_response_text"]

        prompt_chat = build_prompt_chat(tokenizer, prompt)
        full_chat = build_full_chat(tokenizer, prompt, response)

        prompt_inputs = tokenizer(
            prompt_chat,
            return_tensors="pt",
            add_special_tokens=False,
        )

        full_inputs = tokenizer(
            full_chat,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        seq_len = int(full_inputs["input_ids"].shape[-1])
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        prompt_last_pos = max(0, min(prompt_len - 1, seq_len - 1))
        response_last_pos = seq_len - 1

        if args.repr_key == "prompt_last":
            pos = prompt_last_pos
        else:
            pos = response_last_pos

        with torch.inference_mode():
            out = model(
                **full_inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )

        hs = out.hidden_states
        layer_vecs = []
        for layer_idx in selected_layers:
            v = hs[layer_idx][0, pos, :].detach().float().cpu().numpy()
            if args.dtype == "float16":
                v = v.astype(np.float16)
            else:
                v = v.astype(np.float32)
            layer_vecs.append(v)

        vectors.append(np.stack(layer_vecs, axis=0))

        meta.append(
            {
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "seed": row.get("seed"),
                "grid_size": row.get("grid_size"),
                "feedback_type": row.get("feedback_type"),
                "parsed_action": row.get("parsed_action"),
                "agent_pos_before": row.get("agent_pos_before"),
                "agent_pos_after": row.get("agent_pos_after"),
                "valid_move": row.get("valid_move"),
                "done": row.get("done"),
                "prompt_last_pos": prompt_last_pos,
                "response_last_pos": response_last_pos,
                "selected_pos": pos,
                "seq_len": seq_len,
            }
        )

        if i % 10 == 0 or i == len(rows):
            print(f"[extract_repr_A] processed {i}/{len(rows)}")

    X = np.stack(vectors, axis=0)
    meta_json = np.array([json.dumps(m, ensure_ascii=False) for m in meta])

    if args.out is None:
        out_path = run_dir / f"activations_A_{args.repr_key}.npz"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        layers=np.array(selected_layers, dtype=np.int64),
        meta_json=meta_json,
        model_name=np.array(args.model_name),
        repr_key=np.array(args.repr_key),
    )

    print(f"[extract_repr_A] saved: {out_path}")
    print(f"[extract_repr_A] X shape: {X.shape}  # [steps, layers, hidden_dim]")


if __name__ == "__main__":
    main()
