# Grid World Strategy A Smoke Project

This is the smallest runnable version for Strategy A: adjacent exact feedback.

## 1. Install

```bash
cd gridwm_strategy_A
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Run the engineering smoke test

```bash
python scripts/smoke_test.py --config configs/exp_A_adjacent.yaml
```

The default backend is `mock_oracle`, which is only for debugging the pipeline.
It proves that map generation, Strategy A feedback, prompt construction, JSON parsing, environment stepping, and logging all work.

## 3. Run with a real 7B HF model

First install optional dependencies and make sure you have GPU/Hugging Face access:

```bash
pip install torch transformers accelerate sentencepiece safetensors
```

Edit `configs/exp_A_adjacent.yaml`:

```yaml
backend: hf
model_name: Qwen/Qwen2.5-7B-Instruct
num_episodes: 1
```

Then run:

```bash
python scripts/smoke_test.py --config configs/exp_A_adjacent.yaml
```

Outputs:

- `outputs/logs/strategy_A_smoke/steps.jsonl`
- `outputs/logs/strategy_A_smoke/summary.json`

## Coordinate convention

This project uses Cartesian grid coordinates externally.

- Coordinates are `[x, y]`.
- The origin `[0, 0]` is the bottom-left cell.
- `x` increases to the right.
- `y` increases upward.
- `UP` means `y + 1`.
- `DOWN` means `y - 1`.
- `LEFT` means `x - 1`.
- `RIGHT` means `x + 1`.
- Internally, `obstacle_map` is indexed as `obstacle_map[y][x]`.
- For human-readable display, grids are printed top-to-bottom, from `y=size-1` down to `y=0`.

