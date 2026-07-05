# Grid Planner: Strategy A World-Model Experiment

This repository contains a minimal grid-world experiment for studying whether a language model can maintain an explicit world model while navigating under partial observation.

The current implementation focuses on **Strategy A: exact adjacent-cell feedback**.

The experiment is intentionally small and controllable:

- 5x5 grid world
- start at `[0, 0]`
- goal at `[4, 4]`
- fixed number of obstacles
- exact adjacent-cell feedback after each step
- model output includes both an explicit `belief_grid` and an action
- step-level logs are saved for behavior analysis and world-model analysis

The current model backend supports:

- `mock_oracle`
- Hugging Face causal language models, currently tested with `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Coordinate convention

The project uses Cartesian grid coordinates externally.

- Coordinates are written as `[x, y]`.
- The origin `[0, 0]` is the bottom-left cell.
- `x` increases to the right.
- `y` increases upward.
- `UP` means `y + 1`.
- `DOWN` means `y - 1`.
- `LEFT` means `x - 1`.
- `RIGHT` means `x + 1`.

Internally, the true obstacle map is indexed as:

```python
obstacle_map[y][x]
```

For human-readable display, grids are printed from top to bottom:

```text
y=4  . . . . G
y=3  . . . . .
y=2  . . . . .
y=1  . . . . .
y=0  S . . . .
     0 1 2 3 4
     x-axis, origin (0,0) is bottom-left
```

The model-produced `belief_grid` is also expected to use top-to-bottom row order:

- first row: `y = size - 1`
- last row: `y = 0`

---

## 2. Strategy A: exact adjacent-cell feedback

At each step, the environment returns the exact state of the four adjacent cells.

Example feedback:

```json
{
  "type": "adjacent_exact",
  "coordinate_system": "cartesian_bottom_left",
  "position": [1, 0],
  "blocked": [[2, 0]],
  "free": [[1, 1], [0, 0]],
  "wall": [[1, -1]]
}
```

The model is prompted to output JSON only:

```json
{
  "thought": "brief reasoning",
  "nl_obstacles": "brief obstacle/free-cell summary",
  "belief_grid": [
    ["U", "U", "U", "U", "F"],
    ["U", "U", "U", "U", "U"],
    ["U", "U", "U", "U", "U"],
    ["U", "U", "U", "U", "U"],
    ["F", "U", "U", "U", "U"]
  ],
  "action": "RIGHT"
}
```

The three belief labels are:

- `O`: known obstacle
- `F`: known free cell
- `U`: unknown cell

---

## 3. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

For Hugging Face runs, make sure the environment has access to the target model and enough GPU memory.

---

## 4. Smoke test with mock oracle

Run the default smoke test:

```bash
PYTHONPATH=. python scripts/smoke_test.py --config configs/exp_A_adjacent.yaml
```

Expected behavior:

- the mock oracle reaches the goal
- no parse errors
- no invalid moves
- path length equals the shortest path length

Outputs are saved to:

```text
outputs/logs/strategy_A_smoke/steps.jsonl
outputs/logs/strategy_A_smoke/summary.json
```

---

## 5. Ten-seed mock run

The ten-seed mock configuration is:

```text
configs/exp_A_adjacent_mock_10.yaml
```

Run:

```bash
rm -rf outputs/logs/strategy_A_mock_10

PYTHONPATH=. python scripts/smoke_test.py \
  --config configs/exp_A_adjacent_mock_10.yaml
```

Analyze:

```bash
PYTHONPATH=. python scripts/analyze_strategy_A_run.py \
  --run outputs/logs/strategy_A_mock_10
```

Current mock result:

```text
episodes: 10
success: 10/10 = 1.000
mean_steps_success_only: 8
mean_optimality_gap_success_only: 0
total_parse_errors: 0
total_invalid_moves: 0
total_illegal_action_before_repair: 0
total_repair_used: 0
total_repeated_position_action_pairs: 0
```

This confirms that the environment, A* path oracle, logging pipeline, and basic analysis scripts are working correctly.

---

## 6. Ten-seed Qwen run

The ten-seed Qwen configuration is:

```text
configs/exp_A_adjacent_qwen_10.yaml
```

Run:

```bash
rm -rf outputs/logs/strategy_A_qwen_10
mkdir -p outputs/logs

PYTHONPATH=. python scripts/smoke_test.py \
  --config configs/exp_A_adjacent_qwen_10.yaml \
  2>&1 | tee outputs/logs/strategy_A_qwen_10_console.log
```

Analyze behavior:

```bash
PYTHONPATH=. python scripts/analyze_strategy_A_run.py \
  --run outputs/logs/strategy_A_qwen_10
```

Current Qwen behavioral result:

```text
success: 9/10 = 0.900
```

Per-episode result:

```text
A_seed123 | success=True  | steps=11 | shortest=8 | gap=3
A_seed124 | success=True  | steps= 8 | shortest=8 | gap=0
A_seed125 | success=True  | steps=12 | shortest=8 | gap=4
A_seed126 | success=False | steps=20 | shortest=8 | gap=None
A_seed127 | success=True  | steps=10 | shortest=8 | gap=2
A_seed128 | success=True  | steps= 8 | shortest=8 | gap=0
A_seed129 | success=True  | steps=12 | shortest=8 | gap=4
A_seed130 | success=True  | steps=10 | shortest=8 | gap=2
A_seed131 | success=True  | steps= 8 | shortest=8 | gap=0
A_seed132 | success=True  | steps= 8 | shortest=8 | gap=0
```

The failed case `A_seed126` is useful for later hidden-state analysis because the model gets stuck in a local loop while the environment still provides exact adjacent feedback.

---

## 7. Behavioral analysis

Script:

```text
scripts/analyze_strategy_A_run.py
```

Run:

```bash
PYTHONPATH=. python scripts/analyze_strategy_A_run.py \
  --run outputs/logs/strategy_A_qwen_10
```

This script reports:

- episode success rate
- steps used
- shortest path length
- optimality gap
- parse errors
- invalid moves
- illegal actions before repair
- repair usage
- repeated position-action pairs

Behavioral metrics answer:

```text
Can the model reach the goal?
How efficient is the path?
Does the model produce valid JSON?
Does it choose legal actions?
Does it get stuck in repeated local behavior?
```

Behavioral metrics do not by themselves prove that the model maintains a correct world model.

---

## 8. World-model analysis

Script:

```text
scripts/score_world_model_A.py
```

Run:

```bash
PYTHONPATH=. python scripts/score_world_model_A.py \
  --run outputs/logs/strategy_A_qwen_10
```

This script reconstructs the gold Strategy A belief state from the full history of exact adjacent-cell feedback.

It compares the reconstructed gold belief with the model-produced `belief_grid`.

Metrics:

- `all_cell_acc`
- `known_cell_acc`
- `overclaim_rate`
- `unknown_preservation_rate`
- `obstacle_precision`
- `obstacle_recall`
- `obstacle_f1`
- `free_precision`
- `free_recall`
- `free_f1`
- `action_hits_own_obstacle`
- `action_into_unknown`
- `monotonicity_violations`

Current Qwen world-model result:

```text
success_rate: 0.9
mean_all_cell_acc: 0.5673606060606061
mean_known_cell_acc: 0.21566405072538974
mean_overclaim_rate: 0.07184598360875451
mean_unknown_preservation_rate: 0.9281540163912455
mean_obstacle_f1: 0.6666666666666666
mean_free_f1: 0.3623576838917901
total_action_hits_own_obstacle: 0
total_action_into_unknown: 93
total_monotonicity_violations: 26
```

Interpretation:

The model shows stronger behavior-level navigation than explicit world-model maintenance.

Although Qwen reaches the goal in 9 out of 10 episodes, its explicit `belief_grid` is not a reliable persistent state representation.

The main observed failure mode is under-updating:

- the model usually preserves unknown cells as `U`
- it does not heavily overclaim unseen cells
- however, it often fails to mark directly observed free cells as `F`
- it frequently chooses actions into cells that remain `U` in its own `belief_grid`

This suggests that the model may be using local `available_actions` for action selection without fully integrating the same information into the explicit belief grid.

---

## 9. Coordinate and row-order diagnostic

Script:

```text
scripts/diagnose_world_model_A.py
```

Run:

```bash
PYTHONPATH=. python scripts/diagnose_world_model_A.py \
  --run outputs/logs/strategy_A_qwen_10 \
  --episodes A_seed126 A_seed131 A_seed132 A_seed123
```

This script checks whether belief errors can be explained by row-order confusion.

It compares two interpretations of the model's `belief_grid`:

1. `topdown_y_desc`

   - first row is `y = size - 1`
   - last row is `y = 0`
   - this is the intended project convention

2. `bottomup_y_asc`

   - first row is `y = 0`
   - last row is `y = size - 1`
   - this is a possible wrong interpretation

Current diagnostic result:

```text
orientation_guess: row_order_confusion_not_sufficient
mean_known_acc_delta_bottomup_minus_topdown: -0.14252378336675936
```

Per-episode orientation result:

```text
A_seed123 | top_known=0.1705 | bottom_known=0.0360 | delta=-0.1345
A_seed124 | top_known=0.2161 | bottom_known=0.0803 | delta=-0.1358
A_seed125 | top_known=0.2134 | bottom_known=0.0406 | delta=-0.1728
A_seed126 | top_known=0.1965 | bottom_known=0.0756 | delta=-0.1210
A_seed127 | top_known=0.2566 | bottom_known=0.0638 | delta=-0.1928
A_seed128 | top_known=0.2406 | bottom_known=0.1641 | delta=-0.0765
A_seed129 | top_known=0.1467 | bottom_known=0.0608 | delta=-0.0859
A_seed130 | top_known=0.2562 | bottom_known=0.1618 | delta=-0.0944
A_seed131 | top_known=0.2324 | bottom_known=0.0360 | delta=-0.1964
A_seed132 | top_known=0.2276 | bottom_known=0.0125 | delta=-0.2151
```

Interpretation:

Bottom-up row interpretation makes known-cell accuracy worse, not better.

Therefore, the belief errors are not mainly caused by simple row-order confusion.

The more likely explanation is a true explicit belief-update failure or persistent-state maintenance failure.

---

## 10. Diagnostic case files

The row-order diagnostic exports case-level Markdown files:

```text
outputs/logs/strategy_A_qwen_10/diagnostics/A_seed126.md
outputs/logs/strategy_A_qwen_10/diagnostics/A_seed131.md
outputs/logs/strategy_A_qwen_10/diagnostics/A_seed132.md
outputs/logs/strategy_A_qwen_10/diagnostics/A_seed123.md
```

Recommended comparison groups:

- `A_seed126`: failed episode with looping behavior
- `A_seed131`: successful optimal episode
- `A_seed132`: successful optimal episode
- `A_seed123`: successful episode with parse and invalid-move issues

Useful command:

```bash
sed -n '1,200p' outputs/logs/strategy_A_qwen_10/diagnostics/A_seed126.md
```

These case files include:

- step index
- previous position
- action
- next position
- legal actions
- feedback
- gold belief
- raw model belief grid
- top-down interpretation
- bottom-up interpretation
- raw model response

They are intended for qualitative analysis and later hidden-state extraction.

---

## 11. Current research takeaway

The current Strategy A results support a clear distinction between behavioral success and explicit world-model correctness.

The model can often navigate successfully, especially when legal local actions are explicitly provided.

However, the model-produced `belief_grid` does not behave like a stable persistent world model.

Key observations:

- success rate is high relative to belief accuracy
- known-cell accuracy is low
- overclaim rate is low
- unknown preservation rate is high
- many selected action targets are still `U` in the model's own belief grid
- row-order confusion does not explain the error
- the failed episode shows local looping despite exact adjacent feedback

Working hypothesis:

```text
Qwen may use local action affordances for immediate navigation while failing to maintain a faithful explicit belief state across steps.
```

This motivates future mechanistic interpretability analysis.

---

## 12. Suggested next step for mechanistic interpretability

The next stage should avoid immediately repairing the prompt.

Instead, use the current logs to extract hidden states and compare:

- successful optimal episodes vs failed looping episodes
- high world-model accuracy steps vs low world-model accuracy steps
- action-correct but belief-wrong steps
- steps where action targets are `U` in the explicit belief grid
- steps with monotonicity violations

A useful starting point is `A_seed126`, because it is the clearest failure case in the current Qwen run.

Potential probing questions:

```text
Is local obstacle/free-cell information encoded in hidden states even when it is missing from the explicit belief_grid?
Is available_actions information more strongly represented near the action token than near belief_grid tokens?
Do looping failures correspond to weak or unstable representations of previously visited states?
Does the model internally know the next cell is free while failing to write it as F?
```

These questions directly connect the behavioral/world-model gap to mechanistic interpretability.

---

## 13. Git hygiene

Do not commit experiment outputs:

```text
outputs/
__pycache__/
*.pyc
```

Useful cleanup commands:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
git status
```

Commit only source code, configs, and documentation:

```bash
git add README.md scripts configs src
git commit -m "Document Strategy A world-model diagnostics"
git push
```


