# GPT-2 Small Induction Head Discovery (Local, No Notebook)

This is a plain-Python (venv-based) project that reproduces the classic
mechanistic-interpretability "induction head" analysis on GPT-2 small,
entirely on CPU, using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

It walks through the same pipeline used in Neel Nanda's TransformerLens demo
and in the Anthropic "In-context Learning and Induction Heads" paper, but as
a runnable `.py` script instead of a notebook:

1. Build a repeated-random-token sequence (`[BOS] A B C ... A B C ...`)
2. Run GPT-2 small and cache all internal activations
3. Plot per-token loss to see the model "get better" once the pattern repeats
4. Compute an **induction score** per attention head (does it attend from the
   current token back to the token that followed its *previous* occurrence?)
5. Compute a **previous-token score** per head (the upstream half of the circuit)
6. **Ablate** the top candidate heads and confirm loss gets much worse on the
   repeated half (causal evidence, not just correlation)
7. Compute **direct logit attribution** per head, to see which heads push the
   correct next-token logit up the most
8. Save all plots (PNG) and a CSV summary to `outputs/`

## 1. Set up the environment

```bash
cd induction_project
python3 -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> GPT-2 small is ~500MB and runs comfortably on CPU with 2-4GB of RAM, so no
> GPU is required. If you do have a CUDA GPU available, pass `--device cuda`.

## 2. Run it

```bash
python induction_heads.py
```

Optional flags:

```bash
python induction_heads.py --seq-len 30 --device cpu --top-k 5 --seed 42
```

All console output is also human-readable — you'll see tables of the top
induction heads and previous-token heads printed directly to the terminal.

## 3. Outputs

Everything is written to `outputs/` (created automatically):

| File | What it shows |
|---|---|
| `per_token_loss.png` | Loss per position — should drop sharply once the repeat starts |
| `induction_scores.png` | Heatmap of induction score per (layer, head) |
| `prev_token_scores.png` | Heatmap of previous-token score per (layer, head) |
| `top_head_attention_pattern.png` | Attention pattern of the single strongest induction head |
| `ablation_effect.png` | Bar chart: loss increase on repeated tokens after zero-ablating each top head |
| `direct_logit_attribution.png` | Bar chart: per-head contribution to the correct-token logit |
| `induction_heads_summary.csv` | Table of every head's induction score, prev-token score, and DLA, sorted |

## 4. Notes on the TransformerLens API used

This script uses the **current (v3+) `TransformerBridge` API** shown in the
TransformerLens README, not the deprecated `HookedTransformer.from_pretrained`:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
model.enable_compatibility_mode()  # folds LayerNorm / centers weights, matching
                                    # the numerics that most mech-interp tutorials
                                    # (and this script's hook names) assume
```

`enable_compatibility_mode()` matters here because most published mech-interp
walkthroughs (including the induction-head literature) assume the "folded
LayerNorm" numerics of the legacy `HookedTransformer`. Without it, logits will
still be correct (raw HF numerics), but intermediate activation values used
for direct logit attribution won't line up with tutorial expectations.

## 5. Where to go next

- Swap `"gpt2"` for `"gpt2-medium"` / `"gpt2-large"` to see how the induction
  circuit changes with scale (more RAM needed, see main chat for a table).
- Try real repeated text instead of random tokens (e.g., paste a paragraph
  twice) to see induction heads fire on natural language.
- Extend `direct_logit_attribution` into **activation patching** (swap a
  clean run's activations into a corrupted run) for causal circuit tracing.
