"""
induction_heads.py
===================

Local (no-notebook) reproduction of the classic mechanistic-interpretability
"induction head" analysis on GPT-2 small, using TransformerLens.

Run with:
    python induction_heads.py [--device cpu] [--seq-len 25] [--top-k 5] [--seed 0]

See README.md for background and setup instructions.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # no display needed, we just save PNGs
import matplotlib.pyplot as plt

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens import utils as tl_utils

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------
def load_model(device: str = "cpu"):
    """Load GPT-2 small via the current TransformerBridge API and switch on
    'compatibility mode' so activations line up with the folded-LayerNorm
    numerics that most mech-interp tutorials (and this script) assume."""
    print(f"Loading GPT-2 small on device='{device}' ...")
    model = TransformerBridge.boot_transformers("gpt2", device=device)
    model.enable_compatibility_mode()

    # We need per-head attention results (not just the summed output) for
    # direct logit attribution later on.
    try:
        model.set_use_attn_result(True)
    except AttributeError:
        # Fallback for versions where this is a plain config flag.
        model.cfg.use_attn_result = True

    model.eval()
    torch.set_grad_enabled(False)  # pure inference, saves memory/time
    print(
        f"Model loaded: {model.cfg.n_layers} layers x {model.cfg.n_heads} heads, "
        f"d_model={model.cfg.d_model}, d_vocab={model.cfg.d_vocab}"
    )
    return model


# ---------------------------------------------------------------------------
# 2. Repeated-token dataset (the classic induction-head probe)
# ---------------------------------------------------------------------------
def generate_repeated_tokens(model, seq_len: int, batch: int = 4, seed: int = 0):
    """Builds sequences of the form [BOS, t1, t2, ..., tN, t1, t2, ..., tN].

    If the model has learned induction, once it sees the *second* copy of a
    token it should predict the token that followed that same token the
    first time around -- causing a big drop in loss partway through the
    sequence.
    """
    torch.manual_seed(seed)
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        # GPT-2's tokenizer sometimes exposes this as eos_token_id instead.
        bos_id = model.tokenizer.eos_token_id

    # Avoid the very first ~100 token ids (mostly special/rare tokens) for
    # cleaner random sequences.
    half = torch.randint(100, model.cfg.d_vocab, (batch, seq_len))
    bos_col = torch.full((batch, 1), bos_id, dtype=torch.long)
    rep_tokens = torch.cat([bos_col, half, half], dim=1)
    return rep_tokens


def per_token_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Log-probability the model assigned to the *actual* next token at
    every position. Shape: [batch, seq_len - 1]."""
    log_probs = logits.log_softmax(dim=-1)
    correct = log_probs[:, :-1].gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return correct


# ---------------------------------------------------------------------------
# 3. Induction score / previous-token score per head
# ---------------------------------------------------------------------------
def compute_head_scores(cache, n_layers: int, n_heads: int, seq_len: int):
    """For every head, compute:
      - induction_score: mean attention paid from a destination token back to
        the token that *followed* the same token on its previous occurrence
        (offset = -(seq_len - 1) diagonal of the attention pattern, restricted
        to the second half of the sequence).
      - prev_token_score: mean attention paid to the immediately preceding
        token (offset = -1 diagonal), the signature of an upstream
        "previous token head" that feeds an induction head.
    Both are averaged over batch and the valid destination positions.
    Returns two [n_layers, n_heads] tensors.
    """
    induction_scores = torch.zeros(n_layers, n_heads)
    prev_token_scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        # pattern shape: [batch, n_heads, dest_pos, src_pos]
        pattern = cache["pattern", layer]

        # Induction stripe: dest attends to src = dest - (seq_len - 1)
        # i.e. "the token right after where I last appeared".
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=-(seq_len - 1))
        induction_scores[layer] = induction_stripe.mean(dim=(0, -1))

        # Previous-token stripe: dest attends to src = dest - 1
        prev_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=-1)
        prev_token_scores[layer] = prev_stripe.mean(dim=(0, -1))

    return induction_scores, prev_token_scores


# ---------------------------------------------------------------------------
# 4. Ablation: causal check that a head actually matters
# ---------------------------------------------------------------------------
def ablate_head_and_measure_loss(model, tokens, layer: int, head: int, seq_len: int):
    """Zero-ablates a single attention head's output (hook_z) and measures
    the mean loss on the *second* (repeated) half of the sequence, where a
    working induction head should be doing most of the work."""

    def zero_ablate_hook(z, hook):
        z[:, :, head, :] = 0.0
        return z

    hook_name = tl_utils.get_act_name("z", layer)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_ablate_hook)])
    log_probs = per_token_log_probs(logits, tokens)

    # Second half starts right after the BOS + first copy.
    second_half_start = seq_len  # index into log_probs (which is offset by 1 vs tokens)
    second_half_loss = -log_probs[:, second_half_start:].mean().item()
    return second_half_loss


def baseline_loss(model, tokens, seq_len: int):
    logits = model(tokens)
    log_probs = per_token_log_probs(logits, tokens)
    return -log_probs[:, seq_len:].mean().item()


# ---------------------------------------------------------------------------
# 5. Direct logit attribution
# ---------------------------------------------------------------------------
def direct_logit_attribution(model, cache, tokens, n_layers: int, n_heads: int):
    """For every head, estimate how much it directly pushed up the logit of
    the *correct* next token, via the standard TransformerLens recipe:
    stack every head's contribution to the residual stream, apply the final
    LayerNorm scaling factor from the real run, then project onto the
    unembedding direction of the correct next token.
    """
    # [n_layers * n_heads, batch, seq_len, d_model]
    per_head_residual, labels = cache.stack_head_results(layer=-1, return_labels=True)

    # Apply the *actual* final LayerNorm scale from this run (not a fresh one)
    # so the projection is consistent with the real logits.
    scaled_residual = cache.apply_ln_to_stack(
        per_head_residual, layer=-1, mlp_input=False
    )

    # Direction in unembedding space for each correct next token.
    correct_tokens = tokens[:, 1:]  # [batch, seq_len-1]
    unembed = model.W_U  # [d_model, d_vocab]
    # Advanced-indexing a [d_model, d_vocab] tensor with a [batch, seq_len-1]
    # index tensor gives [d_model, batch, seq_len-1] -- works for any batch size.
    correct_directions = unembed[:, correct_tokens]  # [d_model, batch, seq_len-1]

    # scaled_residual: [heads, batch, seq_len, d_model]
    # We only care about positions 0..seq_len-2 (predicting tokens[1:]).
    per_head_dla = torch.einsum(
        "h b s d, d b s -> h b s",
        scaled_residual[:, :, :-1, :],
        correct_directions,
    )
    mean_dla = per_head_dla.mean(dim=(1, 2))  # average over batch & position -> [heads]
    dla_grid = mean_dla.reshape(n_layers, n_heads)
    return dla_grid, labels


# ---------------------------------------------------------------------------
# 6. Plotting helpers (all saved as PNGs, no interactive display)
# ---------------------------------------------------------------------------
def save_heatmap(grid: torch.Tensor, title: str, filename: str, cmap: str = "Reds"):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid.numpy(), cmap=cmap, aspect="auto")
    plt.colorbar(label="score")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def save_line_plot(y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 4))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def save_bar_chart(labels, values, title, ylabel, filename):
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def save_attention_pattern(pattern_2d, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern_2d.numpy(), cmap="Blues", vmin=0, vmax=pattern_2d.max().item())
    plt.colorbar(label="attention weight")
    plt.xlabel("Source position")
    plt.ylabel("Destination position")
    plt.title(title)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seq-len", type=int, default=25, help="length of each half of the repeated sequence")
    parser.add_argument("--batch", type=int, default=4, help="number of random sequences to average over")
    parser.add_argument("--top-k", type=int, default=5, help="how many top induction heads to ablate/report")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load model -----------------------------------------------------
    model = load_model(args.device)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    # --- 2. Build repeated-token probe & run with cache ---------------------
    tokens = generate_repeated_tokens(model, args.seq_len, batch=args.batch, seed=args.seed)
    print(f"\nBuilt repeated-token sequence: shape={tuple(tokens.shape)} "
          f"([BOS] + {args.seq_len} random tokens, repeated twice)")

    print("Running forward pass with activation cache ...")
    logits, cache = model.run_with_cache(tokens)

    # --- 3. Per-token loss curve --------------------------------------------
    log_probs = per_token_log_probs(logits, tokens)
    mean_loss_per_pos = (-log_probs).mean(dim=0)  # average over batch -> [seq_len-1]
    save_line_plot(
        mean_loss_per_pos.numpy(),
        title="Per-token loss (repeated-random-token probe)\n"
              "Expect a sharp drop once the second copy begins",
        xlabel="Sequence position",
        ylabel="Loss (-log prob of correct token)",
        filename="per_token_loss.png",
    )
    first_half_loss = (-log_probs[:, : args.seq_len]).mean().item()
    second_half_loss = (-log_probs[:, args.seq_len :]).mean().item()
    print(f"\nMean loss, first half (no repeat yet):  {first_half_loss:.3f}")
    print(f"Mean loss, second half (repeat present): {second_half_loss:.3f}")
    print(f"  -> loss dropped by {first_half_loss - second_half_loss:.3f} nats once induction became possible")

    # --- 4. Induction score & previous-token score per head -----------------
    induction_scores, prev_token_scores = compute_head_scores(cache, n_layers, n_heads, args.seq_len)
    save_heatmap(induction_scores, "Induction score per head", "induction_scores.png", cmap="Reds")
    save_heatmap(prev_token_scores, "Previous-token score per head", "prev_token_scores.png", cmap="Blues")

    # --- 5. Report + attention pattern for the single strongest head --------
    flat_scores = induction_scores.flatten()
    top_k = min(args.top_k, flat_scores.numel())
    top_vals, top_idx = torch.topk(flat_scores, top_k)
    top_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in top_idx]

    print(f"\nTop {top_k} induction heads (layer, head, score):")
    for (layer, head), score in zip(top_heads, top_vals):
        print(f"  L{layer}H{head}: induction={score:.3f}, prev_token={prev_token_scores[layer, head]:.3f}")

    best_layer, best_head = top_heads[0]
    best_pattern = cache["pattern", best_layer][0, best_head]  # first batch element
    save_attention_pattern(
        best_pattern,
        f"Attention pattern: strongest induction head L{best_layer}H{best_head}",
        "top_head_attention_pattern.png",
    )

    # --- 6. Ablation: causal confirmation ------------------------------------
    print("\nRunning zero-ablation on top heads (this may take a little while on CPU) ...")
    base_loss = baseline_loss(model, tokens, args.seq_len)
    ablation_losses = []
    for layer, head in top_heads:
        loss = ablate_head_and_measure_loss(model, tokens, layer, head, args.seq_len)
        ablation_losses.append(loss)
        print(f"  Ablating L{layer}H{head}: second-half loss {base_loss:.3f} -> {loss:.3f} "
              f"(+{loss - base_loss:.3f})")

    labels = [f"L{l}H{h}" for l, h in top_heads]
    save_bar_chart(
        ["baseline"] + labels,
        [base_loss] + ablation_losses,
        "Effect of zero-ablating each top induction head\n"
        "(loss on the REPEATED half of the sequence)",
        ylabel="Loss (nats)",
        filename="ablation_effect.png",
    )

    # --- 7. Direct logit attribution -----------------------------------------
    print("\nComputing direct logit attribution per head ...")
    try:
        dla_grid, _ = direct_logit_attribution(model, cache, tokens, n_layers, n_heads)
        save_heatmap(dla_grid, "Direct logit attribution per head\n(contribution to correct next-token logit)",
                     "direct_logit_attribution.png", cmap="RdBu")
        top_dla_vals, top_dla_idx = torch.topk(dla_grid.flatten(), top_k)
        top_dla_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in top_dla_idx]
        dla_labels = [f"L{l}H{h}" for l, h in top_dla_heads]
        save_bar_chart(
            dla_labels,
            [v.item() for v in top_dla_vals],
            "Top heads by direct logit attribution",
            ylabel="Mean contribution to correct-token logit",
            filename="direct_logit_attribution_top.png",
        )
    except Exception as e:  # pragma: no cover - defensive, DLA API can vary by TL version
        print(f"  [warning] Direct logit attribution step skipped due to: {e}")
        dla_grid = torch.zeros(n_layers, n_heads)

    # --- 8. Save summary CSV --------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "induction_heads_summary.csv")
    with open(csv_path, "w") as f:
        f.write("layer,head,induction_score,prev_token_score,direct_logit_attribution\n")
        rows = []
        for layer in range(n_layers):
            for head in range(n_heads):
                rows.append((
                    layer, head,
                    induction_scores[layer, head].item(),
                    prev_token_scores[layer, head].item(),
                    dla_grid[layer, head].item(),
                ))
        rows.sort(key=lambda r: r[2], reverse=True)  # sort by induction score desc
        for layer, head, ind, prev, dla in rows:
            f.write(f"{layer},{head},{ind:.4f},{prev:.4f},{dla:.4f}\n")
    print(f"\nSaved full per-head summary to {csv_path}")

    print(f"\nAll done! Plots and CSV are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
