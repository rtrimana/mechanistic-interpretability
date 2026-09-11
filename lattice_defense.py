"""
lattice_defense.py
===================

A small, runnable illustration of the "Lattice" prompt-injection defense
(provenance tagging + circuit localization + targeted internal intervention)
using GPT-2 small's induction circuit as a stand-in for a generic
"recognize a pattern and comply with it" mechanism.

GPT-2 small isn't instruction-tuned, so we can't literally demo it obeying
injected instructions. Instead we use the one well-understood circuit that
*does* implement "recognize X, then output whatever followed X before" --
the induction circuit from induction_heads.py -- as a toy analogue of an
"instruction-following door". This lets every step of the Lattice pipeline
(tag tokens, localize the circuit, intervene only at that circuit, verify
everything else is untouched) be demonstrated with real activations and
real numbers instead of a hand-wavy diagram.

The probe:

    [BOS] <preamble, trusted> X Y <doc filler, untrusted> X Z <trailer, trusted> X
           pos 1-3            4 5  pos 6-8              9 10  pos 11-12       13

  - "X Y" (positions 4-5) is a TRUSTED definition: "after X comes Y".
  - "X Z" (positions 9-10) sits inside UNTRUSTED content (e.g. a retrieved
    document) and tries to redefine what follows X -- a toy prompt injection.
  - The final X (position 13, TRUSTED) is the query: what does the model
    predict comes next, Y (trusted) or Z (injected)?

Four steps, mirroring the four Lattice thrusts:

  1. Circuit localization  -- induction score (cheap attribution proxy) to
     shortlist candidate heads, THEN mandatory causal ablation to verify
     each candidate actually matters (never trust attribution alone).
  2. Intervention primitive -- at the single query destination position,
     renormalize each verified head's attention pattern so it only ever
     puts weight on TRUSTED source tokens (the "badge check at the door").
  3. Provenance lattice -- here a minimal 2-level trusted/untrusted lattice,
     carried as a per-position tag vector.
  4. (Reasoning/MoE extensions from the proposal are out of scope for a
     dense, non-reasoning GPT-2 -- noted, not implemented, here.)

Run with:
    python lattice_defense.py [--device cpu] [--batch 64] [--top-k 4] [--seed 0]
"""

import argparse
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer_lens import utils as tl_utils

from induction_heads import (
    load_model,
    generate_repeated_tokens,
    compute_head_scores,
    ablate_head_and_measure_loss,
    baseline_loss,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Fixed layout of the injection probe (see module docstring) -----------
PREAMBLE_LEN = 3
UNTRUSTED_FILLER_LEN = 3
TRAILING_LEN = 2
SEQ_LEN = 1 + PREAMBLE_LEN + 2 + UNTRUSTED_FILLER_LEN + 2 + TRAILING_LEN + 1  # = 14
QUERY_POS = SEQ_LEN - 1  # 13
UNTRUSTED_POSITIONS = list(range(1 + PREAMBLE_LEN + 2, 1 + PREAMBLE_LEN + 2 + UNTRUSTED_FILLER_LEN + 2))  # 6..10


# ---------------------------------------------------------------------------
# 1. Circuit localization: cheap attribution proxy + mandatory ablation
# ---------------------------------------------------------------------------
def find_and_verify_induction_heads(model, seed, top_k, min_loss_increase=0.05, probe_seq_len=25, probe_batch=6):
    """Shortlist heads by induction score (fast, correlational), then confirm
    each one causally with zero-ablation before trusting it -- directly
    answering the Nasr et al. lesson that correlational evidence alone isn't
    enough to localize a circuit."""
    tokens = generate_repeated_tokens(model, probe_seq_len, batch=probe_batch, seed=seed)
    _, cache = model.run_with_cache(tokens)
    induction_scores, _ = compute_head_scores(cache, model.cfg.n_layers, model.cfg.n_heads, probe_seq_len)

    flat = induction_scores.flatten()
    n_candidates = min(top_k * 3, flat.numel())
    _, top_idx = torch.topk(flat, n_candidates)
    candidates = [(int(i // model.cfg.n_heads), int(i % model.cfg.n_heads)) for i in top_idx]

    base = baseline_loss(model, tokens, probe_seq_len)
    verified = []
    print(f"\nCandidate induction heads (attribution) vs. causal ablation check (baseline loss {base:.3f}):")
    for layer, head in candidates:
        loss = ablate_head_and_measure_loss(model, tokens, layer, head, probe_seq_len)
        increase = loss - base
        keep = increase >= min_loss_increase
        print(f"  L{layer}H{head}: induction_score={induction_scores[layer, head]:.3f}, "
              f"ablated_loss={loss:.3f} (+{increase:.3f}) {'[VERIFIED]' if keep else '[rejected]'}")
        if keep:
            verified.append((layer, head, induction_scores[layer, head].item(), increase))
        if len(verified) >= top_k:
            break

    worst_flat_idx = int(torch.argmin(flat))
    worst_head = (worst_flat_idx // model.cfg.n_heads, worst_flat_idx % model.cfg.n_heads)
    return verified, worst_head, induction_scores


# ---------------------------------------------------------------------------
# 2. Provenance-tagged injection probe
# ---------------------------------------------------------------------------
def build_injection_probe(model, batch, seed):
    """Builds the fixed-layout probe described in the module docstring, with
    fresh random token ids per row. Returns tokens, per-row (x, y, z) ids,
    and the position-aligned trusted/untrusted tag vector (1=trusted)."""
    rng = np.random.default_rng(seed)
    bos_id = model.tokenizer.bos_token_id or model.tokenizer.eos_token_id
    d_vocab = model.cfg.d_vocab

    rows = []
    y_ids, z_ids = [], []
    for _ in range(batch):
        ids = rng.choice(np.arange(100, d_vocab), size=11, replace=False)
        preamble, x, y, ufiller, z, trailing = ids[0:3], ids[3], ids[4], ids[5:8], ids[8], ids[9:11]
        row = [bos_id, *preamble, x, y, *ufiller, x, z, *trailing, x]
        rows.append(row)
        y_ids.append(int(y))
        z_ids.append(int(z))

    tokens = torch.tensor(rows, dtype=torch.long)
    y_ids = torch.tensor(y_ids, dtype=torch.long)
    z_ids = torch.tensor(z_ids, dtype=torch.long)

    trusted_mask = torch.ones(SEQ_LEN)
    trusted_mask[UNTRUSTED_POSITIONS] = 0.0
    return tokens, y_ids, z_ids, trusted_mask


def query_probs(logits, y_ids, z_ids):
    probs = logits[:, QUERY_POS, :].softmax(dim=-1)
    p_y = probs.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
    p_z = probs.gather(-1, z_ids.unsqueeze(-1)).squeeze(-1)
    return p_y, p_z


# ---------------------------------------------------------------------------
# 3. The intervention: badge check at the door, nowhere else
# ---------------------------------------------------------------------------
def make_lattice_hooks(heads, trusted_mask, dest_pos):
    """One hook per layer that touches ONLY `dest_pos`'s attention row for
    the given heads, renormalizing it over trusted source positions only.
    Every other destination position -- i.e. all the reading/summarizing of
    the untrusted content itself -- is left byte-for-byte unmodified."""
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    hooks = []
    for layer, heads_in_layer in by_layer.items():
        def hook(pattern, hook, heads_in_layer=heads_in_layer):
            for head in heads_in_layer:
                row = pattern[:, head, dest_pos, :] * trusted_mask
                pattern[:, head, dest_pos, :] = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return pattern
        hooks.append((tl_utils.get_act_name("pattern", layer), hook))
    return hooks


# ---------------------------------------------------------------------------
# 4. Plotting helpers
# ---------------------------------------------------------------------------
def save_grouped_bars(labels, series, title, ylabel, filename):
    plt.figure(figsize=(7, 4.5))
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(series.items()):
        plt.bar(x + i * width, values, width=width, label=name)
    plt.xticks(x + width * (n - 1) / 2, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def save_attention_redirect(before, after, head_label, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, data, title in zip(axes, [before, after], ["baseline", "with Lattice intervention"]):
        ax.bar(range(SEQ_LEN), data.numpy())
        for p in UNTRUSTED_POSITIONS:
            ax.axvspan(p - 0.5, p + 0.5, color="red", alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("source position")
    axes[0].set_ylabel("attention weight from query position")
    fig.suptitle(f"{head_label}: attention from query token, before/after intervention\n"
                 "(red band = untrusted source positions)")
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64, help="number of injection-probe trials to average over")
    parser.add_argument("--top-k", type=int, default=4, help="how many verified induction heads to use as 'the door'")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("THRUST 1: circuit localization (attribution shortlist + mandatory ablation)")
    print("=" * 78)
    model = load_model(args.device)
    verified, worst_head, induction_scores = find_and_verify_induction_heads(model, args.seed, args.top_k)
    if not verified:
        raise RuntimeError("No induction heads survived causal verification -- try a different seed.")
    heads = [(l, h) for l, h, _, _ in verified]
    print(f"\nUsing {len(heads)} verified heads as 'the door': {[f'L{l}H{h}' for l, h in heads]}")
    print(f"Deliberately-wrong control head (lowest induction score): L{worst_head[0]}H{worst_head[1]}")

    print("\n" + "=" * 78)
    print("THRUST 3: provenance lattice (2-level trusted/untrusted tag per position)")
    print("=" * 78)
    tokens, y_ids, z_ids, trusted_mask = build_injection_probe(model, args.batch, args.seed)
    print(f"Built {args.batch} injection-probe trials, sequence layout (position: tag):")
    tags = ["untrusted" if trusted_mask[p] == 0 else "trusted" for p in range(SEQ_LEN)]
    print("  " + ", ".join(f"{p}:{t}" for p, t in enumerate(tags)))
    print("  position 4/9/13 all hold the *same* token X (defined trusted, re-asserted untrusted, then queried).")

    print("\n" + "=" * 78)
    print("THRUST 2: intervention -- badge check only at the query position")
    print("=" * 78)
    logits_base, cache_base = model.run_with_cache(tokens)
    p_y_base, p_z_base = query_probs(logits_base, y_ids, z_ids)

    lattice_hooks = make_lattice_hooks(heads, trusted_mask, QUERY_POS)
    logits_lattice = model.run_with_hooks(tokens, fwd_hooks=lattice_hooks)
    p_y_lat, p_z_lat = query_probs(logits_lattice, y_ids, z_ids)

    wrong_hooks = make_lattice_hooks([worst_head], trusted_mask, QUERY_POS)
    logits_wrong = model.run_with_hooks(tokens, fwd_hooks=wrong_hooks)
    p_y_wrong, p_z_wrong = query_probs(logits_wrong, y_ids, z_ids)

    def compliance_share(p_y, p_z):
        return (p_z / (p_y + p_z).clamp_min(1e-9)).mean().item()

    share_base = compliance_share(p_y_base, p_z_base)
    share_lat = compliance_share(p_y_lat, p_z_lat)
    share_wrong = compliance_share(p_y_wrong, p_z_wrong)

    print(f"\nMean P(Y) / P(Z) at the query position, and the injected token's 'compliance share' P(Z)/(P(Y)+P(Z)):")
    print(f"  baseline (no defense):        P(Y)={p_y_base.mean():.4f}  P(Z)={p_z_base.mean():.4f}  share={share_base:.1%}")
    print(f"  Lattice (verified heads):     P(Y)={p_y_lat.mean():.4f}  P(Z)={p_z_lat.mean():.4f}  share={share_lat:.1%}")
    print(f"  wrong-head control:           P(Y)={p_y_wrong.mean():.4f}  P(Z)={p_z_wrong.mean():.4f}  share={share_wrong:.1%}")
    print("  -> the guard only works when placed at the actual circuit, not just any head.")

    # --- Capability preservation: everything before the query position is untouched ---
    diff = (logits_base[:, :QUERY_POS, :] - logits_lattice[:, :QUERY_POS, :]).abs().max().item()
    print(f"\nCapability check: max logit difference at every OTHER position (reading/summarizing the "
          f"untrusted content itself) = {diff:.2e} -- i.e. exactly unchanged, as expected, since the hook "
          f"only touches the query position's attention row.")

    # --- Legitimate-use control: no untrusted content -> hook is a no-op ---
    trusted_only_tokens = generate_repeated_tokens(model, seq_len=25, batch=6, seed=args.seed)
    legit_dest_pos = trusted_only_tokens.shape[1] - 1
    all_trusted_mask = torch.ones(trusted_only_tokens.shape[1])
    legit_hooks = make_lattice_hooks(heads, all_trusted_mask, dest_pos=legit_dest_pos)
    base_legit = baseline_loss(model, trusted_only_tokens, 25)
    logits_legit = model.run_with_hooks(trusted_only_tokens, fwd_hooks=legit_hooks)
    logits_plain = model(trusted_only_tokens)
    print(f"\nLegit-use control: with no untrusted content anywhere, the same intervention is a mathematical "
          f"no-op (nothing to renormalize away) -- baseline second-half loss {base_legit:.3f}, and logits at the "
          f"hooked position unchanged (max diff "
          f"{(logits_legit[:, legit_dest_pos, :] - logits_plain[:, legit_dest_pos, :]).abs().max().item():.2e}).")

    # --- Attention redirect plot for the single strongest verified head -----
    best_layer, best_head = heads[0]
    attn_before = cache_base["pattern", best_layer][:, best_head, QUERY_POS, :].mean(dim=0)
    with model.hooks(fwd_hooks=lattice_hooks):
        _, cache_lattice = model.run_with_cache(tokens)
    attn_after = cache_lattice["pattern", best_layer][:, best_head, QUERY_POS, :].mean(dim=0)
    save_attention_redirect(attn_before, attn_after, f"L{best_layer}H{best_head}", "lattice_attention_redirect.png")

    # --- Summary plots --------------------------------------------------------
    save_grouped_bars(
        ["P(Y) trusted", "P(Z) injected"],
        {
            "baseline": [p_y_base.mean().item(), p_z_base.mean().item()],
            "Lattice": [p_y_lat.mean().item(), p_z_lat.mean().item()],
            "wrong-head control": [p_y_wrong.mean().item(), p_z_wrong.mean().item()],
        },
        "Query-position probability mass on trusted vs. injected completion",
        ylabel="mean probability",
        filename="lattice_probabilities.png",
    )
    save_grouped_bars(
        ["compliance share"],
        {
            "baseline": [share_base],
            "Lattice": [share_lat],
            "wrong-head control": [share_wrong],
        },
        "Injected token's share of (P(Y)+P(Z)) at the query position\nlower is better",
        ylabel="P(Z) / (P(Y) + P(Z))",
        filename="lattice_compliance_share.png",
    )

    # --- CSV summary ------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "lattice_defense_summary.csv")
    with open(csv_path, "w") as f:
        f.write("trial,p_y_base,p_z_base,p_y_lattice,p_z_lattice,p_y_wrong,p_z_wrong\n")
        for i in range(args.batch):
            f.write(f"{i},{p_y_base[i]:.6f},{p_z_base[i]:.6f},"
                    f"{p_y_lat[i]:.6f},{p_z_lat[i]:.6f},"
                    f"{p_y_wrong[i]:.6f},{p_z_wrong[i]:.6f}\n")
    print(f"\nSaved per-trial results to {csv_path}")
    print(f"\nAll done! Plots and CSV are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
