import torch

def freeze_all_params(model):
    """Freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_named_params(model, wanted_names):
    """
    Unfreeze any param whose name contains one of the substrings in wanted_names.
    Returns number of parameters unfrozen.
    """
    hits = 0
    for name, p in model.named_parameters():
        if any(w in name for w in wanted_names):
            p.requires_grad = True
            hits += p.numel()
    return hits


def unfreeze_last_parameterized_modules(model, last_n=1):
    """
    Fallback: unfreeze the last N modules (by DFS order) that own parameters.
    Returns list of module names and number of parameters unfrozen.
    """
    mods = []
    for mod_name, mod in model.named_modules():
        has_local_params = any(True for _ in mod.named_parameters(recurse=False))
        if has_local_params:
            mods.append((mod_name, mod))
    to_unfreeze = mods[-last_n:]
    total = 0
    for mod_name, mod in to_unfreeze:
        for _, p in mod.named_parameters(recurse=False):
            p.requires_grad = True
            total += p.numel()
    return [n for n, _ in to_unfreeze], total


def print_trainable_param_stats(model, tag=""):
    """Print ratio of trainable vs total params."""
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable / total params: {trainable:,} / {tot:,} "
          f"({100.0*trainable/tot:.4f}% trainable)")


def freeze_all_except_last(model):
    """
    Freeze all layers except the last parameterized module (or known head).
    """
    # Step 1: freeze everything
    freeze_all_params(model)

    # Step 2: try unfreezing by common head names
    likely_heads = [
        "lm_head", "score_head", "classifier", "cls", "fc", "final_layer",
        "bbox_head", "box_head", "det_head", "detector", "prediction_head",
        "language_model.lm_head", "model.lm_head",
        "ocr_head", "vl_head", "text_head",
        "vision_tower.mlp_head", "vision_head",
    ]

    hit_params = unfreeze_named_params(model, likely_heads)

    if hit_params == 0:
        last_modules, total = unfreeze_last_parameterized_modules(model, last_n=1)
        print(f"Unfroze last module(s): {last_modules} (params: {total:,})")
    else:
        print(f"Unfroze head-like params (params: {hit_params:,})")

    print_trainable_param_stats(model, tag="after-freeze")

    # Optional: for memory saving
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model
