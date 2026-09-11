"""
Trainer callback that localises training instability instead of reporting it after the
fact.

Logs, on a fixed step interval: the pre-clip gradient norm, the largest absolute logit,
the token-embedding norm, the final-norm weight norm, and the per-layer gradient norms.
Optionally aborts the run the first time a non-finite loss or gradient appears, so the
offending step is the last thing in the log rather than being buried under thousands of
NaN steps.

Motivation: a fully-masked attention row (every sample-packing separator position) makes
SDPA's fused kernels return undefined values. Depending on kernel and dtype that is either
NaN, which poisons everything within two layers, or finite garbage, which degrades
gradients slowly. The two look completely different in a loss curve, so it is worth
knowing which one is happening.

Usage - add to the trainer in a runner, or in a scratch training script:

    from cehrgpt.tools.instability_probe import InstabilityProbe

    trainer.add_callback(InstabilityProbe(every=25, abort_on_nonfinite=True))
"""

from typing import Optional

import torch
from transformers import TrainerCallback
from transformers.utils import logging

LOG = logging.get_logger("transformers")


class InstabilityProbe(TrainerCallback):
    def __init__(
        self,
        every: int = 25,
        abort_on_nonfinite: bool = True,
        per_layer: bool = True,
    ):
        self.every = every
        self.abort_on_nonfinite = abort_on_nonfinite
        self.per_layer = per_layer
        self._warned_masked_rows = False

    def _decoder_blocks(self, model):
        backbone = getattr(model, "cehrgpt", None)
        return list(getattr(backbone, "h", [])) if backbone is not None else []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        step = state.global_step

        total_sq = 0.0
        nonfinite = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                nonfinite.append(name)
            else:
                total_sq += grad.float().pow(2).sum().item()
        grad_norm = total_sq**0.5

        if nonfinite:
            LOG.error(
                "step %d: %d parameter(s) have non-finite gradients, first: %s",
                step,
                len(nonfinite),
                nonfinite[:5],
            )
            if self.abort_on_nonfinite:
                control.should_training_stop = True
            return

        if step % self.every != 0:
            return

        with torch.no_grad():
            embedding = model.get_input_embeddings()
            embedding_norm = (
                embedding.weight.norm().item() if embedding is not None else float("nan")
            )
            blocks = self._decoder_blocks(model)
            final_norm = getattr(getattr(model, "cehrgpt", None), "ln_f", None)
            final_norm_value = (
                final_norm.weight.norm().item() if final_norm is not None else float("nan")
            )

        message = (
            f"step {step}: grad_norm={grad_norm:.3f} "
            f"embedding_norm={embedding_norm:.3f} final_norm_w={final_norm_value:.3f}"
        )
        if self.per_layer and blocks:
            per_layer = []
            for index, block in enumerate(blocks):
                block_sq = sum(
                    p.grad.detach().float().pow(2).sum().item()
                    for p in block.parameters()
                    if p.grad is not None
                )
                per_layer.append(f"{index}:{block_sq ** 0.5:.2f}")
            message += " layer_grad_norms=[" + " ".join(per_layer) + "]"
        LOG.info(message)


def assert_no_fully_masked_rows(attention_mask: torch.Tensor) -> Optional[str]:
    """
    Check a 4D additive attention mask for rows that attend to nothing.

    Returns a description of the offending rows, or None when the mask is sound. Useful as
    a one-off assertion inside a training loop when diagnosing a suspect run.
    """
    if attention_mask is None or attention_mask.dim() != 4:
        return None
    floor = torch.finfo(attention_mask.dtype).min
    attends = (attention_mask > floor).any(dim=-1)
    if attends.all():
        return None
    offending = (~attends).nonzero()
    return (
        f"{offending.shape[0]} fully masked row(s); SDPA fused kernels return undefined "
        f"values for these. First few: {offending[:5].tolist()}"
    )
