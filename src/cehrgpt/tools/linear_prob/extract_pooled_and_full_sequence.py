"""Single-pass feature extraction: writes BOTH pooled 2H/1H-source features
AND full per-token hidden_states[-1] sequences per patient, from the SAME
forward pass -- guaranteeing 1H/2H and the cross-attention classifier's raw
inputs share identical packed-row compositions.

Corrected per Claude Code review against the real compute_cehrgpt_features.py:
- (fix) dataset loading via generate_prepared_ds_path, not a raw config path
- (fix) global/local fallback matches real script's duplicate-pooling behavior
  when MOTOR is disabled (produces [hidden[-1] ; hidden[-1]], not H-only)
- (fix) combine_global_local_features flag now actually gated on
- (fix) drop_last reads training_args.dataloader_drop_last
- (fix) output layout matches real script: features_with_label/{split}_features/{uuid}.parquet,
  written incrementally per batch, not accumulated in memory
- (minor) output_attentions=False, num_workers/pin_memory added
"""
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset_collator import SamplePackingCehrGptDataCollator
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.tools.linear_prob.compute_cehrgpt_features import get_torch_dtype

LOG = logging.get_logger("transformers")
H = 768


def extract_pooled_and_segments(hidden_state, attention_mask):
    """
    hidden_state: (1, packed_seq_len, H)
    attention_mask: (1, packed_seq_len)
    Returns: pooled list of (H,) tensors (last real token per patient),
             segments list of (seg_len, H) tensors (full per-patient slice).
    """
    mask = attention_mask[0]
    nonzero = mask.nonzero(as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return [], []
    max_index = nonzero[-1].item()
    padded_mask = f.pad(mask[: max_index + 1], (0, 1))
    zero_positions = torch.nonzero(padded_mask == 0).flatten().tolist()

    pooled, segments = [], []
    start = 0
    for zero_pos in zero_positions:
        if zero_pos > start:
            segment = hidden_state[0, start:zero_pos, :]
            segments.append(segment)
            pooled.append(segment[-1])
        start = zero_pos + 1
    return pooled, segments


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()

    if not cehrgpt_args.sample_packing:
        raise RuntimeError("This script assumes sample_packing=True to match the 1H/2H generation setup.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = get_torch_dtype(model_args.torch_dtype)

    tokenizer = CehrGptTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )
    LOG.info("attn_implementation in use: %s", model.config._attn_implementation)
    LOG.info("include_motor_time_to_event: %s", getattr(model.config, "include_motor_time_to_event", None))

    # FIX (1): resolve the actual prepared-dataset cache path, don't load
    # dataset_prepared_path directly.
    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)
    if not any(prepared_ds_path.glob("*")):
        raise RuntimeError(
            f"No prepared dataset at {prepared_ds_path}. Run the real "
            "compute_cehrgpt_features.py once first with this exact config "
            "so the prepared dataset cache exists, then rerun this script."
        )
    processed_dataset = load_from_disk(str(prepared_ds_path))

    data_collator_fn = partial(
        SamplePackingCehrGptDataCollator,
        cehrgpt_args.max_tokens_per_batch,
        model.config.max_position_embeddings,
    )
    data_collator = data_collator_fn(
        tokenizer=tokenizer,
        max_length=(
            cehrgpt_args.max_tokens_per_batch
            if cehrgpt_args.sample_packing
            else model_args.max_position_embeddings
        ),
        include_values=model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=cehrgpt_args.add_random_token,
    )

    output_root = Path(training_args.output_dir)
    pooled_root = output_root / "features_with_label"
    full_seq_root = output_root / "full_sequence_features"

    for split in ["train", "test"]:
        dataset = processed_dataset[split]
        # FIX: drop_last from real training_args, not hardcoded
        batch_sampler = SamplePackingBatchSampler(
            lengths=dataset["num_of_concepts"],
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=data_collator,
            batch_sampler=batch_sampler,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        split_pooled_dir = pooled_root / f"{split}_features"
        split_seq_dir = full_seq_root / split
        split_pooled_dir.mkdir(parents=True, exist_ok=True)
        split_seq_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting ({split})"):
                subject_ids = batch.pop("person_id").cpu().numpy().astype(int).squeeze(0).tolist()
                labels = batch.pop("classifier_label").cpu().numpy().astype(bool).squeeze(0).tolist()
                for key in ("index_date", "age_at_index", "epoch_times", "ages"):
                    batch.pop(key, None)

                attention_mask = batch["attention_mask"]
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, output_attentions=False, output_hidden_states=True)

                local_hidden = out.hidden_states[-1].cpu().float()

                # FIX (A): replicate the real script's fallback exactly --
                # when MOTOR is disabled, "global" becomes hidden_states[-1]
                # too (duplicate pooling), not an H-only vector.
                global_hidden = getattr(out, "linear_prob_hidden_states", None)
                if global_hidden is None:
                    global_hidden = out.hidden_states[-1]
                global_hidden = global_hidden.cpu().float()

                local_pooled, segments = extract_pooled_and_segments(local_hidden, attention_mask)
                global_pooled, _ = extract_pooled_and_segments(global_hidden, attention_mask)

                if not (len(local_pooled) == len(global_pooled) == len(subject_ids) == len(labels)):
                    raise RuntimeError(
                        f"Segment/metadata count mismatch: {len(local_pooled)} local, "
                        f"{len(global_pooled)} global, {len(subject_ids)} subject_ids, "
                        f"{len(labels)} labels."
                    )

                pooled_rows = []
                for subject_id, label, seg, local_vec, global_vec in zip(
                    subject_ids, labels, segments, local_pooled, global_pooled
                ):
                    # FIX (B): gate concatenation on the real flag
                    if cehrgpt_args.combine_global_local_features:
                        feat = np.concatenate([global_vec.numpy(), local_vec.numpy()])
                    else:
                        feat = local_vec.numpy()

                    pooled_rows.append({
                        "subject_id": subject_id,
                        "boolean_value": label,
                        "features": feat,
                    })
                    # full per-token sequence for Stage B / cross-attention
                    np.save(split_seq_dir / f"{subject_id}.npy", seg.numpy())

                # FIX (C): write incrementally per batch, matching real script's
                # durability -- one parquet per batch, not accumulated in memory.
                pd.DataFrame(pooled_rows).to_parquet(
                    split_pooled_dir / f"{uuid.uuid4()}.parquet", index=False
                )

        LOG.info("Finished split '%s' -- pooled files in %s, full sequences in %s",
                  split, split_pooled_dir, split_seq_dir)


if __name__ == "__main__":
    main()