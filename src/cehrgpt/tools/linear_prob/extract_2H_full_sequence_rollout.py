"""Like extract_pooled_and_full_sequence.py, but forces attn_implementation="eager"
and additionally saves per-layer attention weights per patient (averaged over
heads) for attention rollout, to full_sequence_attentions/{split}/{subject_id}.npy.

COLLATOR CHOICE: non-packing (CehrGptDataCollator, batch_size=1), not
SamplePackingCehrGptDataCollator. Traced this session: eager + sample packing
IS mask-correct (create_sample_packing_attention_mask builds a genuine
block-diagonal mask combined with the causal `self.bias` buffer -- no
cross-patient leakage). The blocker is memory, not correctness:
output_attentions=True materializes the full (batch, num_heads, seq_len,
seq_len) tensor per layer, and under packing seq_len is the WHOLE packed row
(can be >> max_position_embeddings), not one patient's length. That's
quadratic in packed-row length across every layer simultaneously -- easily
hundreds of GB. Non-packing bounds this to one patient's own truncated
length. batch_size is forced to 1 here specifically to keep that bound at a
single patient at a time rather than per_device_eval_batch_size patients.

Everything else (tokenization/truncation via the cached prepared dataset,
value encoding, combine_global_local_features fallback/gating) is identical
to extract_pooled_and_full_sequence.py.

--capture_attentions (default off): when omitted, output_attentions is never
requested and full_sequence_attentions/ is never written -- only pooled
features and full_sequence_features/ get produced, same cost as before this
flag existed. Pass --capture_attentions to additionally capture and save
per-layer attention weights, at the extra memory/time cost described above.
Intended usage: a fast pass over the full cohort without this flag, and a
separate, restricted pass (via --person_ids_file) with this flag on for just
the patients that actually need rollout analysis.
"""

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.tools.linear_prob.compute_cehrgpt_features import get_torch_dtype

LOG = logging.get_logger("transformers")


def _pop_person_ids_file_arg():
    """Strip --person_ids_file[=value] out of sys.argv before parse_runner_args()
    sees it. It isn't a field on CehrGPTArguments/DataTrainingArguments/
    ModelArguments/TrainingArguments, so HfArgumentParser would otherwise
    error on it as an unrecognized argument (parse_runner_args has no
    return_remaining_strings=True escape hatch -- see gpt_runner_util.py)."""
    argv = sys.argv
    person_ids_file = None
    remaining = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--person_ids_file":
            if i + 1 >= len(argv):
                raise RuntimeError("--person_ids_file requires a value")
            person_ids_file = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--person_ids_file="):
            person_ids_file = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    sys.argv = remaining
    return person_ids_file


def _pop_flag_arg(flag_name: str) -> bool:
    """Strip a bare boolean flag (e.g. --capture_attentions) out of sys.argv
    before parse_runner_args() sees it, for the same reason as
    _pop_person_ids_file_arg -- it isn't a field on any of the dataclasses
    parse_runner_args() knows about."""
    found = flag_name in sys.argv
    sys.argv = [a for a in sys.argv if a != flag_name]
    return found


def load_person_id_filter(person_ids_file: str) -> set:
    path = Path(person_ids_file)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(
            f"Unsupported --person_ids_file extension: {path.suffix} (expected .parquet or .csv)"
        )
    if "person_id" in df.columns:
        id_column = "person_id"
    elif "subject_id" in df.columns:
        id_column = "subject_id"
    else:
        raise RuntimeError(
            f"{person_ids_file} must have a 'person_id' or 'subject_id' column; "
            f"found: {list(df.columns)}"
        )
    return set(df[id_column].astype(int).tolist())


def main():
    person_ids_file = _pop_person_ids_file_arg()
    capture_attentions = _pop_flag_arg("--capture_attentions")
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()

    LOG.info("capture_attentions: %s", capture_attentions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = get_torch_dtype(model_args.torch_dtype)

    tokenizer = CehrGptTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    # Force eager attention: build the config explicitly and mutate
    # use_local_attention=False BEFORE model construction. GPT2Block.__init__
    # picks GPT2FlashAttention vs GPT2AttentionRoPE at construction time
    # (gpt2.py:594-601) based on config._attn_implementation and
    # config.use_local_attention -- setting either on model.config AFTER
    # from_pretrained() would be too late, the attention submodules are
    # already instantiated by then.
    config = CEHRGPTConfig.from_pretrained(model_args.model_name_or_path)
    config.use_local_attention = False

    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )

    # Defensive check: confirm the actually-instantiated attention module is
    # the eager class, not just that the config flags look right.
    actual_attn_class = type(model.cehrgpt.h[0].attn).__name__
    if actual_attn_class != "GPT2AttentionRoPE":
        raise RuntimeError(
            f"Expected eager attention (GPT2AttentionRoPE) but got "
            f"{actual_attn_class} -- attn_implementation/use_local_attention "
            "did not take effect as expected."
        )
    LOG.info("Confirmed attention implementation: %s", actual_attn_class)
    LOG.info("include_motor_time_to_event: %s", getattr(model.config, "include_motor_time_to_event", None))

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)
    if not any(prepared_ds_path.glob("*")):
        raise RuntimeError(
            f"No prepared dataset at {prepared_ds_path}. Run the real "
            "compute_cehrgpt_features.py once first with this exact config "
            "so the prepared dataset cache exists, then rerun this script."
        )
    processed_dataset = load_from_disk(str(prepared_ds_path))

    # --person_ids_file filter -- applied here, right after loading, before
    # the per-split loop, to the whole DatasetDict (both "train" and "test").
    # Because this is one filter applied uniformly to both splits: for a
    # correct train subsample AND an unfiltered test set in the same run,
    # pass a person_ids_file containing the UNION of (case-control train
    # ids) + (all test ids) -- not just the train subsample alone, or the
    # test split would be wrongly filtered down to nothing.
    if person_ids_file is not None:
        allowed_ids = load_person_id_filter(person_ids_file)
        LOG.info("Filtering dataset to %d person_ids from %s", len(allowed_ids), person_ids_file)
        processed_dataset = processed_dataset.filter(
            lambda batch: [pid in allowed_ids for pid in batch["person_id"]],
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            num_proc=data_args.preprocessing_num_workers,
        )
        LOG.info(
            "After filtering: train=%d, test=%d",
            len(processed_dataset["train"]) if "train" in processed_dataset else -1,
            len(processed_dataset["test"]) if "test" in processed_dataset else -1,
        )

    data_collator = CehrGptDataCollator(
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
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
    full_attn_root = output_root / "full_sequence_attentions"

    for split in ["train", "test"]:
        if split not in processed_dataset:
            LOG.warning("Split '%s' not present in the prepared dataset -- skipping", split)
            continue
        dataset = processed_dataset[split]

        # batch_size forced to 1: keeps the (num_heads, seq_len, seq_len)
        # attention tensor bounded to one patient's own truncated length,
        # not per_device_eval_batch_size patients padded together.
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        split_pooled_dir = pooled_root / f"{split}_features"
        split_seq_dir = full_seq_root / split
        split_pooled_dir.mkdir(parents=True, exist_ok=True)
        split_seq_dir.mkdir(parents=True, exist_ok=True)
        if capture_attentions:
            split_attn_dir = full_attn_root / split
            split_attn_dir.mkdir(parents=True, exist_ok=True)
        else:
            split_attn_dir = None

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting ({split})"):
                subject_ids = batch.pop("person_id").cpu().numpy().astype(int).squeeze(0).tolist()
                labels = batch.pop("classifier_label").cpu().numpy().astype(bool).squeeze(0).tolist()
                for key in ("index_date", "age_at_index", "epoch_times", "ages"):
                    batch.pop(key, None)

                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, output_attentions=capture_attentions, output_hidden_states=True)

                # batch_size=1, no padding introduced (nothing else in the
                # batch to pad against) -- the whole row is this one
                # patient's real, post-truncation sequence.
                local_hidden = out.hidden_states[-1][0].cpu().float()  # (seq_len, H)

                global_hidden = getattr(out, "linear_prob_hidden_states", None)
                if global_hidden is None:
                    global_hidden = out.hidden_states[-1]
                global_hidden = global_hidden[0].cpu().float()  # (seq_len, H)

                local_vec = local_hidden[-1]
                global_vec = global_hidden[-1]

                if cehrgpt_args.combine_global_local_features:
                    feat = np.concatenate([global_vec.numpy(), local_vec.numpy()])
                else:
                    feat = local_vec.numpy()

                subject_id = subject_ids[0]
                label = labels[0]

                pd.DataFrame([{
                    "subject_id": subject_id,
                    "boolean_value": label,
                    "features": feat,
                }]).to_parquet(split_pooled_dir / f"{uuid.uuid4()}.parquet", index=False)

                np.save(split_seq_dir / f"{subject_id}.npy", local_hidden.numpy())

                if capture_attentions:
                    # (num_layers, seq_len, seq_len): average over heads per
                    # layer, then stack layers. No slicing needed for the same
                    # reason as above -- the whole (seq_len, seq_len) matrix is
                    # this one patient's own attention, nothing to trim.
                    layer_matrices = [
                        layer_attn[0].mean(dim=0).cpu().float().numpy()  # (num_heads, S, S) -> (S, S)
                        for layer_attn in out.attentions
                    ]
                    attn_stack = np.stack(layer_matrices, axis=0)  # (num_layers, S, S)
                    np.save(split_attn_dir / f"{subject_id}.npy", attn_stack)

        LOG.info(
            "Finished split '%s' -- pooled: %s, full sequences: %s, attentions: %s",
            split, split_pooled_dir, split_seq_dir,
            split_attn_dir if capture_attentions else "skipped (--capture_attentions not set)",
        )


if __name__ == "__main__":
    main()
