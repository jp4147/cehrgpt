"""Standalone sanity check: reproduce the tokenization + truncation pipeline
used by extract_pooled_and_full_sequence.py directly on raw patient records
(from the gpt_seq_3year raw parquet files, BEFORE any prepared-dataset
caching), and confirm decoded input_ids line up position-for-position with
the (equally truncated) concept_ids.

Pipeline reproduced, in order:
    HFCehrGptTokenizationMapping.transform(record)
        -> CehrGptDataProcessor.slice_out_input_sequence(record)

Confirmed import paths (checked directly against the repo, not assumed):
    - HFCehrGptTokenizationMapping: cehrgpt.data.hf_cehrgpt_dataset_mapping
      (src/cehrgpt/data/hf_cehrgpt_dataset_mapping.py:414)
    - CehrGptDataProcessor:         cehrgpt.data.cehrgpt_data_processor
      (src/cehrgpt/data/cehrgpt_data_processor.py:30)
    - CehrGptTokenizer:             cehrgpt.models.tokenization_hf_cehrgpt
      (re-export shim used by every other script in this tool family; the
      real implementation lives in cehrgpt.tokenization.tokenization_cehrgpt)

Confirmed CehrGptDataProcessor.__init__ signature
(src/cehrgpt/data/cehrgpt_data_processor.py:31-44):
    tokenizer, max_length, shuffle_records=False, include_values=False,
    include_ttv_prediction=False, include_motor_time_to_event=False,
    motor_sampling_probability=0.5, is_data_in_meds=False, pretraining=True,
    include_demographics=False, add_linear_prob_token=False
-- max_length, pretraining, add_linear_prob_token, include_demographics,
include_values are all real, correctly-named keyword arguments.

Note: this script loads only the model CONFIG (CEHRGPTConfig.from_pretrained),
not the full model weights -- max_position_embeddings comes from the same
config.json a fully loaded model would expose, but this avoids the GPU/weight
loading overhead for what is purely a tokenization/truncation check.
"""

import argparse
from pathlib import Path

import polars as pl

from cehrgpt.data.cehrgpt_data_processor import CehrGptDataProcessor
from cehrgpt.data.hf_cehrgpt_dataset_mapping import HFCehrGptTokenizationMapping
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

# Columns HFCehrGptTokenizationMapping.transform() can read from a raw record
# (see the earlier line-by-line trace of that method): concept_ids and
# concept_value_masks are always required; ages/epoch_times are optional
# (reconstructed if absent); number_as_values+concept_as_values+is_numeric_types
# are required together, else concept_values is required as a fallback source;
# units is only read if any concept_value_mask is set.
RAW_COLUMNS = [
    "concept_ids",
    "concept_value_masks",
    "units",
    "number_as_values",
    "concept_as_values",
    "is_numeric_types",
    "concept_values",
    "ages",
    "epoch_times",
]


def load_raw_record(raw_data_dir: Path, subject_id: int, id_column: str) -> dict:
    lf = pl.scan_parquet(str(raw_data_dir / "**" / "*.parquet"))
    schema_columns = set(lf.collect_schema().names())
    if id_column not in schema_columns:
        raise RuntimeError(
            f"'{id_column}' not found under {raw_data_dir}. "
            f"Columns present: {sorted(schema_columns)}"
        )

    rows = lf.filter(pl.col(id_column) == subject_id).collect()
    if rows.is_empty():
        raise RuntimeError(f"No row found for {id_column}={subject_id} under {raw_data_dir}")
    if len(rows) > 1:
        raise RuntimeError(
            f"{len(rows)} rows found for {id_column}={subject_id}; expected exactly "
            "one raw record per patient -- inspect the raw data before proceeding"
        )

    row_dict = rows.to_dicts()[0]
    record = {}
    for col in RAW_COLUMNS:
        if col in row_dict and row_dict[col] is not None:
            # Force plain Python lists -- CehrGptDataProcessor concatenates
            # concept_ids with "+", which silently breaks (numpy broadcast
            # error) if this is a numpy array instead of a list.
            record[col] = list(row_dict[col])

    if "concept_ids" not in record:
        raise RuntimeError(f"Raw record for {id_column}={subject_id} is missing 'concept_ids'")
    if "concept_value_masks" not in record:
        raise RuntimeError(f"Raw record for {id_column}={subject_id} is missing 'concept_value_masks'")
    have_number_concept = "number_as_values" in record and "concept_as_values" in record
    if not have_number_concept and "concept_values" not in record:
        raise RuntimeError(
            f"Raw record for {id_column}={subject_id} has neither "
            "(number_as_values + concept_as_values) nor a concept_values "
            "fallback column -- HFCehrGptTokenizationMapping.transform will KeyError"
        )

    record["person_id"] = subject_id
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce HFCehrGptTokenizationMapping.transform + "
        "CehrGptDataProcessor.slice_out_input_sequence on raw gpt_seq_3year "
        "records and verify decoded input_ids match concept_ids position-for-position."
    )
    parser.add_argument("--raw_data_dir", required=True, help="Directory of gpt_seq_3year raw parquet files")
    parser.add_argument("--tokenizer_path", required=True, help="Path/name for CehrGptTokenizer.from_pretrained")
    parser.add_argument(
        "--model_path", required=True,
        help="Path/name for CEHRGPTConfig.from_pretrained (only the config.json is read, no weights loaded)",
    )
    parser.add_argument("--subject_ids", required=True, nargs="+", type=int, help="2-3 subject_ids to inspect")
    parser.add_argument("--id_column", default="person_id", help="Raw parquet column identifying the patient")
    args = parser.parse_args()

    tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_path)
    config = CEHRGPTConfig.from_pretrained(args.model_path)

    tokenization_mapping = HFCehrGptTokenizationMapping(concept_tokenizer=tokenizer)
    data_processor = CehrGptDataProcessor(
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        include_values=True,
        pretraining=False,
        include_demographics=False,
        add_linear_prob_token=False,
    )

    raw_data_dir = Path(args.raw_data_dir)
    for subject_id in args.subject_ids:
        print(f"\n{'=' * 80}\nsubject_id={subject_id}\n{'=' * 80}")

        record = load_raw_record(raw_data_dir, subject_id, args.id_column)
        raw_length = len(record["concept_ids"])

        record = tokenization_mapping.transform(record)
        tokenized_length = len(record["input_ids"])
        print(
            f"Raw concept_ids length: {raw_length}  ->  after tokenization/"
            f"invalid-token filtering: {tokenized_length}"
        )

        record = data_processor.slice_out_input_sequence(record)
        final_length = len(record["input_ids"])
        print(
            f"After slice_out_input_sequence (max_length={config.max_position_embeddings}): "
            f"{final_length} tokens"
        )

        input_ids = list(record["input_ids"])
        concept_ids = list(record["concept_ids"])
        if len(input_ids) != len(concept_ids):
            raise RuntimeError(
                f"Length mismatch after truncation: input_ids={len(input_ids)}, "
                f"concept_ids={len(concept_ids)} -- alignment is broken, "
                "investigate before trusting decode output"
            )

        # CehrGptTokenizer.decode(...) already returns List[str], one token
        # per position (tokenization_cehrgpt.py:475-480) -- do not re-split.
        decoded_tokens = tokenizer.decode(input_ids, skip_special_tokens=False)
        if len(decoded_tokens) != len(concept_ids):
            raise RuntimeError(
                f"tokenizer.decode returned {len(decoded_tokens)} tokens but "
                f"concept_ids has {len(concept_ids)} -- investigate before trusting output"
            )

        mismatches = 0
        for i, (decoded, concept_id) in enumerate(zip(decoded_tokens, concept_ids)):
            ok = decoded == concept_id
            if not ok:
                mismatches += 1
            print(f"  [{i:4d}] decoded={decoded!r:30s} concept_id={concept_id!r:30s} {'OK' if ok else 'MISMATCH'}")

        if mismatches == 0:
            print(f"\nAll {len(concept_ids)} positions match exactly.")
        else:
            print(f"\n{mismatches} of {len(concept_ids)} positions DO NOT match.")


if __name__ == "__main__":
    main()
