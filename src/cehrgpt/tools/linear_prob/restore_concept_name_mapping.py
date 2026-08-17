"""Restore a missing/corrupted concept_name_mapping.json for a CehrGptTokenizer
checkpoint, without retraining anything.

Why this doesn't use CehrGptTokenizer.from_pretrained(...): that method itself
hard-requires concept_name_mapping.json to already exist -- it returns None
early if that file is missing or unreadable (tokenization_cehrgpt.py:744-745).
So if that's exactly the file you're trying to restore, the normal loader
can't be used to get the vocabulary -- this script reads the underlying
tokenizer file (cehrgpt_tokenizer.json) directly with the `tokenizers`
library instead.

Reconstruction mirrors the original pruning step in
CehrGptTokenizer.train_tokenizer (tokenization_cehrgpt.py:1131-1136):
    concept_name_mapping = {
        concept_id: concept_name_mapping[concept_id]
        for concept_id in vocab.keys()
        if concept_id in concept_name_mapping
    }
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

TOKENIZER_FILE_NAME = "cehrgpt_tokenizer.json"
CONCEPT_MAPPING_FILE_NAME = "concept_name_mapping.json"


def main():
    parser = argparse.ArgumentParser(
        description="Restore concept_name_mapping.json from an OMOP concept "
        "table, using the intact cehrgpt_tokenizer.json to determine which "
        "concept_ids need a name."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing the intact cehrgpt_tokenizer.json (this is "
        "also where concept_name_mapping.json will be written)",
    )
    parser.add_argument(
        "--vocab_dir",
        required=True,
        help="OMOP vocabulary directory containing a 'concept' parquet table "
        "(with concept_id, concept_name columns) -- ideally the exact same "
        "vocab_dir used at training time, for name fidelity",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    tokenizer_file = checkpoint_dir / TOKENIZER_FILE_NAME
    if not tokenizer_file.exists():
        raise RuntimeError(
            f"{tokenizer_file} not found -- the main tokenizer file itself is "
            "missing, so the vocabulary can't be recovered this way."
        )

    output_file = checkpoint_dir / CONCEPT_MAPPING_FILE_NAME
    if output_file.exists():
        backup_file = checkpoint_dir / (CONCEPT_MAPPING_FILE_NAME + ".bak")
        print(f"{output_file} already exists -- moving it aside to {backup_file} before writing the restored version")
        output_file.rename(backup_file)

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    vocab = tokenizer.get_vocab()
    print(f"Loaded vocabulary from {tokenizer_file}: {len(vocab)} tokens")

    concept_pd = pd.read_parquet(Path(args.vocab_dir) / "concept")
    omop_concept_name_lookup = {
        str(row.concept_id): row.concept_name for row in concept_pd.itertuples()
    }
    print(f"Loaded {len(omop_concept_name_lookup)} concept_id -> concept_name entries from {args.vocab_dir}")

    concept_name_mapping = {
        concept_id: omop_concept_name_lookup[concept_id]
        for concept_id in vocab.keys()
        if concept_id in omop_concept_name_lookup
    }

    unmatched = [c for c in vocab.keys() if c not in omop_concept_name_lookup]
    print(
        f"Restored {len(concept_name_mapping)} of {len(vocab)} vocabulary tokens to a concept name "
        f"({len(unmatched)} tokens have no OMOP match -- expected for special/engineered tokens "
        "like [VS], [VE], year:*, age:*, value bins, demographics, etc.)"
    )

    with open(output_file, "w") as f:
        json.dump(concept_name_mapping, f)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
