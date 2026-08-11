#!/usr/bin/env python3

"""
Generate ESM-2 masked-token log-likelihood ratios (LLRs) for protein sequences.

For each sequence position, the residue is masked and ESM-2 is used to estimate
log probabilities for all 20 standard amino acids. LLR values are calculated as:

    LLR = log P(mutant amino acid | masked sequence)
          - log P(wildtype amino acid | masked sequence)

This script is designed for protein FASTA files and can handle aligned sequences.
By default, gap positions are skipped.

Example usage:

python score_esm2_llrs.py \
  --input-fasta example_sequences.fasta \
  --output-dir outputs \
  --model-name facebook/esm2_t12_35M_UR50D \
  --chunk-size 23 \
  --save-embeddings
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForMaskedLM


STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
GAP_CHARACTERS = {"-", "."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 masked-token LLRs for protein FASTA sequences."
    )

    parser.add_argument(
        "--input-fasta",
        required=True,
        type=Path,
        help="Input protein FASTA file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where per-sequence LLR outputs will be written.",
    )

    parser.add_argument(
        "--model-name",
        default="facebook/esm2_t12_35M_UR50D",
        help="HuggingFace model name or local model path.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=23,
        help="Number of masked positions to process per model forward pass.",
    )

    parser.add_argument(
        "--start-pos",
        type=int,
        default=1,
        help="First 1-indexed sequence position to score.",
    )

    parser.add_argument(
        "--end-pos",
        type=int,
        default=None,
        help="Final 1-indexed sequence position to score. Defaults to sequence length.",
    )

    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save final-layer per-residue embeddings for each unmasked sequence.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-sequence outputs.",
    )

    parser.add_argument(
        "--include-gaps",
        action="store_true",
        help=(
            "Score alignment gap positions. By default, gap positions '-' and '.' "
            "are skipped. Usually leave this off for aligned protein FASTA files."
        ),
    )

    parser.add_argument(
        "--allow-nonstandard",
        action="store_true",
        help=(
            "Allow sequences containing non-standard amino acid symbols. "
            "Positions with non-standard wildtype residues are still skipped."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device selection. Default: auto.",
    )

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for model loading. Default: float32.",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "llr_generation.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)


def safe_sequence_id(seq_id: str) -> str:
    """
    Convert a FASTA record ID into a filesystem-safe directory name.
    """
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", seq_id)
    return safe_id.strip("_")


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_gpu_memory() -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(
            "[GPU MEM] Allocated: %.2f GB | Reserved: %.2f GB",
            allocated,
            reserved,
        )


def validate_sequence(
    sequence: str,
    allow_nonstandard: bool = False,
    include_gaps: bool = False,
) -> tuple[bool, str]:
    """
    Basic generic validation for protein FASTA sequences.

    This intentionally avoids organism-specific assumptions.
    """
    if not sequence:
        return False, "Sequence is empty."

    allowed = set(STANDARD_AMINO_ACIDS)

    if include_gaps:
        allowed = allowed | GAP_CHARACTERS

    observed = set(sequence.upper())

    if allow_nonstandard:
        return True, ""

    invalid = observed - allowed

    if invalid:
        return False, f"Sequence contains unsupported symbols: {sorted(invalid)}"

    return True, ""


def clean_sequence_for_model(sequence: str) -> str:
    """
    Convert sequence to uppercase.

    Note: this does not remove gaps. Gap handling is done position-wise so that
    alignment coordinates can be preserved in the output.
    """
    return sequence.upper()


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
):
    logging.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info("Loading model: %s", model_name)
    logging.info("Device: %s", device)
    logging.info("Dtype: %s", dtype)

    if device.type == "cuda":
        model = EsmForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = EsmForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model.to(device)

    model.eval()
    return model, tokenizer


def get_positions_to_score(
    sequence: str,
    start_pos: int,
    end_pos: Optional[int],
    include_gaps: bool,
) -> list[int]:
    """
    Return 1-indexed sequence/alignment positions to score.

    If include_gaps=False, gap positions are skipped.
    Positions with non-standard wildtype residues are also skipped.
    """
    sequence_length = len(sequence)

    if end_pos is None:
        end_pos = sequence_length

    if start_pos < 1:
        raise ValueError("--start-pos must be >= 1")

    if end_pos > sequence_length:
        raise ValueError(
            f"--end-pos ({end_pos}) is greater than sequence length ({sequence_length})"
        )

    positions = []

    for pos in range(start_pos, end_pos + 1):
        wt_residue = sequence[pos - 1]

        if wt_residue in GAP_CHARACTERS and not include_gaps:
            continue

        if wt_residue not in STANDARD_AMINO_ACIDS:
            continue

        positions.append(pos)

    return positions


def calculate_llrs_for_sequence(
    protein_sequence: str,
    model,
    tokenizer,
    output_dir: Path,
    start_pos: int = 1,
    end_pos: Optional[int] = None,
    chunk_size: int = 23,
    include_gaps: bool = False,
    save_embeddings: bool = False,
) -> dict:
    """
    Calculate masked-token LLRs for one protein sequence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    protein_sequence = clean_sequence_for_model(protein_sequence)
    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt").to(device)

    # Tokenized length includes special start/end tokens.
    model_sequence_length = input_ids.shape[1] - 2

    if model_sequence_length != len(protein_sequence):
        logging.warning(
            "Tokenized sequence length (%s) differs from raw sequence length (%s).",
            model_sequence_length,
            len(protein_sequence),
        )

    positions_to_score = get_positions_to_score(
        sequence=protein_sequence,
        start_pos=start_pos,
        end_pos=end_pos,
        include_gaps=include_gaps,
    )

    llr_data = []

    logging.info("Scoring %s positions.", len(positions_to_score))
    log_gpu_memory()

    for i in range(0, len(positions_to_score), chunk_size):
        chunk_positions = positions_to_score[i : i + chunk_size]

        masked_inputs = []

        for pos in chunk_positions:
            masked = input_ids.clone()

            # Token position is the same as 1-indexed protein position because
            # token index 0 is the special start token.
            masked[0, pos] = tokenizer.mask_token_id
            masked_inputs.append(masked)

        batch_input = torch.cat(masked_inputs, dim=0)
        masked_positions_tensor = torch.tensor(chunk_positions, device=device)

        with torch.no_grad():
            logits = model(batch_input).logits.to(torch.float32)

        log_probs = torch.log_softmax(logits, dim=-1)

        wt_residue_token_ids = input_ids[0, masked_positions_tensor]
        log_prob_wt = log_probs[
            torch.arange(len(chunk_positions), device=device),
            masked_positions_tensor,
            wt_residue_token_ids,
        ]

        for j, pos in enumerate(chunk_positions):
            wt_residue = protein_sequence[pos - 1]
            log_probs_pos = log_probs[j, pos]

            for mutant_aa in STANDARD_AMINO_ACIDS:
                mutant_token_id = tokenizer.convert_tokens_to_ids(mutant_aa)
                log_prob_mutant = log_probs_pos[mutant_token_id].item()
                llr = log_prob_mutant - log_prob_wt[j].item()

                llr_data.append(
                    [
                        pos,
                        wt_residue,
                        mutant_aa,
                        log_prob_mutant,
                        log_prob_wt[j].item(),
                        llr,
                    ]
                )

        logging.info(
            "Processed positions %s-%s",
            chunk_positions[0],
            chunk_positions[-1],
        )

    llr_output_file = output_dir / "llr_values.csv"

    with llr_output_file.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "position",
                "wt_residue",
                "mutant_residue",
                "log_prob_mutant",
                "log_prob_wt",
                "llr",
            ]
        )
        writer.writerows(llr_data)

    logging.info("LLR values saved to %s", llr_output_file)

    embedding_saved = False

    if save_embeddings:
        with torch.no_grad():
            output = model(input_ids, output_hidden_states=True)
            final_embeddings = output.hidden_states[-1][0]

            # Remove special start/end tokens.
            final_embeddings = final_embeddings[1:-1].cpu().numpy()

        emb_file = output_dir / "embeddings.npy"
        np.save(emb_file, final_embeddings)

        embedding_saved = True
        logging.info("Embeddings saved to %s", emb_file)

    return {
        "sequence_length": len(protein_sequence),
        "n_positions_scored": len(positions_to_score),
        "n_llr_rows": len(llr_data),
        "embedding_saved": embedding_saved,
    }


def write_run_summary(summary_rows: list[dict], output_file: Path) -> None:
    fieldnames = [
        "sequence_id",
        "safe_sequence_id",
        "sequence_length",
        "n_positions_scored",
        "n_llr_rows",
        "embedding_saved",
        "status",
        "error_message",
    ]

    with output_file.open(mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def process_fasta_file(args: argparse.Namespace) -> None:
    setup_logging(args.output_dir)

    logging.info("Starting ESM-2 LLR generation.")
    logging.info("Input FASTA: %s", args.input_fasta)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Model: %s", args.model_name)
    logging.info("Chunk size: %s", args.chunk_size)
    logging.info("Save embeddings: %s", args.save_embeddings)
    logging.info("Overwrite existing outputs: %s", args.overwrite)
    logging.info("Include gaps: %s", args.include_gaps)

    if not args.input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {args.input_fasta}")

    device = get_device(args.device)
    dtype = get_torch_dtype(args.dtype)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=device,
        dtype=dtype,
    )

    summary_rows = []

    records = list(SeqIO.parse(args.input_fasta, "fasta"))
    logging.info("Found %s FASTA records.", len(records))

    for record in records:
        sequence_id = record.id
        safe_id = safe_sequence_id(sequence_id)
        sequence = str(record.seq).upper()

        sequence_output_dir = args.output_dir / safe_id
        llr_output_file = sequence_output_dir / "llr_values.csv"

        logging.info("Starting sequence: %s", sequence_id)

        if llr_output_file.exists() and not args.overwrite:
            logging.info(
                "Skipping %s because output already exists: %s",
                sequence_id,
                llr_output_file,
            )

            summary_rows.append(
                {
                    "sequence_id": sequence_id,
                    "safe_sequence_id": safe_id,
                    "sequence_length": len(sequence),
                    "n_positions_scored": "",
                    "n_llr_rows": "",
                    "embedding_saved": "",
                    "status": "skipped_existing_output",
                    "error_message": "",
                }
            )
            continue

        valid, validation_message = validate_sequence(
            sequence=sequence,
            allow_nonstandard=args.allow_nonstandard,
            include_gaps=args.include_gaps,
        )

        if not valid:
            logging.warning(
                "Skipping sequence %s: %s",
                sequence_id,
                validation_message,
            )

            summary_rows.append(
                {
                    "sequence_id": sequence_id,
                    "safe_sequence_id": safe_id,
                    "sequence_length": len(sequence),
                    "n_positions_scored": "",
                    "n_llr_rows": "",
                    "embedding_saved": "",
                    "status": "failed_validation",
                    "error_message": validation_message,
                }
            )
            continue

        try:
            result = calculate_llrs_for_sequence(
                protein_sequence=sequence,
                model=model,
                tokenizer=tokenizer,
                output_dir=sequence_output_dir,
                start_pos=args.start_pos,
                end_pos=args.end_pos,
                chunk_size=args.chunk_size,
                include_gaps=args.include_gaps,
                save_embeddings=args.save_embeddings,
            )

            summary_rows.append(
                {
                    "sequence_id": sequence_id,
                    "safe_sequence_id": safe_id,
                    "sequence_length": result["sequence_length"],
                    "n_positions_scored": result["n_positions_scored"],
                    "n_llr_rows": result["n_llr_rows"],
                    "embedding_saved": result["embedding_saved"],
                    "status": "completed",
                    "error_message": "",
                }
            )

            logging.info("Completed sequence: %s", sequence_id)

        except Exception as error:
            logging.exception("Failed sequence: %s", sequence_id)

            summary_rows.append(
                {
                    "sequence_id": sequence_id,
                    "safe_sequence_id": safe_id,
                    "sequence_length": len(sequence),
                    "n_positions_scored": "",
                    "n_llr_rows": "",
                    "embedding_saved": "",
                    "status": "failed_runtime_error",
                    "error_message": str(error),
                }
            )

        summary_file = args.output_dir / "run_summary.csv"
        write_run_summary(summary_rows, summary_file)

    summary_file = args.output_dir / "run_summary.csv"
    write_run_summary(summary_rows, summary_file)

    logging.info("Finished ESM-2 LLR generation.")
    logging.info("Run summary written to %s", summary_file)


def main() -> None:
    args = parse_args()
    process_fasta_file(args)


if __name__ == "__main__":
    main()