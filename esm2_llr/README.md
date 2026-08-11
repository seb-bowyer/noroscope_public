# ESM-2 masked-token mutation scoring

A reusable command-line workflow for calculating zero-shot amino-acid mutation scores with an ESM-2 masked language model.

For every scored residue, the script masks that position, obtains log probabilities for the 20 canonical amino acids, and reports a log-likelihood ratio (LLR):

`LLR = log P(mutant | sequence context) - log P(wild type | sequence context)`

The implementation supports FASTA input, batched masking, CUDA when available, per-sequence output directories, run logging, restart-safe output handling and optional final-layer residue embeddings.

The default Hugging Face model is deliberately small so the example can be tested without specialist hardware. A different ESM-2 checkpoint can be supplied with `--model-name`.

## Example

```bash
python -m pip install -r requirements.txt
python score_esm2_llrs.py \
  --input-fasta example_sequences.fasta \
  --output-dir outputs \
  --chunk-size 8
```

The research workflow from which this sample was adapted was used at substantially larger scale on GPU/HPC infrastructure. No unpublished project data are included here.
