# Reproduce the FLARE workflow

This repair is based on release commit
`b49944e66892e11d24f81059fc1a87b70c3c45b8` (2026-05-11).
It preserves the original release and adds a separately tested working copy.
No original trained parameters are needed. Public data/models must remain
obtainable; this package is not an offline mirror of those resources.

## Complete pipeline

Generate each model's embeddings with `scripts/generate_embeddings.py`, using
the same corpus, sample filtering, row order, task instructions and pooling.
Record the source dataset ID/config/split/text columns and model revision.
The separate backup model inventory lists IDs and available cache revisions;
those revisions are not proven historical experiment revisions.

Create `embeddings.json` next to your matrices. Paths resolve relative to this
JSON file, so the project can move between machines:

```json
{
  "minilm": "embeddings/minilm.npy",
  "stella": "embeddings/stella.npy"
}
```

Then, from the extracted source root:

```bash
python scripts/run_flow_pipeline.py \
  --embeddings embeddings.json --output-dir data/example \
  --marginal-config configs/marginal.yaml \
  --conditional-config configs/conditional.yaml
```

This prepares each marginal and directed conditional pair, trains the flows,
exports per-example NLLs, and writes `data/example/flare_is_scores.json`.
`workflow_manifest.json` records exact commands, input hashes and configuration
contents. Its `completed` flag becomes true only after scoring succeeds.
Re-running retrains the flows; use a new output directory to keep old results.
There are M marginal trainings and M*(M-1) conditional trainings for M models.

The default split is 80% train, 10% validation, 10% test; final scoring uses test.
To follow older validation-only experiments, use
`--test-size 0 --val-size 0.1 --score-split val`. This is a workflow option,
not a claim that the historical experiment split or results have been recovered.
Pass `--device cpu` to hide all GPUs from the subprocesses.

## Fixes

- Conditional training restores the marginal network from its checkpoint
  architecture instead of rebuilding it from an incompatible conditional YAML.
  It inherits that backbone, checks complete state transfer and verifies that
  the initial conditional density equals the marginal density.
- New checkpoints store explicit architecture kwargs, including adapter rank
  and scale. Training, NLL export and MI bounds share strict checkpoint loading.
- Target train/validation array hashes must agree. Pair preparation uses the
  same target transforms and splits as the corresponding marginal preparation.
- `export_per_example_nll.py` bridges trained release weights to the NPZ format
  consumed by the existing IS aggregator. It checks aligned rows and finite NLLs.
- Three stage lengths and the total epoch cap are configurable. Defaults retain
  the original actual 60/60/220 schedule. Architecture settings are inherited;
  unused misleading conditional YAML entries have been removed. Adapter learning
  rate and the stage transition threshold are explicit in YAML.
- EMA averages model parameters and copies buffers exactly. It no longer averages
  integer permutation indices or initialization flags. ActNorm initializes on
  training samples before the EMA snapshot. Small training sets are not silently
  dropped when the batch size exceeds their length.
- Standardization-only preprocessing is stated explicitly. The implementation
  does not apply PCA; stale PCA objects cause a clear error rather than a silent
  change in transforms.
- A Python 3.10 reference environment and pinned principal dependencies are
  included, replacing reliance on the old Python 3.9 environment. See README
  for installation and the precise environment validation boundary.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/smoke_workflow.py --output-dir /tmp/flare-smoke-new
python scripts/smoke_workflow.py --output-dir /tmp/flare-text-smoke-new --with-embeddings
```

Use a new output directory for each smoke run. The first pipeline uses 64
synthetic vectors in 8 and 6 dimensions. The text pipeline generates two
384-dimensional representations of 64 local texts using cached MiniLM with
mean/CLS pooling. These are two variants of the same encoder, not a comparison
of two independently pretrained models. Each pipeline trains two marginal
flows and both directed conditional pairs, exports 13 held-out test NLL pairs
per direction and verifies the score formula. Each also checks that altered
target validation rows are rejected. Tests use short schedules, not the full
paper training settings. No GPU jobs or full benchmark reproduction are part
of this validation. Baseline training and the other seven embedding models
have not been revalidated by these checks.

For the offline text check on the original server, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. A new machine must first download the MiniLM model.
Evidence and actual test results are stored in `validation/` in the repair pack.
