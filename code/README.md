# FLARE

FLARE evaluates embedding models on an unlabeled corpus with normalizing-flow
information-sufficiency estimates.

The repository is organized around three paths:

- `flare/`: the FLARE normalizing-flow implementation.
- `scripts/`: command line tools for embedding generation, data preparation,
  flow training, scoring, and baselines.
- `scripts/baselines/emir/`: the EMIR baseline source used by
  `scripts/baselines/compute_emir_knife.py`.

## Install

Use Python 3.10 or later. The small CPU workflow was tested on Python 3.10.20.
For the tested package versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
```

`environment_tested.json` records the actual installed builds. The test used an
isolated venv inheriting packages from an existing environment; a fresh install
of every pinned dependency has not been tested. GritLM additionally requires the
optional `gritlm` package; that model was not part of this smoke check.

The original broad dependency specification remains available:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Generate Embeddings

```bash
python scripts/generate_embeddings.py \
  --dataset maiammar/Augmented_stsb_multi_mt \
  --split validation \
  --columns sentence2 \
  --model stella \
  --output-dir embeddings/maiammar_Augmented_stsb_multi_mt/stella_base_en_v2 \
  --output-name val_sentence2_embeddings \
  --batch-size 128 \
  --max-length 512 \
  --gpu-id 0
```

## Prepare Flow Data

The release applies train-fitted StandardScaler transforms, with no PCA.
Legacy PCA options are accepted for compatibility and do not enable PCA.
Prepare the marginal target as well as each conditional pair with the same
target embeddings, validation/test fractions and row order:

```bash
python scripts/prepare/prepare_data.py \
  --y_path embeddings/DATASET/TARGET/abstract_embeddings.npy \
  --output_dir data/DATASET/TARGET_marginal \
  --test_size 0.0 --val_size 0.1
```

```bash
python scripts/prepare/prepare_data.py \
  --x_path embeddings/DATASET/SOURCE/abstract_embeddings.npy \
  --y_path embeddings/DATASET/TARGET/abstract_embeddings.npy \
  --output_dir data/DATASET/SOURCE_to_TARGET_no_pca \
  --test_size 0.0 \
  --val_size 0.1
```

## Train FLARE Flows

```bash
python scripts/train/train_marginal.py \
  --config configs/marginal.yaml \
  --data_dir data/DATASET/TARGET_marginal \
  --out data/DATASET/TARGET_marginal/marginal_model.pth
```

```bash
python scripts/train/train_conditional.py \
  --config configs/conditional.yaml \
  --data_dir data/DATASET/SOURCE_to_TARGET_no_pca \
  --marginal_model data/DATASET/TARGET_marginal/marginal_model.pth \
  --out data/DATASET/SOURCE_to_TARGET_no_pca/cond_flow.pt
```

## Score FLARE

First export aligned per-example NLLs from each trained pair. This step must be
repeated for every source/target pair; the automatic pipeline below does it.

```bash
python scripts/score/export_per_example_nll.py \
  --data-dir data/DATASET/SOURCE_to_TARGET_no_pca \
  --marginal-model data/DATASET/TARGET_marginal/marginal_model.pth \
  --conditional-model data/DATASET/SOURCE_to_TARGET_no_pca/cond_flow.pt \
  --split val --device auto
```

Use `--split test` for a separate test split created with `--test_size > 0`.
The exporter rejects checkpoints trained with a different target split or
transform. Old checkpoints without the new provenance metadata must be retrained.

```bash
python scripts/score/compute_flare_is.py \
  --data-dir data/DATASET \
  --models gte_Qwen2_7B_instruct Linq_Embed_Mistral SFR_Embedding_Mistral \
           Zeta_Alpha_E5_Mistral bge_multilingual_gemma2 GritLM_7B \
           all_MiniLM_L6_v2 stella_base_en_v2 \
  --target-dims configs/model_dims.json \
  --output results/flare_is/DATASET.json
```

## Baselines

Geometry baselines:

```bash
python scripts/baselines/compute_embedding_metrics.py \
  --embedding_dir embeddings/DATASET \
  --embedding_file abstract_embeddings.npy \
  --output metrics.json
```

EMIR/KNIFE baseline:

```bash
python scripts/baselines/compute_emir_knife.py \
  --source-models gte_Qwen2_7B_instruct Linq_Embed_Mistral \
  --target-models all_MiniLM_L6_v2 stella_base_en_v2 \
  --embeddings-dir embeddings/DATASET \
  --embedding-file abstract_embeddings.npy \
  --output-dir results/emir_knife \
  --device auto \
  --batch-size 256 \
  --emir-path scripts/baselines/emir
```
