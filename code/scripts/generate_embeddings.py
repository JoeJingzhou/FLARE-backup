#!/usr/bin/env python3
"""Generate text embeddings for FLARE experiments.

This is the release-friendly replacement for the project-specific
``embedding_converter-*`` scripts. It keeps the model-specific instruction and
pooling behavior used in the paper, while exposing one consistent CLI.

Examples:
    python scripts/generate_embeddings.py \
        --dataset smksaha/apt-eval --split train --columns generation \
        --model gritlm --output-dir embeddings/smksaha_apt-eval/GritLM_7B

    python scripts/generate_embeddings.py \
        --input-file texts.txt --model minilm --output-dir embeddings/example
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "stella": "infgrad/stella-base-en-v2",
    "stella-base-en-v2": "infgrad/stella-base-en-v2",
    "bge-gemma2": "BAAI/bge-multilingual-gemma2",
    "bge-multilingual-gemma2": "BAAI/bge-multilingual-gemma2",
    "gritlm": "GritLM/GritLM-7B",
    "gritlm-7b": "GritLM/GritLM-7B",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "gte-qwen2-7b-instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "linq": "Linq-AI-Research/Linq-Embed-Mistral",
    "linq-embed-mistral": "Linq-AI-Research/Linq-Embed-Mistral",
    "sfr": "Salesforce/SFR-Embedding-Mistral",
    "sfr-embedding-mistral": "Salesforce/SFR-Embedding-Mistral",
    "zeta": "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
    "zeta-alpha-e5-mistral": "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
}

DEFAULT_MAX_LENGTH = {
    "sentence-transformers/all-MiniLM-L6-v2": 256,
    "infgrad/stella-base-en-v2": 512,
    "BAAI/bge-multilingual-gemma2": 4096,
    "GritLM/GritLM-7B": 4096,
    "Alibaba-NLP/gte-Qwen2-7B-instruct": 8192,
    "Linq-AI-Research/Linq-Embed-Mistral": 4096,
    "Salesforce/SFR-Embedding-Mistral": 4096,
    "zeta-alpha-ai/Zeta-Alpha-E5-Mistral": 4096,
}

LAST_TOKEN_MODELS = (
    "bge-multilingual-gemma2",
    "gte-qwen2",
    "linq-embed-mistral",
    "sfr-embedding-mistral",
    "zeta-alpha-e5-mistral",
)

INSTRUCT_MODELS = (
    "bge-multilingual-gemma2",
    "gritlm",
    "linq-embed-mistral",
    "sfr-embedding-mistral",
    "zeta-alpha-e5-mistral",
)


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def model_key(model_name: str) -> str:
    return model_name.lower()


def uses_instruction(model_name: str) -> bool:
    key = model_key(model_name)
    return any(part in key for part in INSTRUCT_MODELS)


def uses_last_token_pooling(model_name: str) -> bool:
    key = model_key(model_name)
    return any(part in key for part in LAST_TOKEN_MODELS)


def format_instruction(model_name: str, text: str, task: str) -> str:
    key = model_key(model_name)
    if "bge-multilingual-gemma2" in key:
        return f"<instruct>{task}\n<query>{text}"
    if "gritlm" in key:
        return text
    return f"Instruct: {task}\nQuery: {text}"


def gritlm_instruction(task: str) -> str:
    return f"<|user|>\n{task}\n<|embed|>\n" if task else "<|embed|>\n"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def choose_device(device: str, gpu_id: int | None) -> torch.device:
    import torch

    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-id was provided but CUDA is not available")
        return torch.device(f"cuda:{gpu_id}")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_texts_from_hf_dataset(
    dataset_name: str,
    split: str,
    columns: list[str] | None,
    dataset_config: str | None,
) -> dict[str, list[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to use --dataset") from exc

    ds = load_dataset(dataset_name, dataset_config, split=split)
    if columns is None:
        columns = detect_text_columns(ds)
    return {col: [clean_text(x) for x in ds[col]] for col in columns}


def detect_text_columns(dataset) -> list[str]:
    columns = []
    sample_count = min(100, len(dataset))
    for col in dataset.column_names:
        values = [dataset[i][col] for i in range(sample_count)]
        string_values = [v for v in values if isinstance(v, str) and v.strip()]
        if len(string_values) >= max(1, sample_count // 2):
            columns.append(col)
    if not columns:
        raise ValueError("No text columns detected. Pass --columns explicitly.")
    return columns


def load_texts_from_file(path: Path, columns: list[str] | None, text_key: str | None) -> dict[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return {columns[0] if columns else "text": [line.rstrip("\n") for line in path.open(encoding="utf-8")]}
    if suffix == ".jsonl":
        key = text_key or (columns[0] if columns else "text")
        texts = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(clean_text(json.loads(line)[key]))
        return {key: texts}
    if suffix == ".json":
        key = text_key or (columns[0] if columns else "text")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {key: [clean_text(item[key] if isinstance(item, dict) else item) for item in data]}
        return {key: [clean_text(x) for x in data[key]]}
    if suffix == ".csv":
        if not columns:
            raise ValueError("CSV input requires --columns")
        output = {col: [] for col in columns}
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    output[col].append(clean_text(row[col]))
        return output
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
        use_device_map: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.use_device_map = use_device_map
        self.is_gritlm = "gritlm" in model_key(model_name)
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self) -> None:
        import torch

        if self.is_gritlm:
            try:
                from gritlm import GritLM
            except ImportError as exc:
                raise RuntimeError("Install gritlm to use GritLM/GritLM-7B") from exc
            self.model = GritLM(self.model_name, torch_dtype=self.dtype, mode="embedding")
            return

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.dtype,
        }
        if self.use_device_map and self.device.type == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        task: str,
        add_instruction: bool,
        normalize: bool,
        pooling: str,
    ) -> np.ndarray:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        if self.is_gritlm:
            instruction = gritlm_instruction(task) if add_instruction else gritlm_instruction("")
            chunks = []
            with torch.no_grad():
                for batch in tqdm(list(batched(texts, batch_size)), desc="Embedding batches"):
                    emb = self.model.encode(batch, instruction=instruction)
                    emb = np.asarray(emb, dtype=np.float32)
                    if normalize:
                        emb = l2_normalize_np(emb)
                    chunks.append(emb)
            return np.vstack(chunks).astype(np.float32)

        assert self.tokenizer is not None
        assert self.model is not None

        chunks = []
        for batch in tqdm(list(batched(texts, batch_size)), desc="Embedding batches"):
            formatted = [
                format_instruction(self.model_name, text, task) if add_instruction else text
                for text in batch
            ]
            inputs = self.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state

                if pooling == "auto":
                    pooled = last_token_pool(hidden, inputs["attention_mask"]) if uses_last_token_pooling(self.model_name) else mean_pool(hidden, inputs["attention_mask"])
                elif pooling == "last":
                    pooled = last_token_pool(hidden, inputs["attention_mask"])
                elif pooling == "mean":
                    pooled = mean_pool(hidden, inputs["attention_mask"])
                elif pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    raise ValueError(f"Unknown pooling mode: {pooling}")

                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.detach().float().cpu().numpy())
            del inputs, outputs, hidden, pooled
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.vstack(chunks).astype(np.float32)


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    import numpy as np

    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return x / denom


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_")


def parse_dtype(dtype: str, device: torch.device) -> torch.dtype:
    import torch

    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[dtype]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate embeddings for FLARE.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", "-d", help="Hugging Face dataset name")
    source.add_argument("--input-file", type=Path, help="Local .txt, .jsonl, .json, or .csv file")

    parser.add_argument("--dataset-config", help="Hugging Face dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--columns", "-c", nargs="+", help="Text columns to embed")
    parser.add_argument("--text-key", help="Text key for JSON/JSONL input")

    parser.add_argument("--model", "-m", default="minilm", help="Model alias or Hugging Face model id")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("embeddings"))
    parser.add_argument("--output-name", help="Base output filename. Defaults to column name.")
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=0, help="Reserved for compatibility; batching is streaming already.")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    parser.add_argument("--gpu-id", type=int, help="Equivalent to --device cuda:N")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--pooling", default="auto", choices=["auto", "mean", "last", "cls"])
    parser.add_argument("--task", default="Given a text, retrieve its semantic representation")
    parser.add_argument("--no-instruction", action="store_true", help="Disable instruction formatting")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--device-map", action="store_true", help="Use transformers device_map='auto' for large models")
    parser.add_argument("--limit", type=int, help="Embed only the first N rows")
    parser.add_argument("--save-metadata", dest="save_metadata", action="store_true")
    parser.add_argument("--no-save-metadata", dest="save_metadata", action="store_false")
    parser.set_defaults(save_metadata=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_name = resolve_model_name(args.model)
    device = choose_device(args.device, args.gpu_id)
    dtype = parse_dtype(args.dtype, device)
    max_length = args.max_length or DEFAULT_MAX_LENGTH.get(model_name, 512)
    add_instruction = uses_instruction(model_name) and not args.no_instruction

    if args.dataset:
        text_columns = load_texts_from_hf_dataset(
            args.dataset,
            split=args.split,
            columns=args.columns,
            dataset_config=args.dataset_config,
        )
        source_name = safe_name(args.dataset)
    else:
        text_columns = load_texts_from_file(args.input_file, args.columns, args.text_key)
        source_name = safe_name(args.input_file.stem)

    if args.limit is not None:
        text_columns = {name: values[: args.limit] for name, values in text_columns.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = EmbeddingModel(
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        use_device_map=args.device_map,
    )

    metadata = {
        "model": model_name,
        "model_arg": args.model,
        "source": args.dataset or str(args.input_file),
        "split": args.split if args.dataset else None,
        "dataset_config": args.dataset_config,
        "columns": list(text_columns.keys()),
        "max_length": max_length,
        "batch_size": args.batch_size,
        "pooling": args.pooling,
        "instruction": add_instruction,
        "task": args.task if add_instruction else None,
        "normalize": args.normalize,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
    }

    for col, texts in text_columns.items():
        if not texts:
            raise ValueError(f"Column {col} is empty")
        print(f"Embedding column={col} rows={len(texts)} model={model_name}")
        embeddings = encoder.encode(
            texts=texts,
            batch_size=args.batch_size,
            max_length=max_length,
            task=args.task,
            add_instruction=add_instruction,
            normalize=args.normalize,
            pooling=args.pooling,
        )
        base = args.output_name or f"{safe_name(source_name)}_{safe_name(col)}"
        if len(text_columns) > 1 and args.output_name:
            base = f"{base}_{safe_name(col)}"
        out_path = args.output_dir / f"{base}.npy"
        import numpy as np

        np.save(out_path, embeddings)
        print(f"Saved {out_path} shape={tuple(embeddings.shape)}")
        metadata.setdefault("outputs", {})[col] = {
            "path": str(out_path),
            "shape": list(embeddings.shape),
        }

    if args.save_metadata:
        meta_path = args.output_dir / f"{args.output_name or safe_name(source_name)}_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
