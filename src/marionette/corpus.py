"""SAE training text: stream parquet row groups, format chats, tokenize, pack and mix contexts.

Datasets are streamed one parquet row group at a time from the Hugging Face Hub, so nothing is
downloaded whole. Packed token contexts go to a uint32 memory-mapped ``.npy`` file: tokens are
cheap to store (4 bytes each), which decouples network speed from GPU time. Activations, by
contrast, are never written to disk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)

Formatter = Callable[[Any], str | None]


@dataclass(frozen=True)
class Source:
    """One streamed parquet dataset on the Hugging Face Hub."""

    repo: str
    patterns: tuple[str, ...]
    column: str
    kind: Literal["text", "chat"]


def _files(source: Source) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files: list[str] = []
    for pattern in source.patterns:
        found = sorted(fs.glob(f"datasets/{source.repo}/{pattern}"))
        if not found:
            raise FileNotFoundError(f"no file matches {pattern!r} in dataset {source.repo}")
        files.extend(found)
    return files


def _read_row_group(path: str, column: str, index: int) -> list[Any]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    # Each call opens its own handle: remote file objects are not safe to share across threads.
    with HfFileSystem().open(path, "rb") as handle:
        table = pq.ParquetFile(handle).read_row_group(index, columns=[column])
    return table.column(0).to_pylist()  # type: ignore[no-any-return]


def _num_row_groups(path: str) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(path, "rb") as handle:
        return int(pq.ParquetFile(handle).num_row_groups)


def iter_documents(source: Source, formatter: Formatter, prefetch: int = 4) -> Iterator[str]:
    """Yield formatted documents in file/row-group order, prefetching row groups in threads."""
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        for path in _files(source):
            n_groups = _num_row_groups(path)
            pending: list[Future[list[Any]]] = []
            next_group = 0
            while next_group < n_groups or pending:
                while next_group < n_groups and len(pending) < prefetch:
                    pending.append(pool.submit(_read_row_group, path, source.column, next_group))
                    next_group += 1
                for row in pending.pop(0).result():
                    text = formatter(row)
                    if text:
                        yield text


def make_formatter(kind: Literal["text", "chat"], tokenizer: Any) -> Formatter:
    """Plain text passes through; chats are rendered with the model's own chat template."""
    if kind == "text":
        return lambda row: row if isinstance(row, str) and row.strip() else None

    def format_chat(row: Any) -> str | None:
        if not isinstance(row, list) or not row:
            return None
        messages = [{"role": m["role"], "content": m["content"]} for m in row]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        return rendered if isinstance(rendered, str) else None

    return format_chat


def tokenize_documents(
    docs: Iterable[str], tokenizer: Any, separator_id: int, batch_docs: int = 128
) -> Iterator[list[int]]:
    """Tokenize in batches and append a document separator."""
    batch: list[str] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= batch_docs:
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                yield [*ids, separator_id]
            batch = []
    if batch:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            yield [*ids, separator_id]


def pack_contexts(
    token_docs: Iterable[list[int]], ctx_len: int, prefix_id: int
) -> Iterator[np.ndarray]:
    """Concatenate documents and cut fixed-length contexts, each starting with ``prefix_id``.

    Position 0 is therefore always the same attention-sink token and is excluded downstream.
    """
    body = ctx_len - 1
    pending: list[int] = []
    for ids in token_docs:
        pending.extend(ids)
        while len(pending) >= body:
            ctx = np.empty(ctx_len, dtype=np.uint32)
            ctx[0] = prefix_id
            ctx[1:] = pending[:body]
            del pending[:body]
            yield ctx


def mix_contexts(
    streams: list[Iterator[np.ndarray]], weights: list[float], seed: int
) -> Iterator[np.ndarray]:
    """Interleave context streams, picking a source per context with the given weights."""
    if len(streams) != len(weights) or not streams:
        raise ValueError("need one weight per stream")
    rng = np.random.default_rng(seed)
    probs = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    while True:
        choice = int(rng.choice(len(streams), p=probs))
        try:
            yield next(streams[choice])
        except StopIteration:
            raise RuntimeError(
                f"source #{choice} ran out of data; lower the token budget"
            ) from None


def write_token_file(
    path: Path, contexts: Iterator[np.ndarray], n_contexts: int, ctx_len: int
) -> None:
    """Materialise ``n_contexts`` contexts; written under a temp name, renamed when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npy")
    tokens = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint32, shape=(n_contexts, ctx_len))
    for i in range(n_contexts):
        tokens[i] = next(contexts)
        if (i + 1) % 10_000 == 0:
            log.info("%s: %d / %d contexts", path.name, i + 1, n_contexts)
    tokens.flush()
    del tokens
    tmp.replace(path)


def open_tokens(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `marionette sae tokens` first")
    tokens: np.ndarray = np.load(path, mmap_mode="r")
    return tokens
