import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Prompt template for motion-to-program generation.
# The motion prefix tokens are prepended before the text by the model,
# so the text input starts with the prompt and ends with the program.
PROMPT_PREFIX = "Program: "


def _cache_path_for(h5_path: str) -> Path:
    """Return the `.pt` cache path corresponding to an HDF5 file.

    The cache lives next to the original file:
        /data/train_diverse.h5  →  /data/train_diverse.cache.pt
    """
    p = Path(h5_path)
    return p.with_suffix(".cache.pt")


def _h5_fingerprint(h5_path: str) -> str:
    """Quick fingerprint: file size + mtime (avoids hashing 14 GB)."""
    p = Path(h5_path)
    stat = p.stat()
    raw = f"{p.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()


def preload_h5_to_cache(h5_path: str) -> Path:
    """Read an HDF5 motion file and save a fast-loading `.pt` cache.

    Returns the cache path.  Safe to call from multiple processes —
    an atomic rename prevents readers from seeing a half-written file.
    """
    cache_path = _cache_path_for(h5_path)
    fingerprint = _h5_fingerprint(h5_path)

    logger.info(f"Building .pt cache from {h5_path} …")
    t0 = time.time()

    programs: List[str] = []
    obs_list: List[torch.Tensor] = []

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        for i, key in enumerate(keys):
            motion = f[key]["motion"][()]
            program = f[key].attrs["program"]
            programs.append(program)
            obs_list.append(torch.tensor(motion, dtype=torch.float32))
            if (i + 1) % 10000 == 0 or (i + 1) == n_keys:
                logger.info(
                    f"  … {i + 1}/{n_keys} samples read "
                    f"({time.time() - t0:.0f}s elapsed)"
                )

    # Stack into a single contiguous tensor for fast serialization
    obs_stacked = torch.stack(obs_list)

    payload = {
        "programs": programs,
        "obs": obs_stacked,
        "fingerprint": fingerprint,
    }

    # Atomic write: save to a temp file first, then rename
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.rename(cache_path)

    elapsed = time.time() - t0
    logger.info(
        f"Cache saved: {cache_path.name}  "
        f"({len(programs)} samples, {obs_stacked.nbytes / 1024**3:.1f} GB, "
        f"{elapsed:.1f}s)"
    )
    return cache_path


def copy_cache_to_local(h5_path: str, local_dir: str = "/tmp/exact_cache") -> Path:
    """Copy the `.pt` cache to a local directory for faster reads.

    Useful when the data lives on NFS — even a single bulk read of a
    14 GB `.pt` file is slow over the network.  Copying to local SSD/tmpfs
    once and reading from there is much faster.

    Returns the local cache path.
    """
    cache_path = _cache_path_for(h5_path)
    if not cache_path.exists():
        preload_h5_to_cache(h5_path)

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    local_cache = local / cache_path.name

    if local_cache.exists() and local_cache.stat().st_size == cache_path.stat().st_size:
        logger.debug(f"Local cache already exists: {local_cache}")
        return local_cache

    import shutil
    logger.info(f"Copying cache to local: {cache_path} → {local_cache}")
    t0 = time.time()
    shutil.copy2(cache_path, local_cache)
    logger.info(f"Copied in {time.time() - t0:.1f}s")
    return local_cache


def load_motion_data_partial(
    h5_path: str,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[torch.Tensor]]:
    """Read a subset of samples directly from HDF5 (no cache).

    This is useful for sweeps where you only need e.g. 10% of data and
    don't want to wait for the full cache to build.

    Args:
        h5_path: Path to the HDF5 motion file.
        max_samples: Maximum number of samples to read.  ``None`` = all.
        shuffle: If True, randomly sample; otherwise take the first N.
        seed: Random seed for reproducible subsampling.

    Returns (programs, obs) where obs is a list of (T, 72) tensors.
    """
    import random as _random

    t0 = time.time()
    programs: List[str] = []
    obs_list: List[torch.Tensor] = []

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_total = len(keys)

        if max_samples is not None and max_samples < n_total:
            if shuffle:
                rng = _random.Random(seed)
                selected = sorted(rng.sample(range(n_total), max_samples))
                keys = [keys[i] for i in selected]
            else:
                keys = keys[:max_samples]
            logger.info(
                f"Reading {len(keys)}/{n_total} samples "
                f"from {Path(h5_path).name} …"
            )
        else:
            logger.info(f"Reading all {n_total} samples from {Path(h5_path).name} …")

        for i, key in enumerate(keys):
            motion = f[key]["motion"][()]
            program = f[key].attrs["program"]
            programs.append(program)
            obs_list.append(torch.tensor(motion, dtype=torch.float32))
            if (i + 1) % 2000 == 0:
                logger.info(
                    f"  … {i + 1}/{len(keys)} samples read "
                    f"({time.time() - t0:.0f}s elapsed)"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Loaded {len(programs)} samples from {Path(h5_path).name} "
        f"in {elapsed:.1f}s"
    )
    return programs, obs_list


def load_motion_data(h5_path: str, local_cache_dir: str | None = None) -> Tuple[List[str], List[torch.Tensor]]:
    """Load motion data from HDF5, using a `.pt` cache when available.

    On first call the HDF5 is read sample-by-sample and a `.pt` cache is
    written alongside the file.  Subsequent calls load the cache in a
    single bulk read (~100× faster over NFS).

    Args:
        h5_path: Path to the HDF5 motion file.
        local_cache_dir: If set, copy the .pt cache to this local directory
            (e.g. ``/tmp/exact_cache``) and load from there.  Useful when
            the data lives on NFS.

    Returns (programs, obs) where obs is a list of (T, 72) tensors.
    """
    cache_path = _cache_path_for(h5_path)
    fingerprint = _h5_fingerprint(h5_path)

    # Ensure the .pt cache file exists on NFS
    if not cache_path.exists():
        preload_h5_to_cache(h5_path)

    # Optionally copy to local storage for faster reads
    if local_cache_dir:
        load_path = copy_cache_to_local(h5_path, local_cache_dir)
    else:
        load_path = cache_path

    # Load from cache
    t0 = time.time()
    try:
        payload = torch.load(load_path, weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            programs = payload["programs"]
            obs_stacked = payload["obs"]
            obs = list(obs_stacked)
            elapsed = time.time() - t0
            logger.info(
                f"Loaded {len(programs)} samples from cache "
                f"{load_path.name} in {elapsed:.1f}s"
            )
            return programs, obs
        else:
            logger.warning(
                "Cache fingerprint mismatch — HDF5 was modified. Rebuilding."
            )
    except Exception as e:
        logger.warning(f"Cache load failed ({e}), rebuilding from HDF5.")

    # Rebuild cache and retry
    preload_h5_to_cache(h5_path)
    if local_cache_dir:
        load_path = copy_cache_to_local(h5_path, local_cache_dir)
    payload = torch.load(load_path if local_cache_dir else cache_path, weights_only=False)
    return payload["programs"], list(payload["obs"])


class TrajectoryGenerationDataset(Dataset):
    """Dataset for program-to-pose pairs from HDF5 files.

    HDF5 format:
        motion_{i}/motion: numpy array of shape (T, 72)
            - 24 SMPL joints * 3 (x, y, z) world positions
            - Root joint at (0, 0, 0), other joints relative to world

    The text input is formatted as:
        "Program: [0,512]head.y(1.5)*rwrist.z(0.4);[512,1024]pelvis.y(0.8)<eos>"

    The motion prefix tokens are prepended by the model, so the effective
    input the LLM sees is:
        [motion_token_1] ... [motion_token_N] Program: [0,512]head.y(...) ...

    Supports fast `.pt` caching: on first load the HDF5 is read and a
    cache file is written alongside it.  Subsequent loads read the cache
    in a single bulk operation (~100× faster on NFS).

    You can also pass pre-loaded ``programs`` and ``obs`` to skip all I/O
    (useful for sharing data across sweep trials).
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        prompt_prefix: str = PROMPT_PREFIX,
        *,
        programs: Optional[List[str]] = None,
        obs: Optional[List[torch.Tensor]] = None,
    ):
        """Initialize dataset from HDF5 file.

        Args:
            path: Path to HDF5 file
            tokenizer: Tokenizer for encoding programs
            max_seq_length: Maximum sequence length for tokenization
            prompt_prefix: Text prefix before the program (e.g. "Program: ")
            programs: Pre-loaded program strings (skip I/O if provided with obs)
            obs: Pre-loaded motion tensors  (skip I/O if provided with programs)
        """
        self.path = path
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
        self.prompt_prefix = prompt_prefix

        if programs is not None and obs is not None:
            # Use pre-loaded data (zero I/O)
            self.programs = programs
            self.obs = obs
            logger.debug(f"Dataset from pre-loaded data: {len(self.programs)} samples")
        else:
            # Load from disk (with .pt cache acceleration)
            self.programs, self.obs = load_motion_data(path)

    def __len__(self) -> int:
        return len(self.programs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - input_ids: tokenized program with prompt (max_seq_length,)
                - attention_mask: attention mask (max_seq_length,)
                - obs: motion observations (T, 72)
        """
        program = self.programs[idx]
        obs = self.obs[idx]

        # Format: "Program: [0,512]head.y(1.5)*...<eos>"
        text = self.prompt_prefix + program + self.tokenizer.eos_token

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "obs": obs,
            "program": program,
        }
