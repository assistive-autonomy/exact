import hashlib
import random
import re
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


# ── Variable-length crop augmentation ─────────────────────────────────────────
# Regex to parse a single program segment: [start,end]sensors
_SEGMENT_RE = re.compile(r"\[(\d+),(\d+)\]([^\[;]+)")


def parse_program_segments(program: str) -> list[tuple[int, int, str]]:
    """Parse a program string into (start_frame, end_frame, sensors) triples.

    Example::

        >>> parse_program_segments("[0,130]chest.z(0.5);[130,260]rwrist.z(1.4)")
        [(0, 130, 'chest.z(0.5)'), (130, 260, 'rwrist.z(1.4)')]
    """
    segments: list[tuple[int, int, str]] = []
    for m in _SEGMENT_RE.finditer(program):
        segments.append((int(m.group(1)), int(m.group(2)), m.group(3).strip()))
    return segments


def crop_program_and_motion(
    program: str,
    obs: torch.Tensor,
    rng: random.Random,
    min_segments: int = 1,
    min_crop_frames: int = 32,
) -> tuple[str, torch.Tensor]:
    """Crop a contiguous subset of program segments and corresponding motion.

    This augmentation:
      1. Parses ``program`` to find temporal segment boundaries.
      2. Picks a random *contiguous* window of segments (never cuts within
         a segment — respects program boundaries exactly).
      3. Slices the corresponding frames from ``obs``.
      4. Renumbers segment frame indices to start from 0.
      5. Zero-pads the cropped motion back to the original length so that
         the collate function (``torch.stack``) and encoder see a
         fixed-size tensor (e.g. 1024 frames).

    This teaches the encoder that "real motion in frames 0–N then zeros"
    should produce a program whose frame numbers go up to N, **not** 1024.
    At inference time (where short ESK/HA12 segments are also padded to
    1024), the encoder now outputs representations the LLM can correctly
    decode into appropriately-ranged programs.

    Args:
        program: Full program string (e.g. ``"[0,130]...;[130,260]..."``)
        obs: Motion tensor of shape ``(T, 72)``
        rng: Seeded ``random.Random`` instance for reproducibility
        min_segments: Minimum number of segments to keep (default 1)
        min_crop_frames: Skip augmentation if the crop would be shorter
            than this many frames (avoids degenerate tiny snippets)

    Returns:
        ``(new_program, new_obs)`` where ``new_obs.shape == obs.shape``
        (zero-padded) and ``new_program`` has renumbered frame indices.
    """
    segments = parse_program_segments(program)
    if len(segments) <= 1:
        # Single-segment programs cannot be cropped further
        return program, obs

    num_segments = len(segments)

    # Pick how many contiguous segments to keep
    keep_count = rng.randint(min_segments, num_segments - 1)
    # Where to start the window (ensure at least 1 segment cropped)
    max_start = num_segments - keep_count
    start_idx = rng.randint(0, max_start)

    selected = segments[start_idx : start_idx + keep_count]

    # Frame range for cropped motion
    frame_start = selected[0][0]
    frame_end = selected[-1][1]
    crop_len = frame_end - frame_start

    # Skip if the crop is too short to be useful
    if crop_len < min_crop_frames:
        return program, obs

    # Crop the motion (clone to avoid mutating the original)
    T_orig = obs.shape[0]
    frame_end_clamped = min(frame_end, T_orig)
    cropped = obs[frame_start:frame_end_clamped].clone()

    # Renumber segment frames to start at 0
    new_parts: list[str] = []
    for seg_start, seg_end, sensors in selected:
        new_parts.append(f"[{seg_start - frame_start},{seg_end - frame_start}]{sensors}")
    new_program = ";".join(new_parts)

    # Zero-pad back to original length
    if cropped.shape[0] < T_orig:
        padding = torch.zeros(
            T_orig - cropped.shape[0], obs.shape[1], dtype=obs.dtype
        )
        cropped = torch.cat([cropped, padding], dim=0)

    return new_program, cropped


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
        augment_crop_prob: float = 0.0,
        augment_min_segments: int = 1,
        augment_min_crop_frames: int = 32,
    ):
        """Initialize dataset from HDF5 file.

        Args:
            path: Path to HDF5 file
            tokenizer: Tokenizer for encoding programs
            max_seq_length: Maximum sequence length for tokenization
            prompt_prefix: Text prefix before the program (e.g. "Program: ")
            programs: Pre-loaded program strings (skip I/O if provided with obs)
            obs: Pre-loaded motion tensors  (skip I/O if provided with programs)
            augment_crop_prob: Probability of applying variable-length crop
                augmentation per sample (0.0 = disabled, 0.5 = 50% of samples).
                When applied, a random contiguous subset of program segments
                is selected and the motion is cropped + zero-padded accordingly.
                This teaches the encoder to handle variable-length inputs.
            augment_min_segments: Minimum number of segments to keep when
                cropping (default 1).
            augment_min_crop_frames: Skip cropping if result would be shorter
                than this many frames (default 32).
        """
        self.path = path
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
        self.prompt_prefix = prompt_prefix
        self.augment_crop_prob = augment_crop_prob
        self.augment_min_segments = augment_min_segments
        self.augment_min_crop_frames = augment_min_crop_frames

        if programs is not None and obs is not None:
            # Use pre-loaded data (zero I/O)
            self.programs = programs
            self.obs = obs
            logger.debug(f"Dataset from pre-loaded data: {len(self.programs)} samples")
        else:
            # Load from disk (with .pt cache acceleration)
            self.programs, self.obs = load_motion_data(path)

        # Training mode flag: augmentations are only applied when True.
        # Set to True by default; callers (e.g. eval datasets) can set False.
        self.training_mode = True

        if self.augment_crop_prob > 0:
            # Count how many multi-segment programs can benefit from cropping
            n_multi = sum(1 for p in self.programs if ";" in p)
            logger.info(
                f"Variable-length crop augmentation enabled: "
                f"prob={self.augment_crop_prob}, min_segs={self.augment_min_segments}, "
                f"min_frames={self.augment_min_crop_frames}, "
                f"eligible={n_multi}/{len(self.programs)} multi-segment programs"
            )

    def __len__(self) -> int:
        return len(self.programs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample, optionally with variable-length crop augmentation.

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

        # ── Variable-length crop augmentation ─────────────────────────────
        # Randomly crop a contiguous subset of program segments and the
        # corresponding motion frames.  The cropped motion is zero-padded
        # back to the original length (e.g. 1024) and the program frame
        # numbers are renumbered from 0.  This teaches the encoder to
        # handle variable-length inputs at inference time.
        if self.augment_crop_prob > 0 and self.training_mode:
            # Use a worker-safe random instance seeded per-call
            rng = random.Random()
            if rng.random() < self.augment_crop_prob:
                program, obs = crop_program_and_motion(
                    program,
                    obs,
                    rng=rng,
                    min_segments=self.augment_min_segments,
                    min_crop_frames=self.augment_min_crop_frames,
                )

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
