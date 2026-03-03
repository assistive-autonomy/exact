import re
from pathlib import Path

from lark import Lark
from lark.exceptions import LarkError
from loguru import logger

DEFAULT_GRAMMAR_PATH = Path(__file__).parent.parent / "programs" / "grammar.lark"

# Precompile regex patterns for program validation/repair
_ALLOWED_CHARS_PATTERN = re.compile(r"^[\w\[\];,\*\.\(\)\-]+$", re.IGNORECASE)
_SEGMENT_PATTERN = re.compile(
    r"\[(\d+),(\d+)\]"  # [start,end]
    r"(?:[a-z]+\.[xyz]\(-?\d+(?:\.\d+)?\))"  # first sensor
    r"(?:\*[a-z]+\.[xyz]\(-?\d+(?:\.\d+)?\))*",  # optional additional sensors
    re.IGNORECASE,
)
_MULTISEGMENT_PATTERN = re.compile(
    rf"^{_SEGMENT_PATTERN.pattern}(?:;{_SEGMENT_PATTERN.pattern})*$", re.IGNORECASE
)


def get_grammar_parser(grammar_path: str | Path | None = None) -> Lark:
    """Get a Lark parser for the grammar.

    Args:
        grammar_path: Path to .lark grammar file (default: exact/programs/grammar.lark)

    Returns:
        Lark parser instance
    """
    if grammar_path is None:
        grammar_path = DEFAULT_GRAMMAR_PATH

    grammar_str = Path(grammar_path).read_text()
    return Lark(grammar_str, parser="lalr")


def validate_program(program: str, grammar_path: str | Path | None = None) -> bool:
    """Validate that a program string conforms to the grammar.

    Args:
        program: Program string to validate
        grammar_path: Path to .lark grammar file

    Returns:
        True if valid, False otherwise
    """
    program = program.strip()

    # Lightweight screen to avoid expensive parsing on obviously malformed strings
    if not program:
        return False
    if not _ALLOWED_CHARS_PATTERN.match(program):
        logger.debug("Program failed char whitelist check")
        return False
    if program.count("[") != program.count("]") or program.count("(") != program.count(")"):
        logger.debug("Program failed bracket/paren balance check")
        return False
    if not _MULTISEGMENT_PATTERN.match(program):
        logger.debug("Program failed coarse segment pattern check")
        return False

    try:
        parser = get_grammar_parser(grammar_path)
        parser.parse(program)
        return True
    except LarkError:
        return False


def repair_program(program: str) -> str:
    """Attempt to repair a malformed program string.

    Common fixes:
    - Remove leading/trailing whitespace
    - Fix spacing around brackets and operators
    - Remove invalid characters

    Args:
        program: Potentially malformed program string

    Returns:
        Repaired program string (may still be invalid)
    """
    # Strip whitespace
    program = program.strip()

    # Remove any spaces around structural characters
    program = re.sub(r'\s*\[\s*', '[', program)
    program = re.sub(r'\s*\]\s*', ']', program)
    program = re.sub(r'\s*\(\s*', '(', program)
    program = re.sub(r'\s*\)\s*', ')', program)
    program = re.sub(r'\s*,\s*', ',', program)
    program = re.sub(r'\s*;\s*', ';', program)
    program = re.sub(r'\s*\*\s*', '*', program)
    program = re.sub(r'\s*\.\s*', '.', program)

    # Remove any remaining whitespace
    program = re.sub(r'\s+', '', program)

    return program


def extract_valid_prefix(program: str, grammar_path: str | Path | None = None) -> str:
    """Extract the longest valid prefix from a program.

    Useful for recovering partial valid programs from malformed outputs.

    Args:
        program: Potentially malformed program string
        grammar_path: Path to .lark grammar file

    Returns:
        Longest valid prefix (empty string if none found)
    """
    # Try the full program first
    if validate_program(program, grammar_path):
        return program

    # Try to find valid motion segments
    repaired = repair_program(program)

    # Split by semicolon and try to validate progressively
    segments = repaired.split(';')
    valid_segments = []

    for segment in segments:
        test_program = ';'.join(valid_segments + [segment])
        if validate_program(test_program, grammar_path):
            valid_segments.append(segment)
        else:
            break

    return ';'.join(valid_segments) if valid_segments else ""


# ── Joints and axes from the grammar ─────────────────────────────────────────

_JOINTS = [
    "pelvis", "torso", "spine", "chest", "neck", "head",
    "lhip", "lknee", "lankle", "ltoe",
    "rhip", "rknee", "rankle", "rtoe",
    "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
    "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
]
_AXES = ["x", "y", "z"]


class ExActGrammarProcessor:
    """Fast, exact grammar-constrained logits processor for ExAct programs.

    Instead of relying on SynCode's generic DFA-based approach (which can
    fail with the incremental Lark parser on our grammar), this processor
    implements a **hand-crafted state machine** tailored to the ExAct
    grammar::

        start:  motion (";" motion)*
        motion: "[" NUMBER "," NUMBER "]" sensor ("*" sensor)*
        sensor: JOINT "." AXIS "(" NUMBER ")"

    The state machine tracks generation character-by-character and builds a
    token-level accept mask by checking which vocabulary tokens represent a
    valid continuation from the current state.

    Masks are computed once per (state, partial) pair and cached — after a
    brief warmup the per-token overhead is a single dict lookup + mask fill.

    For efficient first-time computation, vocabulary tokens are pre-grouped
    by first character, so only tokens whose first character is in the
    current valid-continuation set need to be checked (typically <5% of the
    full 150K vocabulary).
    """

    # ── Grammar states ────────────────────────────────────────────────────
    S_START = "start"               # Expect "["
    S_FRAME1 = "frame1"             # Inside first frame number
    S_FRAME2 = "frame2"             # Inside second frame number
    S_JOINT = "joint"               # Expect/reading joint name
    S_AXIS = "axis"                 # Expect axis character
    S_OPEN_PAREN = "open_paren"     # Expect "("
    S_VALUE = "value"               # Inside sensor value number
    S_AFTER_SENSOR = "after_sensor" # Expect "*", ";", or EOS

    def __init__(self, tokenizer, eos_token_id: int | None = None):
        import torch
        from collections import defaultdict

        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self._vocab_size = len(tokenizer)  # includes added tokens
        self._torch = torch
        self._start_from: int | None = None

        # Pre-compute single-token decodings and group by first character
        self._token_strings: list[str] = []
        self._tokens_by_first_char: dict[str, list[int]] = defaultdict(list)

        for i in range(self._vocab_size):
            try:
                s = tokenizer.decode([i])
            except Exception:
                s = ""
            self._token_strings.append(s)
            if s:
                self._tokens_by_first_char[s[0]].append(i)

        # Cache: state_key -> accept_mask tensor (on CPU for reuse)
        self._mask_cache: dict[str, "torch.Tensor"] = {}

        logger.info(
            f"ExActGrammarProcessor initialised: vocab_size={self._vocab_size}, "
            f"unique_first_chars={len(self._tokens_by_first_char)}"
        )

    # ── Public interface ──────────────────────────────────────────────────

    def reset(self):
        """Reset per-generation state (called before each generation)."""
        self._start_from = None
        # _mask_cache is intentionally preserved across generations

    def __call__(self, input_ids, scores):
        """HuggingFace LogitsProcessor interface: mask invalid tokens."""
        torch = self._torch

        # On first call, record the prompt length so we can decode only
        # the generated portion on subsequent calls.
        if self._start_from is None:
            self._start_from = input_ids.shape[1]

        for idx in range(input_ids.shape[0]):
            gen_ids = input_ids[idx, self._start_from :].tolist()
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            state, partial = self._parse_state(generated_text)
            cache_key = f"{state}|{partial}"

            if cache_key not in self._mask_cache:
                self._mask_cache[cache_key] = self._build_mask(state, partial)

            mask = self._mask_cache[cache_key].to(scores.device)

            # Apply mask — only keep tokens where mask is True
            if mask.shape[0] < scores.shape[1]:
                # Score tensor is wider than our mask (e.g. added tokens)
                full_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
                full_mask[: mask.shape[0]] = mask
                if self.eos_token_id >= mask.shape[0]:
                    # also allow EOS if it's outside base vocab range
                    valid = self._valid_chars(state, partial)
                    if "\0" in valid:
                        full_mask[self.eos_token_id] = True
                scores[idx] = scores[idx].masked_fill(~full_mask, -float("inf"))
            else:
                scores[idx] = scores[idx].masked_fill(~mask[: scores.shape[1]], -float("inf"))

        return scores

    # ── State machine ─────────────────────────────────────────────────────

    def _parse_state(self, text: str) -> tuple[str, str]:
        """Walk *text* through the grammar and return ``(state, partial)``.

        ``partial`` is the unfinished token fragment in the current state
        (e.g. partial joint name ``"he"`` while in ``S_JOINT``).
        """
        if not text:
            return (self.S_START, "")

        i = 0
        n = len(text)

        while i < n:
            # ── Expect "[" ────────────────────────────────────────────
            if text[i] != "[":
                return (self.S_START, "")
            i += 1

            # ── Frame 1 (NUMBER) then "," ─────────────────────────────
            start = i
            while i < n and text[i] not in ",]":
                i += 1
            if i == n:
                return (self.S_FRAME1, text[start:])
            if text[i] != ",":
                # malformed — just return what we have
                return (self.S_FRAME1, text[start:])
            i += 1  # skip ","

            # ── Frame 2 (NUMBER) then "]" ─────────────────────────────
            start = i
            while i < n and text[i] != "]":
                i += 1
            if i == n:
                return (self.S_FRAME2, text[start:])
            i += 1  # skip "]"

            # ── Sensors: JOINT.AXIS(VALUE) separated by "*" ──────────
            while True:
                if i >= n:
                    return (self.S_JOINT, "")

                # Read JOINT (alpha chars)
                start = i
                while i < n and text[i].isalpha():
                    i += 1
                if i == n:
                    return (self.S_JOINT, text[start:])

                # Expect "."
                if text[i] != ".":
                    return (self.S_JOINT, text[start:])
                i += 1

                # Read AXIS (single char)
                if i >= n:
                    return (self.S_AXIS, "")
                i += 1  # skip axis

                # Expect "("
                if i >= n:
                    return (self.S_OPEN_PAREN, "")
                if text[i] != "(":
                    return (self.S_OPEN_PAREN, "")
                i += 1

                # Read VALUE (everything up to ")")
                start = i
                while i < n and text[i] != ")":
                    i += 1
                if i == n:
                    return (self.S_VALUE, text[start:])
                i += 1  # skip ")"

                # After sensor: check for "*", ";" or end
                if i >= n:
                    return (self.S_AFTER_SENSOR, "")

                if text[i] == "*":
                    i += 1
                    continue  # next sensor
                elif text[i] == ";":
                    i += 1
                    break  # next motion segment
                else:
                    return (self.S_AFTER_SENSOR, "")

        # Consumed all text, ended at segment boundary
        return (self.S_AFTER_SENSOR, "") if i == n else (self.S_START, "")

    def _valid_chars(self, state: str, partial: str) -> set[str]:
        """Characters that can validly appear next in *state* with *partial*."""
        if state == self.S_START:
            return {"["}

        if state == self.S_FRAME1 or state == self.S_FRAME2:
            # Frame indices are non-negative integers only (no '.' or '-')
            chars = set("0123456789")
            if partial:  # at least 1 digit → can terminate
                if state == self.S_FRAME1:
                    chars.add(",")
                else:
                    chars.add("]")
            return chars

        if state == self.S_JOINT:
            chars: set[str] = set()
            for j in _JOINTS:
                if j.startswith(partial):
                    if len(partial) < len(j):
                        chars.add(j[len(partial)])
                    else:
                        chars.add(".")  # joint complete
            return chars

        if state == self.S_AXIS:
            return set(_AXES)

        if state == self.S_OPEN_PAREN:
            return {"("}

        if state == self.S_VALUE:
            chars = set("0123456789.")
            if not partial:
                chars.add("-")
            if partial:
                chars.add(")")
            return chars

        if state == self.S_AFTER_SENSOR:
            return {"*", ";", "\0"}  # \0 = EOS sentinel

        return set()

    # ── Token simulation ──────────────────────────────────────────────────

    def _advance(self, state: str, partial: str, ch: str) -> tuple[str, str]:
        """Advance the state machine by one character.  Returns new (state, partial)."""
        if state == self.S_START and ch == "[":
            return (self.S_FRAME1, "")
        if state == self.S_FRAME1:
            return (self.S_FRAME2, "") if ch == "," else (self.S_FRAME1, partial + ch)
        if state == self.S_FRAME2:
            return (self.S_JOINT, "") if ch == "]" else (self.S_FRAME2, partial + ch)
        if state == self.S_JOINT:
            return (self.S_AXIS, "") if ch == "." else (self.S_JOINT, partial + ch)
        if state == self.S_AXIS:
            return (self.S_OPEN_PAREN, "")
        if state == self.S_OPEN_PAREN and ch == "(":
            return (self.S_VALUE, "")
        if state == self.S_VALUE:
            return (self.S_AFTER_SENSOR, "") if ch == ")" else (self.S_VALUE, partial + ch)
        if state == self.S_AFTER_SENSOR:
            if ch == "*":
                return (self.S_JOINT, "")
            if ch == ";":
                return (self.S_START, "")
        # fallback (should not happen for valid chars)
        return (state, partial)

    def _token_ok(self, tok_str: str, state: str, partial: str) -> bool:
        """Return True if *tok_str* is a valid continuation from (state, partial)."""
        s, p = state, partial
        for ch in tok_str:
            if ch not in self._valid_chars(s, p):
                return False
            s, p = self._advance(s, p, ch)
        return True

    # ── Mask construction ─────────────────────────────────────────────────

    def _build_mask(self, state: str, partial: str) -> "torch.Tensor":
        torch = self._torch
        mask = torch.zeros(self._vocab_size, dtype=torch.bool)

        valid_first = self._valid_chars(state, partial)

        # Handle EOS
        if "\0" in valid_first:
            mask[self.eos_token_id] = True
            valid_first = valid_first - {"\0"}

        # Only iterate over tokens whose first character is valid
        for ch in valid_first:
            for tok_id in self._tokens_by_first_char.get(ch, []):
                if tok_id == self.eos_token_id:
                    continue
                if self._token_ok(self._token_strings[tok_id], state, partial):
                    mask[tok_id] = True

        n_allowed = int(mask.sum().item())
        logger.debug(f"Grammar mask [{state}|{partial!r}]: {n_allowed} tokens allowed")
        return mask


def create_grammar_processor(
    tokenizer,
    grammar_path: str | Path | None = None,
    num_samples: int = 1,
) -> ExActGrammarProcessor:
    """Create a grammar-constrained logits processor for ExAct programs.

    Uses a hand-crafted state machine that exactly matches the ExAct
    grammar, avoiding the issues with SynCode's generic DFA approach
    (whitespace flooding, incremental parser failures).

    The processor:
    - Blocks whitespace tokens entirely (no ``%ignore`` leaking)
    - Correctly enforces ``[NUMBER,NUMBER]JOINT.AXIS(NUMBER)`` structure
    - Allows EOS only after a complete sensor (not mid-segment)
    - Caches accept masks per state for O(1) lookup after warmup

    Args:
        tokenizer: HuggingFace tokenizer
        grammar_path: Unused (grammar is hardcoded for reliability)
        num_samples: Unused (processor handles any batch size)

    Returns:
        ExActGrammarProcessor configured for the ExAct grammar
    """
    return ExActGrammarProcessor(tokenizer)


def recover_program(
    program: str,
    max_frame: int = 1024,
    grammar_path: str | Path | None = None,
) -> str:
    """Attempt to recover a valid program from malformed model output.

    The model often produces sensor groups (joint.axis(value)*...) separated
    by semicolons but omits the required [start,end] frame brackets. This
    function detects such patterns and injects evenly-spaced temporal brackets.

    It also handles outputs where a leading numeric value (e.g. ``1.06;``)
    precedes the first sensor group (strip the leading value) and the
    ``joint:value`` format (converted to ``joint.x(value)``).

    Args:
        program: Malformed program string
        max_frame: Maximum frame number for injected brackets
        grammar_path: Path to .lark grammar file

    Returns:
        Recovered program string (empty string if unrecoverable)
    """
    if not program:
        return ""

    program = repair_program(program)

    # Already valid? Nothing to do.
    if validate_program(program, grammar_path):
        return program

    # ---- Pattern A: sensor groups without brackets ----
    # e.g. "1.06;lankle.x(0.5)*rthorax.x(0.3);rshoulder.x(0.4)*lhand.y(0.4)"
    # Strategy: split on ";", drop non-sensor pieces, wrap survivors in [start,end]
    # Also handle comma-separated variants (e.g. "127.0,lelbow.x(1.3)*...")

    # Valid JOINT names from the grammar
    _JOINTS = {
        "pelvis", "torso", "spine", "chest", "neck", "head",
        "lhip", "lknee", "lankle", "ltoe",
        "rhip", "rknee", "rankle", "rtoe",
        "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
        "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
    }

    # Regex for a single sensor: joint.axis(number)
    _SENSOR_RE = re.compile(
        r"^([a-z]+)\.([xyz])\((-?\d+(?:\.\d+)?)\)$", re.IGNORECASE
    )
    # Regex for a sensor group: sensor(*sensor)*
    _GROUP_RE = re.compile(
        r"^[a-z]+\.[xyz]\(-?\d+(?:\.\d+)?\)"
        r"(?:\*[a-z]+\.[xyz]\(-?\d+(?:\.\d+)?\))*$",
        re.IGNORECASE,
    )
    # Regex for colon-format: joint:number
    _COLON_RE = re.compile(
        r"^([a-z]+):(-?\d+(?:\.\d+)?)$", re.IGNORECASE
    )

    def _is_sensor_group(s: str) -> bool:
        """Check if string is a valid sensor group (without brackets)."""
        return bool(_GROUP_RE.match(s))

    def _sensors_have_valid_joints(s: str) -> bool:
        """Check that all joint names in a sensor group are valid."""
        for part in s.split("*"):
            m = _SENSOR_RE.match(part)
            if not m:
                return False
            if m.group(1).lower() not in _JOINTS:
                return False
        return True

    def _convert_colon_segment(s: str) -> str | None:
        """Convert 'joint:value' to 'joint.x(value)' if joint is valid."""
        m = _COLON_RE.match(s)
        if m and m.group(1).lower() in _JOINTS:
            return f"{m.group(1).lower()}.x({m.group(2)})"
        return None

    raw_parts = re.split(r"[;,]", program)
    sensor_groups: list[str] = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # Strip leading numeric garbage (e.g. "1.06" or "0.018vhip..." → "vhip...")
        stripped = re.sub(r"^-?\d+(?:\.\d+)?", "", part)
        if not stripped:
            continue  # Pure number, skip

        # Try as sensor group directly
        if _is_sensor_group(stripped) and _sensors_have_valid_joints(stripped):
            sensor_groups.append(stripped)
            continue

        # Try colon format: "head:0.92"
        converted = _convert_colon_segment(part)
        if converted is not None:
            sensor_groups.append(converted)
            continue

        # Try colon format after stripping leading number
        converted = _convert_colon_segment(stripped)
        if converted is not None:
            sensor_groups.append(converted)
            continue

        # Try the original part (might have brackets already embedded)
        if _is_sensor_group(part) and _sensors_have_valid_joints(part):
            sensor_groups.append(part)

    if not sensor_groups:
        return ""

    # Inject evenly-spaced temporal brackets
    n = len(sensor_groups)
    step = max_frame // n
    segments = []
    for i, group in enumerate(sensor_groups):
        start = i * step
        end = (i + 1) * step if i < n - 1 else max_frame
        segments.append(f"[{start},{end}]{group}")

    recovered = ";".join(segments)

    # Validate the assembled program
    if validate_program(recovered, grammar_path):
        logger.info(f"Recovered program with {n} segments from malformed output")
        return recovered

    # If full program fails, try extracting valid prefix
    valid_prefix = extract_valid_prefix(recovered, grammar_path)
    if valid_prefix:
        logger.info(
            f"Partially recovered program: {len(valid_prefix.split(';'))} of {n} segments"
        )
        return valid_prefix

    return ""


def constrain_frame_numbers(
    program: str,
    max_frame: int = 1024,
) -> str:
    """Constrain frame numbers in a program to valid range [0, max_frame].
    
    Fixes common issues:
    - Frame numbers > max_frame are scaled/clamped
    - Negative frame numbers are set to 0
    - Start frame > end frame are swapped
    
    Args:
        program: Program string with potentially invalid frame numbers
        max_frame: Maximum allowed frame number (default 1024)
        
    Returns:
        Program with constrained frame numbers
    """
    if not program:
        return program
    
    def fix_segment(match):
        start = int(match.group(1))
        end = int(match.group(2))
        rest = match.group(3)  # sensors part
        
        # Check if frames are way out of range (likely model error)
        if start > max_frame or end > max_frame:
            # Scale down proportionally if both are large
            if start > max_frame and end > max_frame:
                scale = max_frame / max(start, end)
                start = int(start * scale)
                end = int(end * scale)
            else:
                # Clamp to max_frame
                start = min(start, max_frame)
                end = min(end, max_frame)
        
        # Ensure non-negative
        start = max(0, start)
        end = max(0, end)
        
        # Ensure start <= end
        if start > end:
            start, end = end, start
        
        # Ensure minimum segment length
        if start == end:
            end = min(start + 1, max_frame)
        
        return f"[{start},{end}]{rest}"
    
    # Pattern to match [start,end]sensors
    segment_pattern = re.compile(
        r"\[(\d+),(\d+)\]([^;\[]+)"
    )
    
    return segment_pattern.sub(fix_segment, program)


def post_process_program(
    program: str,
    grammar_path: str | Path | None = None,
    repair: bool = True,
    max_frame: int = 1024,
) -> tuple[str, bool]:
    """Post-process a generated program to ensure validity.

    Args:
        program: Generated program string
        grammar_path: Path to .lark grammar file
        repair: Whether to attempt repair if invalid
        max_frame: Maximum allowed frame number

    Returns:
        Tuple of (processed_program, is_valid)
    """
    program = program.strip()
    
    # First, constrain frame numbers to valid range
    program = constrain_frame_numbers(program, max_frame)

    # Quick screen before heavier parsing
    if not validate_program(program, grammar_path):
        if not repair:
            return program, False

        # Try to repair and screen again
        repaired = repair_program(program)

        if validate_program(repaired, grammar_path):
            logger.debug(f"Repaired program: '{program}' -> '{repaired}'")
            return repaired, True

        # Try to extract valid prefix
        valid_prefix = extract_valid_prefix(repaired, grammar_path)
        if valid_prefix:
            logger.warning(
                f"Extracted valid prefix from malformed program: '{program}' -> '{valid_prefix}'"
            )
            return valid_prefix, True

        # Try deep recovery (inject brackets, convert colon-format, etc.)
        recovered = recover_program(repaired, max_frame=max_frame, grammar_path=grammar_path)
        if recovered and validate_program(recovered, grammar_path):
            logger.info(
                f"Recovered program from malformed output: '{program[:80]}...' -> '{recovered[:80]}...'"
            )
            return recovered, True

        # Return original with invalid flag
        logger.warning(f"Could not repair program: '{program}'")
        return program, False

    # First, try the program as-is
    return program, True
