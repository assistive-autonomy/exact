import re
from pathlib import Path

from lark import Lark
from lark.exceptions import LarkError
from loguru import logger
from syncode import SyncodeLogitsProcessor, Grammar

DEFAULT_GRAMMAR_PATH = Path(__file__).parent.parent / "programs" / "grammar.lark"

# Precompile regex patterns for program validation/repair
_PROGRAM_PATTERN = re.compile(
    r'\[(\d+),(\d+)\]'  # [start,end]
    r'((?:[a-z]+\.[xyz]\(-?[\d.]+\))(?:\*[a-z]+\.[xyz]\(-?[\d.]+\))*)'  # sensors
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


def create_grammar_processor(
    tokenizer,
    grammar_path: str | Path | None = None,
) -> SyncodeLogitsProcessor:
    """Create a SynCode logits processor for grammar-constrained decoding.

    Args:
        tokenizer: HuggingFace tokenizer
        grammar_path: Path to .lark grammar file (default: exact/programs/grammar.lark)

    Returns:
        SyncodeLogitsProcessor configured with the grammar
    """
    if grammar_path is None:
        grammar_path = DEFAULT_GRAMMAR_PATH

    grammar_str = Path(grammar_path).read_text()
    grammar = Grammar(grammar_str)

    processor = SyncodeLogitsProcessor(
        grammar=grammar,
        tokenizer=tokenizer,
        parse_output_only=True,
    )

    return processor


def post_process_program(
    program: str,
    grammar_path: str | Path | None = None,
    repair: bool = True,
) -> tuple[str, bool]:
    """Post-process a generated program to ensure validity.

    Args:
        program: Generated program string
        grammar_path: Path to .lark grammar file
        repair: Whether to attempt repair if invalid

    Returns:
        Tuple of (processed_program, is_valid)
    """
    # First, try the program as-is
    if validate_program(program, grammar_path):
        return program, True

    if not repair:
        return program, False

    # Try to repair
    repaired = repair_program(program)
    if validate_program(repaired, grammar_path):
        logger.debug(f"Repaired program: '{program}' -> '{repaired}'")
        return repaired, True

    # Try to extract valid prefix
    valid_prefix = extract_valid_prefix(program, grammar_path)
    if valid_prefix:
        logger.warning(
            f"Extracted valid prefix from malformed program: '{program}' -> '{valid_prefix}'"
        )
        return valid_prefix, True

    # Return original with invalid flag
    logger.warning(f"Could not repair program: '{program}'")
    return program, False
