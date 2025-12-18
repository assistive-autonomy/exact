from pathlib import Path

from syncode import SyncodeLogitsProcessor, Grammar

DEFAULT_GRAMMAR_PATH = Path(__file__).parent.parent / "programs" / "grammar.lark"


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

    return SyncodeLogitsProcessor(
        grammar=grammar,
        tokenizer=tokenizer,
        parse_output_only=True,
    )