import pytest
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF


@pytest.fixture
def parser():
    return Lark.open(grammar_filename="exact/programs/grammar.lark", start="start", parser="earley")


def test_parser_initialization(parser):
    """Test that the parser initializes correctly."""
    assert parser is not None


def test_correct_grammar(parser):
    """Test that valid grammar expressions are parsed correctly."""
    assert parser.parse("lknee(1.0)") is not None
    assert parser.parse("lknee(1.0)*rknee(0.5)") is not None
    assert parser.parse("head(3.0)") is not None
    assert parser.parse("lhip(-1.2)") is not None  # Test negative number
    assert parser.parse("rshoulder(0.0)*lwrist(-0.5)*head(1.0)") is not None  # Test multiple parts with zero and negative


@pytest.mark.parametrize("invalid_input,exception_type", [
    ("unknown(1.0)", UnexpectedCharacters),  # Invalid body part
    ("lknee(-1.0", UnexpectedEOF),  # Missing closing parenthesis
    ("lknee1.0)", UnexpectedCharacters),  # Missing opening parenthesis
    ("lknee(1.0)*", UnexpectedEOF),  # Incomplete expression after *
    ("*lknee(1.0)", UnexpectedCharacters),  # Invalid start with *
    ("lknee(1.0)*rknee", UnexpectedEOF),  # Incomplete second part
    ("lknee(1.0) rfoot(0.5)", UnexpectedCharacters),  # Space instead of *
    ("lknee(1.0)*rknee(0.5)+head(3.5)", UnexpectedCharacters),  # Invalid operator +
    ("lknee(1.0.5)", UnexpectedCharacters),  # Invalid number format
    ("lknee(1a.0)", UnexpectedCharacters),  # Invalid character in number
])
def test_incorrect_grammar(parser, invalid_input, exception_type):
    """Test that invalid grammar expressions raise the expected exceptions."""
    with pytest.raises(exception_type):
        parser.parse(invalid_input)