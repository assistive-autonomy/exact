import pytest
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF


@pytest.fixture
def parser():
    return Lark.open(
        grammar_filename="exact/programs/grammar.lark", start="start", parser="earley"
    )


def test_parser_initialization(parser):
    assert parser is not None


def test_single_motion(parser):
    assert parser.parse("[0,100]head.z(1.0)") is not None
    assert parser.parse("[0,500]lknee.x(1.0)*rknee.y(0.5)") is not None
    assert parser.parse("[100,200]pelvis.z(0.5)*head.x(1.2)*lhand.y(0.8)") is not None


def test_multiple_motions(parser):
    assert parser.parse("[0,100]head.z(1.0),[100,200]pelvis.y(0.5)") is not None
    assert parser.parse("[0,50]lhand.x(0.5),[50,100]rhand.x(0.5),[100,150]head.z(1.0)") is not None


def test_negative_values(parser):
    assert parser.parse("[0,100]head.z(-1.0)") is not None
    assert parser.parse("[0,100]pelvis.x(-0.5)*head.y(-2.0)") is not None


def test_all_body_parts(parser):
    parts = [
        "pelvis", "torso", "spine", "chest", "neck", "head",
        "lhip", "lknee", "lankle", "ltoe",
        "rhip", "rknee", "rankle", "rtoe",
        "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
        "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
    ]
    for part in parts:
        assert parser.parse(f"[0,10]{part}.z(1.0)") is not None


def test_all_axes(parser):
    for axis in ["x", "y", "z"]:
        assert parser.parse(f"[0,10]head.{axis}(1.0)") is not None


@pytest.mark.parametrize(
    "invalid_input,exception_type",
    [
        ("head.z(1.0)", UnexpectedCharacters),  # missing time interval
        ("[0,100]unknown.z(1.0)", UnexpectedCharacters),  # invalid body part
        ("[0,100]head.w(1.0)", UnexpectedCharacters),  # invalid axis
        ("[0,100]head.z(1.0)*", UnexpectedEOF),  # trailing *
        ("[0,100]*head.z(1.0)", UnexpectedCharacters),  # leading *
        ("[0,100]head.z(1)", UnexpectedCharacters),  # missing decimal
        ("[0,100]head.z(1.0) [100,200]pelvis.y(0.5)", UnexpectedCharacters),  # missing comma
    ],
)
def test_invalid_grammar(parser, invalid_input, exception_type):
    with pytest.raises(exception_type):
        parser.parse(invalid_input)
