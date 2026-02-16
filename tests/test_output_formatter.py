import pytest
from gpt_token_explorer import OutputFormatter

def test_output_formatter_can_be_created():
    """OutputFormatter can be instantiated"""
    formatter = OutputFormatter()
    assert formatter is not None

def test_probability_bar_with_zero():
    """Probability bar with 0.0 shows all empty"""
    bar = OutputFormatter.probability_bar(0.0, max_width=10)
    assert bar == '░' * 10

def test_probability_bar_with_one():
    """Probability bar with 1.0 shows all filled"""
    bar = OutputFormatter.probability_bar(1.0, max_width=10)
    assert bar == '█' * 10

def test_probability_bar_with_half():
    """Probability bar with 0.5 shows half filled"""
    bar = OutputFormatter.probability_bar(0.5, max_width=10)
    assert bar == '█' * 5 + '░' * 5

def test_probability_bar_rejects_negative():
    """Probability bar rejects negative values"""
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        OutputFormatter.probability_bar(-0.1)

def test_probability_bar_rejects_greater_than_one():
    """Probability bar rejects values > 1"""
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        OutputFormatter.probability_bar(1.5)

def test_probability_color_returns_green_for_high():
    """Probability color returns green for > 0.5"""
    color = OutputFormatter.probability_color(0.8)
    assert color == OutputFormatter.GREEN

def test_probability_color_returns_yellow_for_medium():
    """Probability color returns yellow for 0.1 < p <= 0.5"""
    color = OutputFormatter.probability_color(0.3)
    assert color == OutputFormatter.YELLOW

def test_probability_color_returns_red_for_low():
    """Probability color returns red for <= 0.1"""
    color = OutputFormatter.probability_color(0.05)
    assert color == OutputFormatter.RED

def test_probability_color_boundary_at_point_five():
    """Probability color boundary at 0.5"""
    assert OutputFormatter.probability_color(0.51) == OutputFormatter.GREEN
    assert OutputFormatter.probability_color(0.5) == OutputFormatter.YELLOW

def test_probability_color_boundary_at_point_one():
    """Probability color boundary at 0.1"""
    assert OutputFormatter.probability_color(0.11) == OutputFormatter.YELLOW
    assert OutputFormatter.probability_color(0.1) == OutputFormatter.RED

def test_probability_color_rejects_negative():
    """Probability color rejects negative values"""
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        OutputFormatter.probability_color(-0.1)

def test_probability_color_rejects_greater_than_one():
    """Probability color rejects values > 1"""
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        OutputFormatter.probability_color(1.5)
