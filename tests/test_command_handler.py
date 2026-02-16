import pytest
import torch
from unittest.mock import Mock, patch
from gpt_token_explorer import ModelManager, CommandHandler

def test_command_handler_can_be_created():
    """CommandHandler can be instantiated with ModelManager"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    assert handler is not None
    assert handler.model_manager is manager

def test_validate_input_rejects_empty_string():
    """validate_input returns False for empty string"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    is_valid, error = handler.validate_input("")
    assert is_valid is False
    assert error == "Empty input. Please provide text."

def test_validate_input_rejects_whitespace():
    """validate_input returns False for whitespace-only input"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    is_valid, error = handler.validate_input("   \t\n  ")
    assert is_valid is False
    assert error == "Empty input. Please provide text."

def test_validate_input_rejects_none():
    """validate_input returns False for None input"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    is_valid, error = handler.validate_input(None)
    assert is_valid is False
    assert error == "Empty input. Please provide text."

def test_validate_input_rejects_when_model_not_loaded():
    """validate_input returns False when model not loaded"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    is_valid, error = handler.validate_input("Hello world")
    assert is_valid is False
    assert error == "Model not loaded."

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_validate_input_accepts_valid_input_with_loaded_model(mock_model_class, mock_tokenizer_class):
    """validate_input returns True for valid input with loaded model"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=49152)
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    # Create handler with loaded model
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    is_valid, error = handler.validate_input("Hello world")
    assert is_valid is True
    assert error is None

def test_complete_returns_probabilities():
    """Complete command returns top-k token probabilities"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.complete("The capital of France is", top_k=5)

    assert result["success"] is True
    assert "results" in result
    assert len(result["results"]) == 5
    assert all("token" in r and "probability" in r for r in result["results"])
    # Probabilities should be in descending order
    probs = [r["probability"] for r in result["results"]]
    assert probs == sorted(probs, reverse=True)

def test_complete_rejects_empty_input():
    """Complete command rejects empty input"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.complete("")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Empty input. Please provide text."

def test_complete_rejects_unloaded_model():
    """Complete command rejects requests when model not loaded"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.complete("Hello world")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Model not loaded."

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_complete_handles_exceptions(mock_model_class, mock_tokenizer_class):
    """Complete command handles exceptions gracefully"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=49152)
    mock_tokenizer.encode.side_effect = RuntimeError("Mock error")
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    # Create handler
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.complete("test")
    assert result["success"] is False
    assert "error" in result
    assert "Mock error" in result["error"]

def test_tokenize_breaks_text_into_tokens():
    """Tokenize command breaks text into tokens with IDs"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.tokenize("Hello world!")

    assert result["success"] is True
    assert "tokens" in result
    assert len(result["tokens"]) > 0
    assert all("text" in t and "token_id" in t for t in result["tokens"])
    # NEW: Validate count and input fields
    assert result["count"] == len(result["tokens"])
    assert result["input"] == "Hello world!"

def test_tokenize_rejects_empty_input():
    """Tokenize command rejects empty input"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.tokenize("")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Empty input. Please provide text."

def test_tokenize_rejects_unloaded_model():
    """Tokenize command rejects requests when model not loaded"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.tokenize("Hello world")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Model not loaded."

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_tokenize_handles_exceptions(mock_model_class, mock_tokenizer_class):
    """Tokenize command handles exceptions gracefully"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=49152)
    mock_tokenizer.encode.side_effect = RuntimeError("Tokenization error")
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    # Create handler
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.tokenize("test")
    assert result["success"] is False
    assert "error" in result
    assert "Tokenization error" in result["error"]

def test_generate_produces_tokens():
    """Generate command produces n tokens step-by-step"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.generate("Hello", n_tokens=3)

    assert result["success"] is True
    assert "steps" in result
    assert len(result["steps"]) == 3
    assert "final_text" in result

def test_generate_rejects_empty_input():
    """Generate command rejects empty input"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.generate("")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Empty input. Please provide text."

def test_generate_rejects_unloaded_model():
    """Generate command rejects requests when model not loaded"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    result = handler.generate("Hello world")
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Model not loaded."

def test_generate_rejects_invalid_n_tokens():
    """Generate command rejects negative or zero n_tokens"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)

    result = handler.generate("Hello", n_tokens=0)
    assert result["success"] is False
    assert "n_tokens must be positive" in result["error"]

    result = handler.generate("Hello", n_tokens=-5)
    assert result["success"] is False
    assert "n_tokens must be positive" in result["error"]

def test_generate_rejects_invalid_show_alternatives():
    """Generate command rejects negative or zero show_alternatives"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)

    result = handler.generate("Hello", show_alternatives=0)
    assert result["success"] is False
    assert "show_alternatives must be positive" in result["error"]

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_generate_handles_exceptions(mock_model_class, mock_tokenizer_class):
    """Generate command handles exceptions gracefully"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=49152)
    mock_tokenizer.encode.side_effect = RuntimeError("Generation error")
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    # Create handler
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.generate("test")
    assert result["success"] is False
    assert "error" in result
    assert "Generation error" in result["error"]

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_generate_clamps_show_alternatives_to_vocab_size(mock_model_class, mock_tokenizer_class):
    """Generate command clamps show_alternatives to vocabulary size"""
    # Setup mocks with small vocab for testing
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=100)  # Small vocab for testing

    # Mock encode to return token IDs
    mock_input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.encode.return_value = mock_input_ids

    # Mock decode
    mock_tokenizer.decode.return_value = "test"

    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # Setup model mock
    mock_model = Mock()
    mock_model.device = torch.device('cpu')

    # Mock model output with logits for vocab size 100
    mock_logits = torch.randn(1, 1, 100)  # batch=1, seq_len=1, vocab=100
    mock_outputs = Mock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs

    mock_model_class.from_pretrained.return_value = mock_model

    # Create handler and test with show_alternatives > vocab_size
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    # Request 1000000 alternatives but vocab is only 100
    result = handler.generate("test", n_tokens=2, show_alternatives=1000000)

    # Should succeed (not crash)
    assert result["success"] is True
    assert len(result["steps"]) == 2

    # Each step should have alternatives clamped to vocab size (100)
    for step in result["steps"]:
        assert len(step["alternatives"]) <= 100
