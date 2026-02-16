import pytest
from unittest.mock import Mock, patch
from gpt_token_explorer import ModelManager

def test_model_manager_can_be_created_without_loading():
    """ModelManager can be instantiated without auto-loading"""
    manager = ModelManager(auto_load=False)
    assert manager is not None
    assert manager.model is None
    assert manager.tokenizer is None

@patch('gpt_token_explorer.AutoTokenizer')
@patch('gpt_token_explorer.AutoModelForCausalLM')
def test_model_manager_loads_with_mocked_model(mock_model_class, mock_tokenizer_class):
    """ModelManager loads model and tokenizer (mocked for fast testing)"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.__len__ = Mock(return_value=49152)
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    # Test loading
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")

    # Verify calls
    mock_tokenizer_class.from_pretrained.assert_called_once_with(
        "HuggingFaceTB/SmolLM-135M",
        use_fast=True
    )
    mock_model_class.from_pretrained.assert_called_once()
    mock_model.eval.assert_called_once()

    # Verify state
    assert manager.is_loaded()
    assert manager.model is mock_model
    assert manager.tokenizer is mock_tokenizer

def test_model_manager_loads_real_model_integration():
    """ModelManager loads real model from HuggingFace (integration test)"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    assert manager.is_loaded()
    assert manager.model is not None
    assert manager.tokenizer is not None
