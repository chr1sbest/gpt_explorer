#!/usr/bin/env python3
"""
GPT Token Explorer - Interactive REPL for learning token generation

Educational tool demonstrating next-token prediction, probability distributions,
and autoregressive generation in GPT models.
"""

import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', message='.*torch_dtype.*is deprecated.*')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import shlex
import threading
import time
from tqdm import tqdm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
import numpy as np

class ModelManager:
    """Manages HuggingFace transformer model loading and operations"""

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M"

    def __init__(self, auto_load=True, model_name=None):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name or self.DEFAULT_MODEL
        if auto_load:
            self.load_model()

    def load_model(self):
        """Load model and tokenizer from HuggingFace"""
        print(f"\n📥 Loading {self.model_name}...")
        print(f"   First run downloads tokenizer + model files (~1-2GB)")
        print(f"   Files cached in ~/.cache/huggingface/ for future use")
        print()

        try:
            # Load tokenizer (fast, ~5-10MB)
            print("   [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )

            # Load model (large, ~1-2GB depending on model)
            print("   [2/2] Loading model weights (this may take a minute)...")
            import transformers
            transformers.logging.set_verbosity_error()  # Suppress progress bars

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            self.model.eval()  # Set to evaluation mode

            vocab_size = len(self.tokenizer)
            print(f"\n✓ Ready! Vocabulary: {vocab_size:,} tokens\n")

        except Exception as e:
            stop_spinner.set()
            spinner_thread.join(timeout=0.5)
            print(f"\n✗ Error loading model: {e}")
            raise

    def _show_spinner(self, stop_event, message):
        """Show spinner animation during model loading"""
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        while not stop_event.is_set():
            print(f'\r⚙️  {message} {spinner_chars[idx % len(spinner_chars)]}',
                  end='', flush=True)
            idx += 1
            time.sleep(0.1)
        print('\r' + ' ' * 50 + '\r', end='', flush=True)

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None

class CommandHandler:
    """Handles command execution and inference operations"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def validate_input(self, text):
        """Validate user input"""
        if not text or not text.strip():
            return False, "Empty input. Please provide text."
        if not self.model_manager.is_loaded():
            return False, "Model not loaded."
        return True, None

    def complete(self, text, top_k=10):
        """Get next token probabilities

        Args:
            text: Input text prompt
            top_k: Number of top predictions to return

        Returns:
            dict with success, prompt, and results list
        """
        valid, error = self.validate_input(text)
        if not valid:
            return {"success": False, "error": error}

        try:
            # Tokenize input
            input_ids = self.model_manager.tokenizer.encode(
                text,
                return_tensors='pt'
            ).to(self.model_manager.model.device)

            # Get model output
            with torch.no_grad():
                outputs = self.model_manager.model(input_ids)

            # Extract logits for last position
            logits = outputs.logits[:, -1, :]

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Get top-k
            top_k_probs, top_k_ids = torch.topk(probs[0], top_k)

            # Decode tokens
            results = []
            for prob, token_id in zip(top_k_probs, top_k_ids):
                token = self.model_manager.tokenizer.decode([token_id])
                results.append({
                    "token": token,
                    "token_id": int(token_id),
                    "probability": float(prob)
                })

            return {
                "success": True,
                "prompt": text,
                "results": results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def tokenize(self, text):
        """Break text into tokens with IDs

        Args:
            text: Input text to tokenize

        Returns:
            dict with success, text, and tokens list
        """
        valid, error = self.validate_input(text)
        if not valid:
            return {"success": False, "error": error}

        try:
            # Encode to get token IDs
            token_ids = self.model_manager.tokenizer.encode(text)

            # Decode each token individually
            tokens = []
            for token_id in token_ids:
                token_text = self.model_manager.tokenizer.decode([token_id])
                tokens.append({
                    "text": token_text,
                    "token_id": int(token_id)
                })

            return {
                "success": True,
                "input": text,
                "tokens": tokens,
                "count": len(tokens)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate(self, text, n_tokens=10, show_alternatives=3):
        """Generate n tokens autoregressively with probabilities

        Args:
            text: Input prompt
            n_tokens: Number of tokens to generate
            show_alternatives: Number of alternative tokens to show at each step

        Returns:
            dict with success, steps, and final_text
        """
        # Validate parameters
        if n_tokens <= 0:
            return {"success": False, "error": "n_tokens must be positive"}
        if show_alternatives <= 0:
            return {"success": False, "error": "show_alternatives must be positive"}

        valid, error = self.validate_input(text)
        if not valid:
            return {"success": False, "error": error}

        try:
            input_ids = self.model_manager.tokenizer.encode(
                text,
                return_tensors='pt'
            ).to(self.model_manager.model.device)

            steps = []
            current_text = text

            for step in range(n_tokens):
                # Get next token probabilities
                with torch.no_grad():
                    outputs = self.model_manager.model(input_ids)

                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

                # Clamp show_alternatives to vocab size
                show_alternatives = min(show_alternatives, probs.shape[-1])

                # Get top alternatives
                top_probs, top_ids = torch.topk(probs[0], show_alternatives)

                # Sample the most likely token
                next_token_id = top_ids[0]
                next_token = self.model_manager.tokenizer.decode([next_token_id])

                # Record alternatives
                alternatives = []
                for prob, token_id in zip(top_probs, top_ids):
                    token = self.model_manager.tokenizer.decode([token_id])
                    alternatives.append({
                        "token": token,
                        "probability": float(prob),
                        "selected": (token_id == next_token_id)
                    })

                # Append to input for next iteration
                current_text += next_token
                input_ids = torch.cat([
                    input_ids,
                    next_token_id.unsqueeze(0).unsqueeze(0)
                ], dim=1)

                # Record step with accumulated text
                steps.append({
                    "step": step + 1,
                    "current_text": current_text,
                    "alternatives": alternatives,
                    "selected": next_token
                })

            return {
                "success": True,
                "prompt": text,
                "steps": steps,
                "final_text": current_text
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

class OutputFormatter:
    """Formats command results for terminal display"""

    # ANSI color codes
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    RESET = '\033[0m'

    @staticmethod
    def probability_bar(prob, max_width=20):
        """Create probability bar visualization

        Args:
            prob: Probability value (0.0 to 1.0)
            max_width: Maximum width of bar in characters

        Returns:
            String representation of probability bar

        Raises:
            ValueError: If prob is not between 0 and 1
        """
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be between 0 and 1, got {prob}")
        filled = int(prob * max_width)
        return '█' * filled + '░' * (max_width - filled)

    @staticmethod
    def probability_color(prob):
        """Get color based on probability threshold

        Args:
            prob: Probability value (0.0 to 1.0)

        Returns:
            ANSI color code string

        Raises:
            ValueError: If prob is not between 0 and 1
        """
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be between 0 and 1, got {prob}")
        if prob > 0.5:
            return OutputFormatter.GREEN
        elif prob > 0.1:
            return OutputFormatter.YELLOW
        else:
            return OutputFormatter.RED

    @staticmethod
    def format_complete(result):
        """Format complete command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        lines = [f"\n{OutputFormatter.BOLD}Next Token Predictions:{OutputFormatter.RESET}\n"]

        for i, item in enumerate(result["results"], 1):
            prob = item["probability"]
            token = item["token"]
            token_id = item["token_id"]

            color = OutputFormatter.probability_color(prob)
            bar = OutputFormatter.probability_bar(prob)

            lines.append(
                f"   {i:2d}. {color}{token:20s}{OutputFormatter.RESET} "
                f"{prob:.4f} {bar} {OutputFormatter.GRAY}[{token_id}]{OutputFormatter.RESET}"
            )

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_tokenize(result):
        """Format tokenize command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        lines = [
            f"\n{OutputFormatter.BOLD}Tokenization:{OutputFormatter.RESET}",
            f"   Input: \"{result['input']}\"",
            f"   Tokens: {result['count']}\n"
        ]

        for i, token in enumerate(result["tokens"], 1):
            token_text = token["text"]
            token_id = token["token_id"]
            lines.append(
                f"   {i:2d}. {OutputFormatter.BLUE}\"{token_text}\"{OutputFormatter.RESET} "
                f"{OutputFormatter.GRAY}[ID: {token_id}]{OutputFormatter.RESET}"
            )

        lines.append(f"\n   {OutputFormatter.YELLOW}Note: Tokens split by Byte-Pair Encoding (BPE){OutputFormatter.RESET}\n")
        return "\n".join(lines)

    @staticmethod
    def format_generate(result):
        """Format generate command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        lines = [f"\n{OutputFormatter.BOLD}Step-by-Step Generation:{OutputFormatter.RESET}\n"]

        for step_data in result["steps"]:
            step = step_data["step"]
            current = step_data["current_text"]
            selected = step_data["selected"]
            alternatives = step_data["alternatives"]

            lines.append(f"   {OutputFormatter.BOLD}Step {step}:{OutputFormatter.RESET} \"{current}\"")

            # Show top alternatives
            alt_strs = []
            for alt in alternatives:
                color = OutputFormatter.GREEN if alt["selected"] else OutputFormatter.GRAY
                alt_strs.append(f"{color}{alt['token']}{OutputFormatter.RESET} ({alt['probability']:.2f})")

            lines.append(f"   → {' | '.join(alt_strs)}\n")

        lines.append(f"   {OutputFormatter.BOLD}Final:{OutputFormatter.RESET} \"{result['final_text']}\"\n")
        return "\n".join(lines)

class TokenREPL:
    """Interactive REPL for GPT token exploration"""

    COMMANDS = ['complete', 'generate', 'tokenize', 'help', 'quit', 'exit']

    def __init__(self, model_name=None):
        self.model_manager = None
        self.command_handler = None
        self.formatter = OutputFormatter()
        self.model_name = model_name

        # prompt_toolkit session
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.COMMANDS, ignore_case=True)
        )

    def start(self):
        """Start the REPL"""
        self.print_welcome()

        # Load model
        self.model_manager = ModelManager(model_name=self.model_name)
        self.command_handler = CommandHandler(self.model_manager)

        # Show first-time tips
        self.print_tips()

        # Main loop
        while True:
            try:
                user_input = self.session.prompt('\ngpt> ').strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'help':
                    self.print_help()
                    continue

                self.execute_command(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                print("\nGoodbye!")
                break

    def print_welcome(self):
        """Print welcome banner"""
        print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}  GPT Token Explorer - Learn How LLMs Generate Text{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")

    def print_tips(self):
        """Print usage tips"""
        print(f"Try these commands:")
        print(f'  {OutputFormatter.BLUE}complete{OutputFormatter.RESET} "Hello, my name is"')
        print(f'  {OutputFormatter.BLUE}generate{OutputFormatter.RESET} "Once upon a time" 10')
        print(f'  {OutputFormatter.BLUE}tokenize{OutputFormatter.RESET} "ChatGPT is cool!"')
        print(f'  {OutputFormatter.BLUE}help{OutputFormatter.RESET}')

    def print_help(self):
        """Print help message"""
        help_text = f"""
{OutputFormatter.BOLD}Available Commands:{OutputFormatter.RESET}

  {OutputFormatter.BLUE}complete{OutputFormatter.RESET} <text>
      Show next token probabilities
      Example: complete "The capital of France is"

  {OutputFormatter.BLUE}generate{OutputFormatter.RESET} <text> [n]
      Generate n tokens step-by-step (default n=10)
      Example: generate "Once upon a time" 5

  {OutputFormatter.BLUE}tokenize{OutputFormatter.RESET} <text>
      Break text into tokens with IDs
      Example: tokenize "ChatGPT is amazing!"

  {OutputFormatter.BLUE}help{OutputFormatter.RESET}
      Show this help message

  {OutputFormatter.BLUE}quit{OutputFormatter.RESET} / {OutputFormatter.BLUE}exit{OutputFormatter.RESET}
      Exit the explorer

{OutputFormatter.BOLD}Tips:{OutputFormatter.RESET}
  • Trailing spaces are auto-stripped (they change tokenization)
  • Try factual prompts: "The capital of France is"
  • Avoid prompts common in lists: "My name is"
"""
        print(help_text)

    def execute_command(self, user_input):
        """Parse and execute user command"""
        try:
            # Use shlex to properly handle quoted strings
            parts = shlex.split(user_input)
        except ValueError:
            # Fallback to simple split if shlex fails
            parts = user_input.split()

        if not parts:
            return

        cmd = parts[0].lower()

        if cmd == 'complete':
            if len(parts) < 2:
                print(f"{OutputFormatter.RED}Usage: complete <text>{OutputFormatter.RESET}")
                return
            # Join remaining parts as text (handles multi-word input)
            text = ' '.join(parts[1:])

            # Strip trailing spaces (BPE tokenization quirk)
            if text.endswith(' '):
                print(f"{OutputFormatter.YELLOW}Note: Stripped trailing space (affects tokenization){OutputFormatter.RESET}")
                text = text.rstrip()

            result = self.command_handler.complete(text)
            print(self.formatter.format_complete(result))

        elif cmd == 'tokenize':
            if len(parts) < 2:
                print(f"{OutputFormatter.RED}Usage: tokenize <text>{OutputFormatter.RESET}")
                return
            text = ' '.join(parts[1:])
            result = self.command_handler.tokenize(text)
            print(self.formatter.format_tokenize(result))

        elif cmd == 'generate':
            if len(parts) < 2:
                print(f"{OutputFormatter.RED}Usage: generate <text> [n]{OutputFormatter.RESET}")
                return

            # Check if last part is a number (n_tokens)
            if len(parts) >= 3 and parts[-1].isdigit():
                text = ' '.join(parts[1:-1])
                n = int(parts[-1])
            else:
                text = ' '.join(parts[1:])
                n = 10

            # Strip trailing spaces (BPE tokenization quirk)
            if text.endswith(' '):
                print(f"{OutputFormatter.YELLOW}Note: Stripped trailing space (affects tokenization){OutputFormatter.RESET}")
                text = text.rstrip()

            result = self.command_handler.generate(text, n_tokens=n)
            print(self.formatter.format_generate(result))

        else:
            print(f"{OutputFormatter.RED}Unknown command: {cmd}{OutputFormatter.RESET}")
            print("Type 'help' for available commands.")

def select_model_interactive():
    """Interactive model selector"""
    print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}  Select Language Model{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")

    models = [
        {
            'name': 'HuggingFaceTB/SmolLM-135M',
            'display': 'SmolLM-135M (Recommended)',
            'size': '135MB',
            'load_time': '~3s',
            'description': '✓ Fastest'
        },
        {
            'name': 'Qwen/Qwen2.5-1.5B',
            'display': 'Qwen 2.5 1.5B',
            'size': '1.5GB',
            'load_time': '~15s',
            'description': '✓ Higher quality'
        }
    ]

    for i, model in enumerate(models, 1):
        print(f"{OutputFormatter.BOLD}{i}. {model['display']}{OutputFormatter.RESET}")
        print(f"   Size: {model['size']} | Load time: {model['load_time']}")
        print(f"   {OutputFormatter.GREEN}{model['description']}{OutputFormatter.RESET}")
        print()

    while True:
        try:
            choice = input("Choose model (1-2) [default: 1]: ").strip()
            if not choice:
                choice = '1'

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected = models[choice_idx]
                print(f"\n✓ Selected: {selected['display']}")
                return selected['name']
            else:
                print(f"{OutputFormatter.RED}Please enter 1 or 2{OutputFormatter.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{OutputFormatter.YELLOW}Using default (SmolLM-135M){OutputFormatter.RESET}")
            return models[0]['name']
        except EOFError:
            return models[0]['name']

def main():
    """Entry point for GPT token explorer"""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPT Token Explorer - Learn how LLMs generate text',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use (skips interactive selection)'
    )

    args = parser.parse_args()

    # If no model specified, show interactive selector
    model_name = args.model
    if model_name is None:
        model_name = select_model_interactive()

    repl = TokenREPL(model_name=model_name)
    repl.start()

if __name__ == '__main__':
    main()
