# 🤖 GPT Token Explorer

Interactive Python REPL for learning how GPT models generate text. Demonstrates next-token prediction, probability distributions, and autoregressive generation step-by-step.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Complete** - See probability distributions for the next token
- **Generate** - Watch autoregressive generation unfold token-by-token
- **Tokenize** - Understand how text splits into BPE subword tokens
- **Attention** - Interactive HTML heatmaps showing transformer attention weights
- **Model Comparison** - Compare SmolLM (fast) vs Qwen (quality)

Rich REPL with command history, autocomplete, and colored output.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/chr1sbest/gpt_explorer.git
cd gpt_explorer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run - you'll choose a model interactively
./explore.sh

# Or specify model directly
./explore.sh --model HuggingFaceTB/SmolLM-135M  # Fast, 135MB
./explore.sh --model Qwen/Qwen2.5-1.5B         # Better quality, 1.5GB
```

**First run:** Choose your model:
1. **SmolLM-135M** (recommended) - Fastest, 135MB, ~3s load
2. **Qwen 2.5 1.5B** - Higher quality, 1.5GB, ~15s load

## 💡 Usage

**See next-token probabilities:**
```
gpt> complete "The capital of France is"

   Next Token Predictions:
   1.  Paris               0.2734 █████░░░░░░░░░░░░░░░
   2.  located             0.0635 █░░░░░░░░░░░░░░░░░░░
   3.  a                   0.0453 ░░░░░░░░░░░░░░░░░░░░
```

**Generate text step-by-step:**
```
gpt> generate "The quick brown fox" 3

   Step 1: "The quick brown fox jumps"
   → jumps (0.81) | jumped (0.09) | is (0.02)

   Step 2: "The quick brown fox jumps over"
   → over (0.45) | on (0.12) | into (0.08)

   Step 3: "The quick brown fox jumps over the"
   → the (0.52) | a (0.18) | an (0.06)

   Final: "The quick brown fox jumps over the"
```

**Understand tokenization:**
```
gpt> tokenize "ChatGPT is amazing!"

   Tokenization:
   Input: "ChatGPT is amazing!"
   Tokens: 5

   1. "Chat"      [ID: 13667]
   2. "GPT"       [ID: 38]
   3. " is"       [ID: 318]
   4. " amazing"  [ID: 4998]
   5. "!"         [ID: 0]

   Note: Spaces become part of tokens with BPE
```

**Visualize attention weights:**
```
gpt> attention "The capital of France is"

   ✓ Attention visualization exported!
   File: attention.html
   Tokens: 5
   Layer: 30, Head: 1

   Open attention.html in your browser to view!
```

Opens an interactive HTML heatmap showing how each token attends to previous tokens. Hover over cells to see exact weights!

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `complete <text>` | Show next token probabilities | `complete "Hello world"` |
| `generate <text> [n]` | Generate n tokens step-by-step | `generate "Once upon a time" 10` |
| `tokenize <text>` | Break text into BPE tokens | `tokenize "Hello!"` |
| `attention <text> [file]` | Export attention heatmap to HTML | `attention "Hello world"` |
| `help` | Show all commands | |
| `quit` | Exit | |

## 🧠 How It Works

GPT models predict **one token at a time** through autoregressive generation:

1. **Tokenization** - Text split into subword tokens using BPE (Byte-Pair Encoding)
2. **Forward Pass** - Transformer model produces scores (logits) for each vocabulary token
3. **Softmax** - Logits converted to probabilities that sum to 1.0
4. **Selection** - Highest probability token selected (greedy decoding)
5. **Repeat** - Selected token appended to input, process repeats

**Attention Mechanism:**

Both models use **Grouped Query Attention (GQA)**, a modern variant of the transformer attention mechanism:

```
Input: "The capital of France is"

Layer 1-30 (SmolLM) or 1-28 (Qwen):
  Each token attends to ALL previous tokens:

  "is" looks back at:
    "The"     → 0.05  (low relevance)
    "capital" → 0.10
    "of"      → 0.08
    "France"  → 0.72  ← HIGHEST! The model focuses here
    "is"      → 0.05

  The attention weights determine which previous tokens
  are most important for predicting the next token.
```

**Architecture details:**
- **SmolLM:** 30 layers × 9 attention heads (GQA with 3 KV heads)
- **Qwen:** 28 layers × 12 attention heads (GQA with 2 KV heads)

GQA is faster than standard multi-head attention while maintaining quality. Each forward pass computes hundreds of attention operations across all layers!

**Why "The quick brown fox" → "jumps" (80% confidence)?**

The model learned this pattern from training data. During training on billions of tokens, it saw "The quick brown fox jumps" thousands of times, so it assigns high probability to "jumps" when it sees that context.

**BPE Tokenization Example:**

```
"ChatGPT" → ["Chat", "GPT"]  (split into common subwords)
"running" → ["run", "ning"]   (stem + suffix)
" cat"    → [" cat"]          (space + word as single token)
```

This allows models to handle rare words by breaking them into known subwords.

**Trailing Space Quirk:**

The tool auto-strips trailing spaces because they change tokenization:
- `"France is"` → predicts `" Paris"` (space+Paris as one token) ✓
- `"France is "` → predicts `"1"` (after standalone space token) ✗

See the yellow warning when this happens!

## 🎛️ Models

**SmolLM-135M** (HuggingFace)
- Parameters: 135M
- Vocabulary: 49K tokens
- Load time: ~3s
- Best for: Quick demos, learning basics

**Qwen 2.5 1.5B** (Alibaba)
- Parameters: 1.5B
- Vocabulary: 152K tokens
- Load time: ~15s
- Best for: Higher quality predictions

Models are downloaded once to `~/.cache/huggingface/`

## 📋 Requirements

- Python 3.8+
- ~2GB disk space (model cache)
- ~2GB RAM for SmolLM, ~4GB for Qwen

**Apple Silicon (M1/M2/M3):** The `explore.sh` script automatically uses ARM64 architecture.

## 🔧 Troubleshooting

**Architecture mismatch on Apple Silicon?**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
arch -arm64 pip install torch --index-url https://download.pytorch.org/whl/cpu
arch -arm64 pip install transformers prompt_toolkit numpy tqdm
```

**Models predicting numbers instead of words?**

Try factual prompts or famous phrases. Prompts like "My name is" trigger number predictions because they appear in numbered contexts (forms, lists) in training data. Use prompts like:
- "The capital of France is"
- "Once upon a time"
- "The quick brown fox"

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Ideas:
- Add sampling strategies (temperature, top-k, nucleus)
- Support more models (GPT-2, larger LLMs)
- Visualization features (attention heatmaps)
- Batch processing mode

## 📚 References

**Key Papers:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017) - Transformer architecture
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - Ainslie et al. (2023) - Grouped Query Attention
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al. (2019) - GPT-2
- [Neural Machine Translation with Subword Units](https://arxiv.org/abs/1508.07909) - Sennrich et al. (2016) - BPE tokenization

**Models:**
- [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M) - HuggingFace
- [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B) - Alibaba Cloud

**Frameworks:**
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model loading
- [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/) - REPL interface

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**Educational project** • Models from HuggingFace Hub
