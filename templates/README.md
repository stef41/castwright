# Castwright Seed Templates

This directory contains ready-to-use seed example templates for common instruction-tuning tasks.

## Template Format

Each JSON file is a list of objects matching the `Seed` format:

```json
[
  {
    "instruction": "The task instruction or prompt",
    "output": "The expected response",
    "input": "(optional) Additional input context",
    "system": "(optional) System prompt"
  }
]
```

Only `instruction` and `output` are required. The `input` and `system` fields are optional.

## Available Templates

| File | Task | Description |
|------|------|-------------|
| `code_generation.json` | Instruction → Code | Python function generation from natural language specs |
| `qa_pairs.json` | Question → Answer | Technical knowledge Q&A pairs |
| `summarization.json` | Document → Summary | Long-text summarization examples |
| `classification.json` | Text → Label | Sentiment classification examples |

## Usage

Load a template as seeds and generate more examples:

```python
from castwright import generate, load_seeds, GenerationConfig
from castwright.providers import OpenAIProvider

seeds = load_seeds("castwright/templates/qa_pairs.json")

result = generate(
    seeds=seeds,
    provider=OpenAIProvider(api_key="..."),
    config=GenerationConfig(n=50, temperature=0.9),
)

print(f"Generated {result.n_generated} examples")
```

## Creating Custom Templates

Create a JSON file with at least 3 diverse seed examples following the format above. More seeds with greater diversity lead to higher-quality generated outputs.
