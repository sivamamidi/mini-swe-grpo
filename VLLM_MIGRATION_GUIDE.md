# Mini-SWE-RL → vLLM Migration Guide

This document explains how to replace Ollama with vLLM inside Mini-SWE-RL.

Goal:

- Replace Ollama inference with vLLM
- Use OpenAI-compatible API
- Serve model from RunPod GPU
- Keep RL/GRPO training unchanged
- Improve rollout generation speed

---

# Architecture

Old pipeline:

Ollama
→ local inference
→ agent.py
→ environment
→ reward

New pipeline:

vLLM
→ OpenAI API
→ agent.py
→ environment
→ reward

---

# Why Replace Ollama?

vLLM is much better for RL systems because:

- Faster inference
- Better batching
- Better GPU utilization
- Lower latency
- Higher rollout throughput
- Industry-standard RL serving stack

Modern RL systems like:
- DeepSeek
- OpenRLHF
- verl
- DeepSWE

all use vLLM-style serving.

---

# RunPod Setup

Recommended GPU:

- RTX 5090
- 32GB VRAM

Recommended model:

Qwen2.5-Coder-1.5B-Instruct

---

# Install Dependencies

```bash
pip install vllm
pip install openai
pip install transformers accelerate peft trl bitsandbytes
```

---

# Start vLLM Server

Run this on RunPod:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-Coder-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 1024
```

Important:

DO NOT use:

```bash
--chat-template-content-format openai
```

It breaks message formatting for this setup.

---

# Verify vLLM Server

Check models:

```bash
curl http://localhost:8000/v1/models
```

Expected:

```json
{
  "data": [
    {
      "id": "./models/Qwen2.5-Coder-1.5B-Instruct"
    }
  ]
}
```

The model id must match EXACTLY in Python requests.

---

# Test vLLM

Create:

test_vllm.py

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="./models/Qwen2.5-Coder-1.5B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a coding assistant."
        },
        {
            "role": "user",
            "content": (
                "Fix this Python bug:\n\n"
                "def add(a,b):\n"
                "    return a-b\n\n"
                "Return ONLY the corrected function."
            )
        }
    ],
    temperature=0.8,
    max_tokens=128
)

print(response.choices[0].message.content)
```

Run:

```bash
python test_vllm.py
```

Expected output:

```python
def add(a,b):
    return a+b
```

---

# agent.py Migration

---

# OLD OLLAMA CODE

```python
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
```

And:

```python
query_ollama(...)
```

must be removed.

---

# NEW VLLM CODE

Replace with:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "./models/Qwen2.5-Coder-1.5B-Instruct"
```

---

# New query_vllm Function

```python
def query_vllm(prompt: str,
               model: str = MODEL_NAME,
               temperature: float = 0.0) -> str:

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding assistant. "
                        "Return ONLY valid Python code. "
                        "No markdown. No explanations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=512,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"ERROR: {e}"
```

---

# Output Cleaning

LLMs often produce:

- markdown
- explanations
- code fences

These break environment execution.

Use:

```python
def extract_code(response: str) -> str:

    response = response.replace("```python", "")
    response = response.replace("```", "")
    response = response.strip()

    return response
```

---

# Important RL Concept

GRPO requires rollout diversity.

Use:

```python
temperature=0.7
```

or:

```python
temperature=0.8
```

during RL rollout generation.

Do NOT use greedy decoding for RL training.

---

# Reward Pipeline

The environment computes:

```python
reward, info = env.step(fixed_code)
```

Typical reward:

```python
reward = 1 if all tests pass else 0
```

This creates binary reinforcement learning.

---

# RL Training Flow

agent.py:
- generates rollouts

env.py:
- computes rewards

grpo_trainer.py:
- computes advantages
- updates weights

---

# Advantage Formula

GRPO computes normalized advantages:

A_i = (r_i - mean(r)) / std(r)

This allows the model to learn from:
- better rollouts
- worse rollouts

within each rollout group.

---

# Why Mixed Rewards Matter

Good RL signal requires:

- some successful rollouts
- some failed rollouts

If:
- all pass
OR
- all fail

then advantages become near zero.

Learning stalls.

This is the "sweet spot" from the lecture.

---

# Recommended Settings

Model:
- Qwen2.5-Coder-1.5B-Instruct

Temperature:
- 0.7–0.8

Max tokens:
- 512

GPU memory utilization:
- 0.7–0.9

Training:
- LoRA recommended

---

# LoRA Recommendation

Install:

```bash
pip install peft bitsandbytes
```

Use:

```python
from peft import LoraConfig, get_peft_model
```

Example:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Wrap model:

```python
model = get_peft_model(model, lora_config)
```

---

# Training Command

Run:

```bash
python grpo_trainer_v2.py
```

Expected logs:

- Generating rollouts
- Computing rewards
- Computing advantages
- Optimizer step

---

# GPU Monitoring

Use:

```bash
watch -n 1 nvidia-smi
```

Monitor:
- VRAM
- utilization
- crashes

---

# Common Problems

## 404 model not found

Fix:
- use exact model id from:

```bash
curl http://localhost:8000/v1/models
```

---

## can only concatenate str (not "list") to str

Cause:
- bad chat-template configuration

Fix:
- remove:

```bash
--chat-template-content-format openai
```

---

## CUDA OOM

Reduce:
- batch size
- max tokens
- model size

---

# Recommended Workflow

Best workflow:

Local machine:
- VS Code
- Cursor
- GitHub

RunPod:
- vLLM serving
- RL training
- checkpoints

Workflow:

Local edit
→ git push
→ RunPod git pull
→ train

This is much better than editing files directly in browser terminals.

---

# Final Goal

Pipeline:

vLLM
→ rollout generation
→ reward computation
→ GRPO updates
→ improved coding behavior

This reproduces a miniature version of:
- DeepSeek RL
- OpenRLHF
- DeepSWE
- SWE-RL
