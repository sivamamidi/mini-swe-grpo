"""
Quick smoke test for the vLLM server.

Run this BEFORE using agent.py or training scripts to verify
the vLLM OpenAI-compatible API is reachable and responding.

Usage:
    python test_vllm.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="./Mini-SWE-RL/models/Qwen2.5-Coder-1.5B-Instruct",
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
