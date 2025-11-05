import re
import asyncio
import random
from time import time
from typing import List, Tuple

from openai import AsyncOpenAI
from src.config import OPENAI_API_KEY, MODELS_LIST

MAX_PARALLEL_REQUESTS = 20
_semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def call_chat_model(messages: list[dict], model: str, max_retries: int = 5):
    if model not in MODELS_LIST:
        raise ValueError(f"Model {model} not in allowed MODELS_LIST")
    async with _semaphore:
        for attempt in range(max_retries):
            try:
                if "o3" in model:
                    resp = await client.chat.completions.create(
                        model=model,
                        max_completion_tokens=2048,
                        temperature=1,
                        messages=messages,
                        reasoning_effort="low",
                    )
                elif model == "gpt-5-mini":
                    resp = await client.chat.completions.create(
                        model=model,
                        max_completion_tokens=2048,
                        messages=messages,
                        reasoning_effort="low",
                    )
                else:
                    resp = await client.chat.completions.create(
                        model=model,
                        max_completion_tokens=2048,
                        temperature=0,
                        messages=messages,
                    )
                return resp
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + random.random()
                print(f"{type(e).__name__}: {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

async def call_many_with_prefix(prefix_msgs: list[dict], prompts: List[str], model: str):
    tasks = []
    for p in prompts:
        msgs = prefix_msgs + [{"role": "user", "content": p}]
        tasks.append(call_chat_model(msgs, model))
    return await asyncio.gather(*tasks)

def get_few_shot_prompt(prompts_and_responses: List[Tuple[str, str]]) -> List[dict]:
    msgs = []
    for p, r in prompts_and_responses:
        msgs.append({"role": "user", "content": p})
        msgs.append({"role": "assistant", "content": r})
    return msgs

def format_answer(response: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else ""
