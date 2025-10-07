"""
Asynchronous LLM Example

This example demonstrates concurrent/parallel LLM requests using asyncio.
Shows the new simplified API with multiple providers running in parallel.

Prerequisites:
    Set environment variables for the providers you want to use:
    - OPENAI_API_KEY=your-key
    - ANTHROPIC_API_KEY=your-key
    - GROQ_API_KEY=your-key

Usage:
    python 13_llm_async.py
"""

import asyncio
import os
import time
from typing import List, Tuple

from agnt5 import lm


async def example_parallel_requests():
    """Example: Make multiple requests in parallel to the same provider."""
    print("=== Example 1: Parallel Requests (Same Provider) ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # List of questions to ask
    questions = [
        "What is Python?",
        "What is Rust?",
        "What is JavaScript?",
        "What is Go?",
    ]

    # Create tasks for parallel execution
    async def ask_question(question: str) -> Tuple[str, str]:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            prompt=f"{question} Answer in one sentence.",
            temperature=0.7,
            max_tokens=50
        )
        return question, response.text

    # Measure time for parallel execution
    start_time = time.time()
    tasks = [ask_question(q) for q in questions]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Print results
    for question, answer in results:
        print(f"Q: {question}")
        print(f"A: {answer}\n")

    print(f"Total time (parallel): {elapsed:.2f}s")
    print(f"Average per request: {elapsed/len(questions):.2f}s\n")


async def example_multi_provider_parallel():
    """Example: Query multiple providers in parallel."""
    print("=== Example 2: Multi-Provider Parallel Requests ===\n")

    providers = []

    # Add available providers
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI GPT-4o-mini", "openai/gpt-4o-mini"))

    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic Claude 3.5 Haiku", "anthropic/claude-3-5-haiku-20241022"))

    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq Llama 3.3 70B", "groq/llama-3.3-70b-versatile"))

    if not providers:
        print("⚠️  Skipped: No API keys set\n")
        return

    # Same question to all providers
    question = "What is the meaning of life? Answer in one sentence."

    # Create tasks for each provider
    async def query_provider(provider_name: str, model: str) -> Tuple[str, str, float]:
        start = time.time()
        response = await lm.generate(
            model=model,
            prompt=question,
            temperature=0.7,
            max_tokens=100
        )
        elapsed = time.time() - start
        return provider_name, response.text, elapsed

    # Execute in parallel
    start_time = time.time()
    tasks = [query_provider(name, model) for name, model in providers]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - start_time

    # Print results
    print(f"Question: {question}\n")
    for provider_name, answer, elapsed in results:
        print(f"[{provider_name}] ({elapsed:.2f}s)")
        print(f"  {answer}\n")

    print(f"Total time (parallel): {total_elapsed:.2f}s")
    print(f"Fastest response: {min(r[2] for r in results):.2f}s\n")


async def example_batch_processing():
    """Example: Process a batch of items with rate limiting."""
    print("=== Example 3: Batch Processing with Concurrency Control ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Batch of items to process
    items = [
        "apple",
        "banana",
        "orange",
        "grape",
        "strawberry",
        "watermelon",
        "pineapple",
        "mango",
    ]

    # Process with concurrency limit
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

    async def process_item(item: str) -> Tuple[str, str]:
        async with semaphore:
            response = await lm.generate(
                model="openai/gpt-4o-mini",
                prompt=f"Describe the taste of {item} in 5 words or less.",
                temperature=0.7,
                max_tokens=20
            )
            return item, response.text

    # Process all items
    start_time = time.time()
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Print results
    print("Taste descriptions:")
    for item, description in results:
        print(f"  {item.capitalize()}: {description}")

    print(f"\nProcessed {len(items)} items in {elapsed:.2f}s")
    print(f"Average per item: {elapsed/len(items):.2f}s\n")


async def example_fallback_providers():
    """Example: Fallback to alternative providers on failure."""
    print("=== Example 4: Provider Fallback Strategy ===\n")

    # Try providers in order of preference
    providers_config = [
        ("OpenAI", "openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ("Groq", "groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ("Anthropic", "anthropic/claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
    ]

    question = "What is AGNT5? Answer in one sentence."

    for provider_name, model, env_var in providers_config:
        if not os.getenv(env_var):
            print(f"⏭️  Skipping {provider_name}: {env_var} not set")
            continue

        try:
            print(f"🔄 Trying {provider_name}...")

            response = await lm.generate(
                model=model,
                prompt=question,
                temperature=0.7,
                max_tokens=100
            )

            print(f"✅ Success with {provider_name}")
            print(f"Response: {response.text}\n")
            break

        except Exception as e:
            print(f"❌ Failed with {provider_name}: {e}")
            continue
    else:
        print("⚠️  All providers failed or unavailable\n")


async def example_concurrent_conversations():
    """Example: Handle multiple concurrent conversations."""
    print("=== Example 5: Concurrent Conversations ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Different conversation contexts
    conversations = [
        ("Math Tutor", "Explain the Pythagorean theorem in simple terms."),
        ("Cooking Assistant", "How do I make a perfect omelet?"),
        ("Travel Guide", "What are the top 3 attractions in Paris?"),
    ]

    async def handle_conversation(role: str, question: str) -> Tuple[str, str]:
        response = await lm.generate(
            model="openai/gpt-4o-mini",
            system_prompt=f"You are a helpful {role}.",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=150
        )
        return role, response.text

    # Handle all conversations in parallel
    tasks = [handle_conversation(role, q) for role, q in conversations]
    results = await asyncio.gather(*tasks)

    # Print results
    for role, response in results:
        print(f"[{role}]")
        print(f"  {response}\n")


async def main():
    print("=== AGNT5 Asynchronous LLM Examples ===")
    print("Concurrent requests with simplified API\n")

    try:
        # Run all examples
        await example_parallel_requests()
        await example_multi_provider_parallel()
        await example_batch_processing()
        await example_fallback_providers()
        await example_concurrent_conversations()

        print("=== All Examples Complete ===")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
