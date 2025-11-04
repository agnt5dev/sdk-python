"""Debug GPT-5 responses to see full output."""

import asyncio
import os
import httpx
import json


async def test_model(model_name: str):
    """Test a specific model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "input": "What are you? Answer in one sentence.",
        "store": False
    }

    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    print(f"Payload: {payload}\n")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)

            if response.status_code == 200:
                data = response.json()
                print("Full Response:")
                print(json.dumps(data, indent=2))
                print(f"\nOutput items count: {len(data.get('output', []))}")

                for i, item in enumerate(data.get('output', [])):
                    print(f"\nOutput Item {i+1}:")
                    print(f"  Type: {item.get('type')}")
                    if item.get('type') == 'message':
                        print(f"  Content items: {len(item.get('content', []))}")
                        for j, content in enumerate(item.get('content', [])):
                            print(f"    Content {j+1}:")
                            print(f"      Type: {content.get('type')}")
                            print(f"      Text: {content.get('text', 'N/A')[:100]}")
                    elif item.get('type') == 'reasoning':
                        print(f"  Summary: {item.get('summary')}")

                print(f"\nUsage:")
                usage = data.get('usage', {})
                print(f"  Input: {usage.get('input_tokens')}")
                print(f"  Output: {usage.get('output_tokens')}")
                print(f"  Output details: {usage.get('output_tokens_details')}")

            else:
                print(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    await test_model("gpt-5")
    await test_model("gpt-5-mini")


if __name__ == "__main__":
    asyncio.run(main())
