"""Debug test to see actual API responses."""

import asyncio
import os
import httpx


async def test_raw_api():
    """Test raw API call to see response structure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Test GPT-5
    payload = {
        "model": "gpt-5",
        "input": "What are you? Answer in one sentence.",
        "store": False
    }

    print("Sending request to Responses API...")
    print(f"URL: {url}")
    print(f"Payload: {payload}\n")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}\n")

            if response.status_code == 200:
                data = response.json()
                print("Response JSON:")
                import json
                print(json.dumps(data, indent=2))
            else:
                print(f"Error response: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_raw_api())
