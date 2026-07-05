from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

PASS_QUERY = "How often should I deworm my cat?"
FAIL_QUERY = "What's the weather in Paris today?"


async def stream_query(label: str, content: str) -> None:
    print(f"\n=== {label} ===")
    async for chunk in client.runs.stream(
        None,
        "agent_with_helpfulness",
        input={"messages": [{"role": "human", "content": content}]},
        stream_mode="updates",
    ):
        print(chunk)


async def main():
    await stream_query("Likely pass", PASS_QUERY)
    await stream_query("Likely fail/retry", FAIL_QUERY)


if __name__ == "__main__":
    asyncio.run(main())
