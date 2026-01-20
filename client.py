import asyncio
import os
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "run_research_task",
                arguments={
                    "query": "Explain Retrieval Augmented Generation and why it matters"
                },
            )

            print("\n=== AGENT OUTPUT ===")
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
