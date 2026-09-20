import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)


async def main():

    # --------------------------------------------------
    # 1. Locate the MCP server
    # --------------------------------------------------

    client_directory = Path(__file__).parent

    server_path = client_directory / "Servers" / "filesystem_server.py"

    print("MCP Server:", server_path)

    # --------------------------------------------------
    # 2. Define how the MCP server should be started
    # --------------------------------------------------

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    )

    # --------------------------------------------------
    # 3. Start the MCP server and create a connection
    # --------------------------------------------------

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # --------------------------------------------------
            # 4. Initialize MCP session
            # --------------------------------------------------

            await session.initialize()

            print("\nMCP connection established!")

            # --------------------------------------------------
            # 5. Ask MCP server for available tools
            # --------------------------------------------------

            response = await session.list_tools()

            # --------------------------------------------------
            # 6. Print the tools
            # --------------------------------------------------

            print("\nAvailable MCP Tools:\n")

            for tool in response.tools:

                print("Tool name:", tool.name)
                print("Description:", tool.description)
                print("Input schema:", tool.input_schema)
                print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())

