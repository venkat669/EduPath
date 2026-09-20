import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

async def main():

    # Get the directory containing this client file
    client_directory = Path(__file__).parent
   
    # Move from:
    # MCP/Clients
    #
    # to:
    # MCP/Servers/filesystem_server.py
    server_path = (
        client_directory.parent
        / "Servers"
        / "filesystem_server.py"
    )

    print(f"Starting MCP server: {server_path}")
    print(f"Client directory: {client_directory}")
    print(f"Server path: {server_path}")

    # Tell MCP how to start the server
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    ) 

     #python .\Servers\filesystem_server.py  -> This is the command that is given in the powerShell Cli , it will start the MCP Server . I am constructing this and executing it in the code 
    print(f"Server Params {server_params}")
    # Start the MCP server and create stdio communication
    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):

        # Create an MCP session
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            print("Initializing MCP session...")

            # MCP handshake
            await session.initialize()

            print("Connected to MCP server!")

            # Discover tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:
                print(f" \n - {tool.name}: {tool.description}   - {tool}   \n\n")

            # Call the list_files tool
            result = await session.call_tool(
                "list_files",
                {}
            )

            print("\nTool result:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
