import asyncio
from pathlib import Path

import ollama

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)


async def main():

    # ============================================================
    # 1. Locate MCP Server
    # ============================================================

    client_directory = Path(__file__).parent

    server_path = client_directory / "Servers" / "filesystem_server.py"

    print("MCP Server:", server_path)

    # ============================================================
    # 2. Configure MCP Server
    # ============================================================

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    )

    # ============================================================
    # 3. Start MCP Server
    # ============================================================

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # ====================================================
            # 4. Initialize MCP
            # ====================================================

            await session.initialize()

            print("\nMCP connection established!")

            # ====================================================
            # 5. Get tools from MCP Server
            # ====================================================

            response = await session.list_tools()

            print("\nMCP Tools:")

            for tool in response.tools:
                print(" -", tool.name)

            # ====================================================
            # 6. Convert MCP tools to Ollama format
            # ====================================================

            ollama_tools = []

            for tool in response.tools:

                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                )

            print("\nTools sent to Qwen:")

            for tool in ollama_tools:
                print(" -", tool["function"]["name"])

            # ====================================================
            # 7. Conversation
            # ====================================================

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant. "
                        "Use the available tools whenever they "
                        "are appropriate."
                    ),
                }
            ]

            while True:

                user_input = input("\nYou: ")

                if user_input.lower() in ["exit", "quit"]:
                    break

                messages.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # =================================================
                # 8. Ask Qwen
                # =================================================

                response = ollama.chat(
                    model="qwen3:14b",
                    messages=messages,
                    tools=ollama_tools,
                )

                assistant_message = response["message"]

                # =================================================
                # 9. Add Qwen response to conversation
                # =================================================

                messages.append(assistant_message)

                # =================================================
                # 10. Check whether Qwen requested a tool
                # =================================================

                tool_calls = assistant_message.get("tool_calls", [])

                if not tool_calls:

                    print("\nAI:", assistant_message.get("content", ""))

                    continue

                # =================================================
                # 11. Execute MCP tool
                # =================================================

                for tool_call in tool_calls:

                    tool_name = tool_call["function"]["name"]

                    tool_arguments = tool_call["function"]["arguments"]

                    print("\nQwen requested MCP tool:")
                    print(" Tool:", tool_name)
                    print(" Arguments:", tool_arguments)

                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_arguments,
                    )

                    print("\nMCP result:")
                    print(result)

                    # =============================================
                    # 12. Send MCP result back to Qwen
                    # =============================================

                    messages.append(
                        {
                            "role": "tool",
                            "content": str(result),
                        }
                    )

                # =================================================
                # 13. Ask Qwen for final answer
                # =================================================

                final_response = ollama.chat(
                    model="qwen3:14b",
                    messages=messages,
                )

                final_message = final_response["message"]

                messages.append(final_message)

                print("\nAI:", final_message.get("content", ""))


if __name__ == "__main__":
    asyncio.run(main())
