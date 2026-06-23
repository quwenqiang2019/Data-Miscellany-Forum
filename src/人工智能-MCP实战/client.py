from fastmcp import Client
import asyncio

async def main():
    async with Client("server.py") as client: 
        tools = await client.list_tools() 
        print("Available tools:", tools) 
        result = await client.call_tool("add", {"a": 5, "b": 7}) 
        print("Result:", result.content[0].text)
        
asyncio.run(main())