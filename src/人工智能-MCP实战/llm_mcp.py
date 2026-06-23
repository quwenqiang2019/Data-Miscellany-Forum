import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os

# 大模型请求地址
BASE_URL = "https://api.deepseek.com/v1"
API_KEY = "sk-xxxxxxxxxxxxxxxx"
MODEL_NAME = "deepseek-chat"

async def main():
    try: # 初始化MCP客户端 stdio 方式
        client = MultiServerMCPClient(
            {
                "Demo Server": 
                {
                "command":"python3",
                "args": [os.path.abspath("server.py")],  # 注意mcp的路径
                "transport":"stdio"
                } # 其它MCP   
            }
            )
        # 获取工具
        tools = await client.get_tools()
        if not tools:
            raise ValueError("未获取到任何工具")
        # 初始化一个 ChatOpenAI 实例，用于与大模型交互
        llm = ChatOpenAI(base_url=BASE_URL, openai_api_key=API_KEY, model=MODEL_NAME, timeout=60.0, max_retries=2)
        # 创建agent
        agent = create_react_agent(llm, tools)
        while True:  
            user_input = input("\n 请输入需求（或输入 exit 退出）：\n> ")
            if user_input.strip().lower() == "exit":
                break
            async for chunk in agent.astream({"messages": user_input}):
                print(chunk)
    except Exception as e:
        print(f"程序初始化失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())