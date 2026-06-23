from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-d195bef79a94410686e71c7002aeb050",
    base_url="https://api.deepseek.com/v1",
)
agent = create_deep_agent(
    model=model,
    skills=["./skills"],  # 指向刚才建的技能文件夹
    system_prompt="你是一个科研助理，能帮用户找最新的学术论文。"
)

response = agent.invoke({"messages": [HumanMessage(content="帮我找三篇关于 LangChain Deep Agents 的最新论文")]})
print(response["messages"][-1].content)