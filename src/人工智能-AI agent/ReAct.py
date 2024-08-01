'''
如何在LangChain中使用ReAct构建AI Agent
'''


from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI

# 模型
# api_key = "xxx"
api_key = "xxx"
# api_key = "xxx"

model = ChatOpenAI(model="gpt-3.5-turbo",
                   openai_api_key=api_key,
                   openai_api_base="https://api.openai.com/v1")
# 直接让模型计算数字，模型会算错
model.invoke([HumanMessage(content="你帮我算下，3.941592623412424+4.3434532535353的结果")])


# 下面开始使用ReAct机制，定义工具，让LLM使用工具做专业的事情。

# 定义工具，要继承自LangChain的BaseTool
class SumNumberTool(BaseTool):
    name = "数字相加计算工具"
    description = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        return a.value + b.value

# 工具合集
tools = [SumNumberTool()]
# 提示词，直接从langchain hub上下载，因为写这个ReAct机制的prompt比较复杂，直接用现成的。
prompt = hub.pull("hwchase17/structured-chat-agent")
# 定义AI Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)
# 使用Memory记录上下文
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
# 定义AgentExecutor，必须使用AgentExecutor，才能执行代理定义的工具
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)
# 测试使用到工具的场景
agent_executor.invoke({"input": "你帮我算下3.941592623412424+4.3434532535353的结果"})

