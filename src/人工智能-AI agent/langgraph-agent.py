import os
import json
import requests
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from typing import TypedDict
import operator

# ========== 1. 配置 ==========
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# llm = ChatOpenAI(model="gpt-4o-mini")

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"                                                                                              
llm = ChatOpenAI(                                                                                                                            
         model="deepseek-chat",  # 或 "deepseek-reasoner"                                                                                         
         temperature=0                                       
     )


# ========== 2. 定义工具 ==========
@tool
def search_web(query: str):
    """搜索互联网获取最新信息
    
    Args:
        query: 搜索关键词
    """
    print(f"[Tool] 正在搜索: {query}")
    
    # 这里用一个简化的实现
    # 实际项目可以接入真实的搜索API
    return f"关于'{query}'的最新搜索结果：[这里是模拟的搜索结果]"

@tool
def get_weather(city: str):
    """查询城市天气
    
    Args:
        city: 城市名称
    """
    print(f"[Tool] 正在查询天气: {city}")
    
    weather_data = {
        "北京": "北京今天16度，天气晴朗",
        "上海": "上海今天20度，多云",
        "深圳": "深圳今天28度，有雨"
    }
    
    return weather_data.get(city, f"抱歉，暂时没有{city}的天气信息")

@tool
def save_user_info(name: str, age: int, email: str, phone: str):
    """保存用户信息到数据库
    
    Args:
        name: 用户姓名
        age: 用户年龄
        email: 邮箱地址
        phone: 手机号
    """
    print(f"[Tool] 正在保存用户信息: {name}")
    
    # 实际项目这里应该是真实的数据库操作
    print(f"  - 姓名: {name}")
    print(f"  - 年龄: {age}")
    print(f"  - 邮箱: {email}")
    print(f"  - 手机: {phone}")
    
    return f"✅ 已成功保存 {name} 的信息"

# ========== 3. 创建工具节点 ==========
tools = [search_web, get_weather, save_user_info]
tool_node = ToolNode(tools)
# 绑定工具到模型
llm_with_tools = llm.bind_tools(tools)

# ========== 4. 定义图的状态 ==========
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# ========== 5. 定义节点函数 ==========
def call_model(state):
    """调用大模型，让它决定要不要用工具"""
    print(f"\n[call_model] 收到消息: {state['messages'][-1].content}")
    
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    
    # 检查AI是否要调用工具
    if response.tool_calls:
        print(f"[call_model] AI决定调用工具: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print("[call_model] AI决定直接回答")
    
    return {"messages": [response]}

# ========== 6. 定义路由函数 ==========
def should_continue(state: AgentState):
    """判断是否需要调用工具"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果AI生成了tool_calls，就去执行工具
    if last_message.tool_calls:
        return "tools"
    # 否则直接结束
    return END

# ========== 7. 构建图 ==========
workflow = StateGraph(AgentState)
# 添加节点
workflow.add_node("agent", call_model)  # AI决策节点
workflow.add_node("tools", tool_node)    # 工具执行节点
# 设置入口
workflow.add_edge(START, "agent")
# 添加条件边：AI决定后，要么调用工具，要么结束
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
# 工具执行完后，回到AI节点让它总结结果
workflow.add_edge("tools", "agent")
# 编译
graph = workflow.compile()

# ========== 8. 测试函数 ==========
def test_agent(query):
    print("\n" + "="*70)
    print(f"👤 用户: {query}")
    print("="*70)
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        {"recursion_limit": 10}  # 防止无限循环
    )
    
    final_answer = result["messages"][-1].content
    print(f"\n🤖 助手: {final_answer}")
    print("="*70 + "\n")
    
# ========== 9. 运行测试 ==========
if __name__ == "__main__":
    # 测试1：普通对话
    test_agent("你好，请介绍一下你自己")
    
    # 测试2：天气查询
    test_agent("北京今天天气怎么样？")
    
    # 测试3：联网搜索
    test_agent("Claude 4.5 Sonnet有什么新功能？")
    
    # 测试4：用户信息
    test_agent("我叫李四，25岁，邮箱lisi@example.com，手机13987654321")
    
    # 测试5：复杂任务（可能调用多个工具）
    test_agent("帮我查一下上海的天气，然后搜索一下最近的AI新闻")