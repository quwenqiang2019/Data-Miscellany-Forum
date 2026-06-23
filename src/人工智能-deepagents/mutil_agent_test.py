import warnings
from typing import Annotated, Literal, NotRequired, TypedDict
# 导入Deep Agents的核心组件
from deepagents import create_deep_agent
from deepagents.graph import CompiledSubAgent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
warnings.filterwarnings("ignore", message=".*ddgs.*")
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# 定义网络搜索工具，给政策专家用，用来查最新的政策
@tool
def internet_search(query: str) -> str:
    """使用 DuckDuckGo 进行网络搜索，查询最新的政策、新闻等信息"""
    print(f"\n[工具调用] internet_search(query='{query}')")
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"- [{r['title']}]({r['href']}): {r['body'][:200]}")
        if not results:
            return "未找到相关结果。"
        return "\n".join(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 定义行政审批专家的状态结构
class ExpertState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    expert_output: NotRequired[str]



# --- 子智能体 1: 行政文秘专家 (使用字典式SubAgent，最简单的那种) ---
# 就一个字典，5行配置，搞定！
secretary_expert = {
    "name": "行政文秘专家",
    "description": "精通公文写作和文档处理，负责各类行政文书、方案报告和公函的撰写",
    "system_prompt": (
        "你是一位行政文秘专家。请在回答开头加上'[子智能体: 行政文秘专家]'。\n"
        "你具备强大的文字写作和文档编辑能力，可以：\n"
        "1. 撰写各类行政公文、通知、报告\n"
        "2. 制定工作方案、计划总结\n"
        "3. 编写会议纪要、调研报告\n"
        "专注于用专业的文笔完成各类文书撰写任务。"
    ),
}

# --- 子智能体 2: 政策咨询专家 (使用CompiledSubAgent，带搜索工具的自定义Agent) ---
def create_policy_advisor_subagent(model):
    from langchain.agents import create_agent
    # 我们自己创建一个带搜索工具的Agent
    policy_graph = create_agent(
        model,
        tools=[internet_search],
        system_prompt=(
            "你是一位城市政策咨询专家。\n"
            "你必须为每个政策查询调用 internet_search 工具搜索相关政策文件。\n"
            "绝不能在没有搜索的情况下编造政策内容。\n"
            "当用户询问政策法规、补贴申请、办事流程等问题时：\n"
            "1. 立即使用 internet_search 工具搜索最新政策\n"
            "2. 基于搜索结果提供准确的政策信息和办事指南\n"
            "3. 回答开头必须加上'[子智能体: 政策咨询专家]'\n"
            "4. 如果搜索结果不确定，要明确告知用户以官方最新发布为准"
        )
    )
    # 把我们自定义的Agent，包装成CompiledSubAgent，主智能体就能调度它了！
    return CompiledSubAgent(
        name="政策咨询专家",
        description="负责城市政策法规咨询、补贴申请指引和办事流程指导",
        runnable=policy_graph
    )

# --- 子智能体 3: 行政审批专家 (使用CompiledSubAgent，自定义LangGraph工作流) ---
def create_approval_subagent(model):
    # 我们自己定义一个审批的工作流：接收申请 -> 审核材料 -> 审批完成
    def receive_node(state: ExpertState):
        """第一个节点：接收用户的申请"""
        query = state["messages"][-1].content
        print(f"\n[子智能体: 行政审批专家] 受理窗口 - 接收申请...")
        return {"expert_output": f"申请事项: {query}"}
    
    def review_node(state: ExpertState):
        """第二个节点：审核用户的材料"""
        output = state.get("expert_output", "")
        print(f"[子智能体: 行政审批专家] 审核窗口 - 审查材料...")
        return {"expert_output": output + " → 材料审核中"}
    
    def approve_node(state: ExpertState):
        """第三个节点：完成审批，返回结果"""
        output = state.get("expert_output", "")
        print(f"[子智能体: 行政审批专家] 审批窗口 - 办理审批...")
        return {"messages": [AIMessage(content=f"[子智能体: 行政审批专家] 办事流程指导：{output} → 审批完成。请携带身份证到市民中心窗口办理。")]}
    
    # 把节点串成LangGraph的图
    builder = StateGraph(ExpertState)
    builder.add_node("receive", receive_node)
    builder.add_node("review", review_node)
    builder.add_node("approve", approve_node)
    builder.add_edge(START, "receive")
    builder.add_edge("receive", "review")
    builder.add_edge("review", "approve")
    builder.add_edge("approve", END)
    
    # 把我们自定义的工作流图，包装成CompiledSubAgent
    return CompiledSubAgent(
        name="行政审批专家",
        description="负责行政审批事项咨询、SOP流程指导和办事指南服务",
        runnable=builder.compile()
    )

def run_city_brain_tests():
    # 初始化大模型
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-xxxxxxxxxxxx",
        base_url="https://api.deepseek.com/v1",
    )
    # 初始化两个自定义的子智能体
    policy_expert = create_policy_advisor_subagent(model)
    approval_expert = create_approval_subagent(model)
    
    # 创建主智能体：城市智慧大脑，把三个子智能体都传进去！
    main_agent = create_deep_agent(
        model=model,
        name="城市智慧大脑",
        system_prompt=(
            "你是城市智慧大脑智能体的核心调度中枢。\n"
            "根据用户问题类型，将任务分配给最合适的专家子智能体：\n"
            "- 行政文秘专家：处理公文写作、方案报告、文档撰写等文书相关问题\n"
            "- 政策咨询专家：处理政策法规、补贴申请、办事流程相关问题\n"
            "- 行政审批专家：处理行政审批事项、SOP流程和办事指南相关问题"
        ),
        subagents=[secretary_expert, policy_expert, approval_expert]
        
    )

# 测试用例，覆盖不同的场景
    test_cases = [
        # ("帮我写一份关于城市环境整治的工作报告", "应路由到 行政文秘专家"),
        # ("个人创业有什么补贴政策？", "应路由到 政策咨询专家"),
        # ("我想注册一家公司，需要准备什么材料？", "应路由到 行政审批专家"),
        # ("你好，今天天气不错", "闲聊 - 无需路由到子代理"),
        ("我需要开办一家餐馆，帮我写一份申请报告，再查下有什么补贴政策，最后告诉我办理流程", "需要多子代理协调：文秘(报告) -> 政策(补贴) -> 审批(流程)"),
    ]
    
    # 跑测试，看结果
    active_subagents = {}
    for i, (query, expected) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"测试用例 {i}: {expected}")
        print(f"用户问题: {query}")
        print(f"{'='*70}")
        # 流式调用主智能体，它会自动调度子智能体
        for chunk in main_agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2"
        ):
            print("chunk", chunk)
            # 这里我们可以监控子智能体的生命周期
            is_subagent = any(s.startswith("tools:") for s in chunk["ns"])
            if chunk["type"] == "updates":
                for node_name, data in chunk["data"].items():
                    if not chunk["ns"] and node_name == "model_request":
                        for msg in data.get("messages", []):
                            for tc in getattr(msg, "tool_calls", []):
                                if tc.get("name") == "task":
                                    sub_id = tc.get("id", "")
                                    sub_type = tc.get("args", {}).get("subagent_type", "unknown")
                                    active_subagents[sub_id] = {
                                        "type": sub_type,
                                        "description": tc.get("args", {}).get("description", "")[:50],
                                        "status": "pending"
                                    }
                                    print(f"\n[lifecycle] PENDING → 子智能体 '{sub_type}'")

if __name__ == "__main__":
    run_city_brain_tests()
