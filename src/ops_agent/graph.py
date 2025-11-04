import json
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import get_llm
from .prompts import SYSTEM_PROMPT, REFLECT_PROMPT
from .shell import ShellExecutor
from .visual import render_graph_status, render_preview_command, render_execution_output


class AgentState(TypedDict):
    user_input: str
    history: List[Dict[str, str]]  # 记录简单对话与观察
    candidate: Dict[str, str]      # {command, rationale, visual_hint, risk}
    approval: Optional[bool]
    observation: str
    continue_flag: bool


def _safe_json_loads(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        # 尝试截取第一个JSON片段
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return {"command": "", "rationale": "解析失败", "visual_hint": "", "risk": "unknown"}


def build_app(executor: ShellExecutor, non_interactive_decline: bool = False):
    llm = get_llm()

    def analyze_node(state: AgentState):
        render_graph_status("analyze", approval_needed=False)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"用户目标/命令: {state['user_input']}\n历史: {state.get('history', [])}"),
        ]
        try:
            ai = llm.invoke(messages)
            data = _safe_json_loads(ai.content)
        except Exception as e:
            data = {
                "command": "uptime",
                "rationale": f"LLM调用失败，使用安全只读命令。错误: {e}",
                "visual_hint": "用列展示负载与时间",
                "risk": "low",
            }
        candidate = {
            "command": data.get("command", ""),
            "rationale": data.get("rationale", ""),
            "visual_hint": data.get("visual_hint", ""),
            "risk": data.get("risk", "low"),
        }
        return {"candidate": candidate}

    def confirm_node(state: AgentState):
        render_graph_status("confirm", approval_needed=True)
        c = state["candidate"]
        render_preview_command(
            c.get("command", ""),
            c.get("rationale", ""),
            c.get("risk", "low"),
            c.get("visual_hint", ""),
        )
        if non_interactive_decline:
            approve = False
        else:
            ans = input("确认执行该命令吗？[y/N]: ").strip().lower()
            approve = ans in ("y", "yes")
        return {"approval": approve}

    def execute_node(state: AgentState):
        render_graph_status("execute", approval_needed=False)
        cmd = state["candidate"].get("command", "")
        if not cmd:
            return {"observation": "没有可执行命令"}
        code, stdout, stderr = executor.run(cmd)
        render_execution_output(code, stdout, stderr)
        obs = f"returncode={code}\nstdout=\n{stdout}\nstderr=\n{stderr}"
        return {"observation": obs}

    def reflect_node(state: AgentState):
        render_graph_status("reflect", approval_needed=False)
        history = state.get("history", [])
        history.append({"user": state["user_input"]})
        history.append({"observation": state.get("observation", "")})

        messages = [
            SystemMessage(content=REFLECT_PROMPT),
            HumanMessage(content=json.dumps({
                "history": history,
                "last_candidate": state.get("candidate", {}),
            }, ensure_ascii=False)),
        ]
        try:
            ai = llm.invoke(messages)
            data = _safe_json_loads(ai.content)
            cont = bool(data.get("continue", False))
            next_cmd = data.get("next_command", "")
        except Exception as e:
            cont = False
            data = {
                "next_command": "",
                "rationale": f"LLM调用失败，建议停止。错误: {e}",
                "visual_hint": "",
                "risk": "low",
            }
            next_cmd = ""
        candidate = {
            "command": next_cmd,
            "rationale": data.get("rationale", ""),
            "visual_hint": data.get("visual_hint", ""),
            "risk": data.get("risk", "low"),
        }
        return {
            "continue_flag": cont,
            "candidate": candidate,
            "history": history,
        }

    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "confirm")

    # 确认后的条件路由：批准->执行；拒绝->反思
    graph.add_conditional_edges(
        "confirm",
        lambda s: bool(s.get("approval", False)),
        {True: "execute", False: "reflect"},
    )

    graph.add_edge("execute", "reflect")

    # 反思后的条件路由：继续->回到确认；停止->结束
    graph.add_conditional_edges(
        "reflect",
        lambda s: bool(s.get("continue_flag", False)),
        {True: "confirm", False: END},
    )

    app = graph.compile()
    return app