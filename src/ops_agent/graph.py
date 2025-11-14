import os
import json
from typing import Dict, List, Optional, TypedDict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import get_llm
from .prompts import SYSTEM_PROMPT, REFLECT_PROMPT
from .shell import ShellExecutor
from .config import RuntimeConfig
from .visual import render_graph_status, render_preview_command, render_execution_output

# 仅用于类型注解，避免在本地CLI环境下缺失时报错
try:
    from langgraph_api.runnable import RunnableConfig  # type: ignore
except Exception:  # pragma: no cover
    RunnableConfig = Any  # 动态兼容，无强制依赖


class AgentState(TypedDict):
    user_input: str
    history: List[Dict[str, str]]  # 记录简单对话与观察
    candidate: Dict[str, str]      # {command, rationale, visual_hint, risk}
    approval: Optional[bool]
    observation: str
    continue_flag: bool
    # 新增：中断相关字段与决策路由
    decision: Optional[str]  # "approve" | "reject" | "interrupt"
    interrupt_text: Optional[str]


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


def build_app(config: RunnableConfig):
    """LangGraph CLI 工厂函数：必须接收一个参数(RunnableConfig)。

    在此处根据 config.metadata 或环境变量构建执行器与图。
    """

    def _meta(key: str, default: Any = None) -> Any:
        try:
            md = getattr(config, "metadata", {}) or {}
            return md.get(key, default)
        except Exception:
            return default

    # 从 metadata 或环境变量推导运行配置
    execution_mode = _meta("execution_mode", os.environ.get("OPS_EXECUTION_MODE", "dry_run"))
    timeout_seconds = int(_meta("timeout_seconds", os.environ.get("OPS_TIMEOUT_SECONDS", 60)))
    ssh_host = _meta("ssh_host", os.environ.get("OPS_SSH_HOST"))
    ssh_user = _meta("ssh_user", os.environ.get("OPS_SSH_USER"))
    ssh_key_path = _meta("ssh_key_path", os.environ.get("OPS_SSH_KEY_PATH"))
    ssh_port = int(_meta("ssh_port", os.environ.get("OPS_SSH_PORT", 22)))
    non_interactive_decline = bool(_meta(
        "non_interactive_decline",
        os.environ.get("OPS_NON_INTERACTIVE_DECLINE", "0") in ("1", "true", "True")
    ))

    runtime_cfg = RuntimeConfig(
        execution_mode=execution_mode,
        timeout_seconds=timeout_seconds,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key_path=ssh_key_path,
        ssh_port=ssh_port,
    )
    executor = ShellExecutor(runtime_cfg)
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
            # 非交互模式直接拒绝
            approve = False
            decision = "reject"
        else:
            ans = input("确认执行该命令吗？[y/N/i] (y=执行, n=拒绝, i=中断沟通): ").strip().lower()
            if ans in ("y", "yes"):
                approve = True
                decision = "approve"
            elif ans in ("i", "interrupt"):
                approve = None
                decision = "interrupt"
            else:
                approve = False
                decision = "reject"
        return {"approval": approve, "decision": decision}

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
    # 不再使用单独的中断节点，改为在确认处直接结束本轮

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "confirm")

    # 确认后的条件路由：批准->执行；拒绝->反思
    graph.add_conditional_edges(
        "confirm",
        lambda s: s.get("decision"),
        {"approve": "execute", "reject": "reflect", "interrupt": END},
    )

    graph.add_edge("execute", "reflect")

    # 反思后的条件路由：继续->回到确认；停止->结束
    graph.add_conditional_edges(
        "reflect",
        lambda s: bool(s.get("continue_flag", False)),
        {True: "confirm", False: END},
    )

    # 中断后直接结束本轮，CLI将保留上下文以继续对话

    app = graph.compile()
    return app
