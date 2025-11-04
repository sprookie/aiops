import os
import sys
from typing import Dict

from rich.console import Console
from rich.panel import Panel
import typer
from dotenv import load_dotenv

from .config import RuntimeConfig
from .shell import ShellExecutor
from .graph import build_app
from .llm import DEFAULT_BASE_URL


console = Console()


def _set_deepseek_env():
    # 加载项目根目录的.env
    load_dotenv()
    # 为OpenAI兼容客户端设置环境变量，优先使用DEEPSEEK_API_KEY（但不再提供任何默认密钥）
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    # 基础URL，如未设置则使用DeepSeek默认地址
    if "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ.get("DEEPSEEK_API_BASE", DEFAULT_BASE_URL)


def run(
    execution_mode: str = typer.Option(
        "dry_run", help="执行模式: dry_run/local_linux/wsl/ssh"
    ),
    ssh_host: str = typer.Option(None, help="SSH主机"),
    ssh_user: str = typer.Option(None, help="SSH用户"),
    ssh_key_path: str = typer.Option(None, help="SSH私钥路径"),
    timeout: int = typer.Option(60, help="命令超时时间(秒)"),
    once: str = typer.Option(None, help="单次输入后退出"),
):
    """启动交互式运维Agent。"""

    _set_deepseek_env()

    cfg = RuntimeConfig(
        execution_mode=execution_mode,
        timeout_seconds=timeout,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key_path=ssh_key_path,
    )
    executor = ShellExecutor(cfg)
    app = build_app(executor, non_interactive_decline=bool(once))

    if once:
        user_input = once.strip()
        state: Dict = {
            "user_input": user_input,
            "history": [],
            "candidate": {"command": "", "rationale": "", "visual_hint": "", "risk": "low"},
            "approval": None,
            "observation": "",
            "continue_flag": False,
        }
        try:
            final = app.invoke(state)
            console.print(Panel(str(final), title="最终状态", border_style="cyan"))
        except Exception as e:
            console.print(Panel(f"执行出错: {e}", title="错误", border_style="red"))
        return

    console.print(Panel("Linux 运维Agent已启动。输入你的目标或命令（输入 exit 退出）", title="DeepSeek LangGraph 运维Agent"))

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n已退出。")
            break

        if user_input.lower() in ("exit", "quit"):  # 退出
            console.print("再见！")
            break

        # 初始状态
        state: Dict = {
            "user_input": user_input,
            "history": [],
            "candidate": {"command": "", "rationale": "", "visual_hint": "", "risk": "low"},
            "approval": None,
            "observation": "",
            "continue_flag": False,
        }

        # 运行一次完整流程（LangGraph会根据边自动流转）
        try:
            final = app.invoke(state)
            console.print(Panel(str(final), title="最终状态", border_style="cyan"))
        except Exception as e:
            console.print(Panel(f"执行出错: {e}", title="错误", border_style="red"))


if __name__ == "__main__":
    typer.run(run)