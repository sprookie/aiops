from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


console = Console()


def render_graph_status(current: str, approval_needed: bool):
    """渲染简易节点可视化状态。"""
    stages = [
        ("分析", "analyze"),
        ("确认", "confirm"),
        ("执行", "execute"),
        ("反思", "reflect"),
    ]

    table = Table(title="运维Agent流程", expand=True)
    table.add_column("阶段")
    table.add_column("节点")
    table.add_column("状态")

    for label, node in stages:
        if node == current:
            status = "[bold green]进行中[/]"
        else:
            status = "待处理"
        if node == "confirm" and approval_needed:
            status = "[bold yellow]等待用户确认[/]"
        table.add_row(label, node, status)

    console.print(Panel(table, title="LangGraph 节点可视化", border_style="cyan"))


def render_preview_command(cmd: str, rationale: str, risk: str, visual_hint: str):
    hdr = Text("下一个命令预览", style="bold bright_white")
    body = Text()
    body.append(f"命令: {cmd}\n", style="bold green")
    body.append(f"理由: {rationale}\n")
    body.append(f"风险: {risk}\n", style="bold yellow")
    body.append(f"可视化建议: {visual_hint}\n", style="cyan")
    console.print(Panel(body, title=hdr, border_style="magenta"))


def render_execution_output(code: int, stdout: str, stderr: str):
    table = Table(title="命令执行回显", expand=True)
    table.add_column("键")
    table.add_column("值")
    table.add_row("returncode", str(code))
    table.add_row("stdout", stdout if stdout else "<空>")
    table.add_row("stderr", stderr if stderr else "<空>")
    console.print(Panel(table, title="执行结果", border_style="green"))