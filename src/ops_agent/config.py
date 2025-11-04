from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """运行配置，用于控制执行安全与环境。

    execution_mode:
      - "dry_run": 仅展示命令不执行（默认，适合Windows本地演示）
      - "local_linux": 在本机Linux下执行（通过bash）
      - "wsl": 在WSL环境中执行（通过wsl bash）
      - "ssh": 通过SSH在远程Linux执行
    """

    execution_mode: str = "dry_run"
    safe_confirm: bool = True
    timeout_seconds: int = 60

    # SSH 配置（当 execution_mode == "ssh" 时使用）
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_port: int = 22