import platform
import shlex
import shutil
import subprocess
from typing import Tuple

from .config import RuntimeConfig


class ShellExecutor:
    """安全的Shell执行器，支持多种环境与干跑模式。"""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.env_mode = self._detect_env_mode(config.execution_mode)

    def _detect_env_mode(self, preferred: str) -> str:
        if preferred == "ssh":
            return "ssh"
        if preferred == "local_linux":
            return "local_linux"
        if preferred == "wsl":
            return "wsl" if shutil.which("wsl") else "dry_run"

        # 自动探测：Linux->local_linux；Windows且有WSL->wsl；否则dry_run
        system = platform.system().lower()
        if system == "linux":
            return "local_linux"
        if system == "windows" and shutil.which("wsl"):
            return "wsl"
        return "dry_run"

    def run(self, command: str) -> Tuple[int, str, str]:
        """执行命令并返回 (returncode, stdout, stderr)。在dry_run下不执行。"""
        if self.env_mode == "dry_run":
            preview = f"[DRY-RUN] 将执行: {command}"
            return 0, preview, ""

        if self.env_mode == "local_linux":
            cmd = ["bash", "-lc", command]
            return self._exec_process(cmd)

        if self.env_mode == "wsl":
            # 通过WSL执行bash
            wsl_cmd = ["wsl", "bash", "-lc", command]
            return self._exec_process(wsl_cmd)

        if self.env_mode == "ssh":
            return self._exec_ssh(command)

        return 1, "不支持的执行模式", ""

    def _exec_process(self, cmd_list):
        try:
            proc = subprocess.run(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.timeout_seconds,
                text=True,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or "", e.stderr or "命令执行超时"
        except Exception as e:
            return 1, "", f"执行错误: {e}"

    def _exec_ssh(self, command: str) -> Tuple[int, str, str]:
        # 仅在提供SSH配置时执行；否则退回dry-run
        from paramiko import SSHClient, AutoAddPolicy

        if not all([
            self.config.ssh_host,
            self.config.ssh_user,
            self.config.ssh_key_path,
        ]):
            preview = (
                "[DRY-RUN] SSH信息未配置(ssh_host/ssh_user/ssh_key_path)，将不执行: "
                + command
            )
            return 0, preview, ""

        try:
            client = SSHClient()
            client.set_missing_host_key_policy(AutoAddPolicy())
            client.connect(
                hostname=self.config.ssh_host,
                username=self.config.ssh_user,
                key_filename=self.config.ssh_key_path,
                port=self.config.ssh_port,
                timeout=self.config.timeout_seconds,
            )

            stdin, stdout, stderr = client.exec_command(command, timeout=self.config.timeout_seconds)
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            exit_status = stdout.channel.recv_exit_status()
            client.close()
            return exit_status, out, err
        except Exception as e:
            return 1, "", f"SSH执行错误: {e}"