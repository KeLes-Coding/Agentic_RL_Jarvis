# device_management/device_providers.py
import logging
import subprocess
import time
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

# 全局变量，用于追踪由此模块启动的SSH隧道进程
ssh_tunnel_processes = []


def get_ssh_tunnel_processes():
    """返回当前所有活跃的隧道进程。"""
    return ssh_tunnel_processes


class DeviceProvider(ABC):
    """设备提供者的抽象基类。"""

    def __init__(self, config: Dict[str, Any], adb_path: str):
        self.config = config
        self.adb_path = adb_path
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def find_devices(self) -> List[str]:
        """发现并返回此提供者可用的设备序列号列表。"""
        pass

    def run_adb_command(self, command: List[str], timeout: int = 20) -> Optional[str]:
        """一个通用的ADB命令执行函数"""
        try:
            full_command = [self.adb_path] + command
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                encoding="utf-8",
            )
            return result.stdout.strip()
        except FileNotFoundError:
            self.logger.error(
                f"错误: '{self.adb_path}' 命令未找到。请确保它在您的系统 PATH 中或在config.yaml中配置正确。"
            )
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"执行 '{' '.join(command)}' 时出错: {e}\n错误输出:\n{e.stderr}"
            )
            return None
        except subprocess.TimeoutExpired:
            self.logger.error(f"命令 '{' '.join(command)}' 超时。")
            return None
        except Exception as e:
            self.logger.error(f"执行ADB命令时发生未知错误: {e}")
            return None


class LocalDeviceProvider(DeviceProvider):
    """发现本地连接的设备。"""

    def find_devices(self) -> List[str]:
        if not self.config.get("enabled"):
            return []

        self.logger.info("正在从 [本地] 发现设备...")
        output = self.run_adb_command(["devices"])
        if not output:
            return []

        devices = []
        lines = output.strip().split("\n")
        for line in lines[1:]:
            if "\tdevice" in line:
                devices.append(line.split("\t")[0])
        return devices


class RemoteIPDeviceProvider(DeviceProvider):
    """通过IP地址发现远程设备。"""

    def find_devices(self) -> List[str]:
        if not self.config.get("enabled"):
            return []

        devices = []
        for remote in self.config.get("remotes", []):
            host = remote.get("host")
            if host:
                self.logger.info(f"正在连接到远程主机: {host}...")
                connect_output = self.run_adb_command(["connect", host])
                if connect_output and (
                    "connected" in connect_output
                    or "already connected" in connect_output
                ):
                    devices.append(host)
                else:
                    self.logger.error(f"连接到远程主机 {host} 失败。")
        return devices


class SSHReverseTunnelDeviceProvider(DeviceProvider):
    """通过SSH反向隧道发现设备。"""

    def find_devices(self) -> List[str]:
        if not self.config.get("enabled"):
            return []

        devices = []
        for tunnel in self.config.get("ssh_reverse_tunnels", []):
            port = tunnel.get("local_port")
            if port:
                address = f"localhost:{port}"
                self.logger.info(f"正在尝试通过反向隧道连接设备: {address}")
                connect_output = self.run_adb_command(["connect", address])
                if connect_output and (
                    "connected" in connect_output
                    or "already connected" in connect_output
                ):
                    devices.append(address)
                else:
                    self.logger.error(f"通过反向隧道连接到 {address} 失败。")
        return devices


class SSHForwardTunnelDeviceProvider(DeviceProvider):
    """通过SSH正向隧道发现设备，并提升连接稳定性。"""

    def find_devices(self) -> List[str]:
        if not self.config.get("enabled"):
            return []

        all_devices = []
        for conn in self.config.get("ssh_connections", []):
            user, host, port = conn["ssh_user"], conn["ssh_host"], conn["ssh_port"]
            remote_adb = conn["remote_adb_path"]
            local_port_counter = conn.get("local_start_port", 15555)

            self.logger.info(f"正在通过SSH [{user}@{host}:{port}] 发现远程设备...")

            ssh_options = [
                "-o",
                "ServerAliveInterval=60",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ConnectTimeout=10",
            ]

            remote_devices = self._discover_remote_devices(
                user, host, port, remote_adb, ssh_options
            )
            if not remote_devices:
                self.logger.warning(f"在远程服务器 [{host}] 上未发现可用的安卓虚拟机。")
                continue

            for serial, remote_port in remote_devices:
                local_address = self._establish_tunnel(
                    user,
                    host,
                    port,
                    local_port_counter,
                    remote_port,
                    serial,
                    ssh_options,
                )
                if local_address:
                    all_devices.append(local_address)
                local_port_counter += 1

        return all_devices

    def _discover_remote_devices(
        self, user, host, port, remote_adb, ssh_options
    ) -> List[Tuple[str, int]]:
        ssh_cmd = (
            ["ssh"]
            + ssh_options
            + [f"{user}@{host}", "-p", str(port), f"{remote_adb} devices"]
        )
        try:
            remote_output = subprocess.check_output(
                ssh_cmd, text=True, timeout=30, encoding="utf-8"
            )
        except Exception as e:
            self.logger.error(f"通过SSH执行远程ADB命令失败: {e}")
            return []

        remote_devices = []
        for line in remote_output.strip().split("\n")[1:]:
            if "\tdevice" in line:
                serial = line.split("\t")[0]
                match = re.match(r"emulator-(\d+)", serial)
                if match:
                    remote_port = int(match.group(1)) + 1
                    remote_devices.append((serial, remote_port))
                elif ":" in serial:
                    try:
                        remote_port = int(serial.split(":")[-1])
                        remote_devices.append((serial, remote_port))
                    except ValueError:
                        self.logger.warning(f"无法从远程设备 '{serial}' 解析端口。")
        return remote_devices

    def _establish_tunnel(
        self, user, host, port, local_port, remote_port, serial, ssh_options
    ) -> Optional[str]:
        self.logger.info(f"为远程设备 '{serial}' (端口:{remote_port}) 建立隧道...")

        tunnel_cmd = (
            ["ssh", "-N", "-f"]
            + ssh_options
            + [
                "-L",
                f"{local_port}:localhost:{remote_port}",
                f"{user}@{host}",
                "-p",
                str(port),
            ]
        )

        try:
            process = subprocess.Popen(tunnel_cmd)
            ssh_tunnel_processes.append(process)
            self.logger.info(
                f"已启动SSH隧道 (PID: {process.pid}): 本地端口 {local_port} -> 远程端口 {remote_port}"
            )
            time.sleep(2)
        except Exception as e:
            self.logger.error(f"启动SSH隧道失败: {e}")
            return None

        local_address = f"localhost:{local_port}"
        connect_output = self.run_adb_command(["connect", local_address])
        if connect_output and (
            "connected" in connect_output or "already connected" in connect_output
        ):
            self.logger.info(
                f"成功通过隧道连接到设备: {local_address} (远程: {serial})"
            )
            return local_address
        else:
            self.logger.error(f"连接到 {local_address} 失败。请检查隧道或远程ADB服务。")
            process.terminate()
            return None
