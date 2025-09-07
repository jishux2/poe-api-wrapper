from dataclasses import dataclass
from typing import Optional

@dataclass
class ProxyConfig:
    ip: str
    port: str
    protocol: str = "http"  # 默认使用http协议

    def get_url(self) -> str:
        return f"{self.protocol}://{self.ip}:{self.port}"

def create_v2ray_proxy() -> list[ProxyConfig]:
    """创建默认的V2rayN代理配置"""
    return [
        ProxyConfig(
            ip="127.0.0.1",
            port="10809",
            protocol="http"
        )
    ]