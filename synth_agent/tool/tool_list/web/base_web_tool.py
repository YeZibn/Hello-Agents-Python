from synth_agent.tool.tool import Tool, ToolParameter
from typing import Dict, Any, List, Optional
from abc import abstractmethod
import urllib.request
import urllib.error
import json


class BaseWebTool(Tool):
    """Web工具基类"""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.timeout = 30
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        pass

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type="string", description="目标URL地址", required=True),
            ToolParameter(name="timeout", type="integer", description="请求超时时间(秒)", required=False, default=30)
        ]

    def _validate_url(self, url: str) -> bool:
        return url.startswith(("http://", "https://"))

    def _make_request(self, url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        if not self._validate_url(url):
            return {"success": False, "error": "无效的URL格式"}

        merged_headers = self.headers.copy()
        if headers:
            merged_headers.update(headers)

        request_headers = merged_headers.copy()

        try:
            if method == "GET":
                req = urllib.request.Request(url, headers=request_headers)
            elif method == "POST":
                if "Content-Type" not in request_headers:
                    request_headers["Content-Type"] = "application/json"
                json_data = json.dumps(data).encode("utf-8") if data else None
                req = urllib.request.Request(url, data=json_data, headers=request_headers, method="POST")
            else:
                return {"success": False, "error": f"不支持的请求方法: {method}"}

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return {
                    "success": True,
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "body": response.read().decode("utf-8")
                }
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"HTTP错误: {e.code} - {e.reason}"}
        except urllib.error.URLError as e:
            return {"success": False, "error": f"网络错误: {str(e.reason)}"}
        except Exception as e:
            return {"success": False, "error": f"请求失败: {str(e)}"}

    def _get(self, url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
        return self._make_request(url, method="GET", headers=headers)

    def _post(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        return self._make_request(url, method="POST", data=data, headers=headers)
