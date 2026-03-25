from synth_agent.tool.tool_list.web.base_web_tool import BaseWebTool, ToolParameter
from typing import Dict, Any, List


class UrlSearchTool(BaseWebTool):
    def __init__(self):
        super().__init__(name="url_search", description="根据URL获取网页内容")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }

    def run(self, parameters: Dict[str, Any]) -> str:
        url = parameters.get("url", "")
        if not url:
            return "错误: 请提供URL地址"

        result = self._get(url)

        if not result.get("success"):
            return f"获取网页失败: {result.get('error', '未知错误')}"

        body = result.get("body", "")
        return self._extract_text(body)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type="string", description="目标URL地址", required=True)
        ]

    def _extract_text(self, html: str, max_length: int = 4000) -> str:
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'&\w+;', '', text)
        text = re.sub(r'&#\d+;', '', text)

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text if text else "页面内容为空或无法提取"
