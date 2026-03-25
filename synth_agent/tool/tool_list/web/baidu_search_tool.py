from synth_agent.tool.tool_list.web.base_web_tool import BaseWebTool, ToolParameter
from typing import Dict, Any, List


class BaiduSearchTool(BaseWebTool):
    def __init__(self):
        super().__init__(name="baidu_search", description="使用百度搜索引擎搜索关键词")

    def run(self, parameters: Dict[str, Any]) -> str:
        query = parameters.get("query", "")
        if not query:
            return "错误: 请提供搜索关键词"

        try:
            from baidusearch.baidusearch import search
            results = search(query)
            if not results:
                return "未找到搜索结果，请尝试其他关键词"

            output = []
            for i, item in enumerate(results[:10], 1):
                title = item.get("title", "")
                url = item.get("url", "")
                if title:
                    output.append(f"{i}. {title}")
                    if url:
                        output.append(f"   链接: {url}")

            return "百度搜索结果:\n" + "\n".join(output) if output else "未找到搜索结果，请尝试其他关键词"

        except ImportError:
            return "错误: 请先安装 baidusearch 库 (pip install baidusearch)"
        except Exception as e:
            return f"搜索失败: {str(e)}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string", description="搜索关键词", required=True)
        ]
