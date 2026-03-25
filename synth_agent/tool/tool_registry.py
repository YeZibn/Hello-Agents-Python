from synth_agent.tool.tool import Tool
from typing import List, Dict
from synth_agent.tool.tool_list.bash_tool import BashTool
import re



class ToolRegistry:
    """HelloAgents工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """注册Tool对象"""
        if tool.name in self._tools:
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get_tool_list(self) -> List[Tool]:
        """获取所有注册的Tool对象"""
        return list(self._tools.values())

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            params = tool.get_parameters()
            param_info = []
            for param in params:
                required = "必填" if param.required else "可选"
                param_info.append(f"{param.name}({param.type}, {required}): {param.description}")
            
            param_str = "\n    ".join(param_info) if param_info else "无参数"
            descriptions.append(f"## {tool.name}\n{tool.description}\n参数:\n    {param_str}{dep_str}")

        return "\n\n".join(descriptions) if descriptions else "暂无可用工具"

    def get_tools_schema(self) -> List[dict]:
        """
        获取所有工具的 OpenAI tools 格式 schema
        用于传递给 LLM 的 tools 参数
        """
        tools_schema = []
        for tool in self._tools.values():
            properties = {}
            required = []
            
            for param in tool.get_parameters():
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if param.required:
                    required.append(param.name)
            
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            tools_schema.append(tool_schema)
        
        return tools_schema

    def get_tool(self, name: str) -> Tool | None:
        """根据名称获取工具"""
        return self._tools.get(name)

    def execute_tool(self, tool_name: str, tool_input) -> str:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入，可以是字符串或字典
        """
        try:
            tool = self.get_tool(tool_name)
            if tool:
                # 如果输入是字典，直接使用
                if isinstance(tool_input, dict):
                    params = tool_input
                else:
                    # 解析字符串参数
                    params = self._parse_string_input(tool_name, tool_input, tool)
                
                # 验证必填参数
                tool_params = tool.get_parameters()
                for param in tool_params:
                    if param.required and param.name not in params:
                        return f"错误：工具 '{tool_name}' 缺少必填参数 '{param.name}'"
                
                # 执行工具
                result = tool.run(params)
                return result
            
            return f"错误：工具 '{tool_name}' 不存在"
        except Exception as e:
            return f"执行工具时出错: {str(e)}"
    
    def _parse_string_input(self, tool_name: str, tool_input: str, tool: Tool) -> dict:
        """解析字符串输入为参数字典"""
        params = {}
        if tool_input:
            # 检查是否包含键值对
            if "=" in tool_input:
                # 按逗号分割参数，但需要处理引号内的逗号
                import re
                # 使用正则表达式分割，考虑引号内的内容
                param_pairs = re.findall(r'([^=,\s]+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\s]+)', tool_input)
                
                for key, value in param_pairs:
                    # 去除引号
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    params[key.strip()] = value.strip()
            else:
                # 对于没有键值对的情况，根据工具类型进行特殊处理
                if tool_name == "bash":
                    # 对于bash工具，直接将输入作为command参数
                    params["command"] = tool_input
                else:
                    # 其他工具尝试智能推断参数
                    tool_params = tool.get_parameters()
                    if len(tool_params) == 1:
                        # 如果只有一个参数，直接使用
                        params[tool_params[0].name] = tool_input
                    else:
                        # 多参数但未指定参数名，报错
                        raise ValueError(f"工具 '{tool_name}' 需要指定参数名称，格式应为: {tool_name}[参数名=值]")
        
        return params

if __name__ == "__main__":
    response = "Action: Finish[当前项目的最外层结构如下：项目根目录位于 D:\agentWorkShop\Hello-Agents-Python，包含1个配置文件、2个代码目录和3个文件。]"


    finish_match = re.search(r'Action: Finish(.*)$', response)
    if finish_match:
        finish_answer = finish_match.group(1).strip()
        print(f"Finish: {finish_answer}")
        
   
