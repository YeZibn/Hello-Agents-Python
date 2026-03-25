from typing import Optional, List, Dict
from synth_agent.agent.agent import Agent
from synth_agent.config import rag_config
from synth_agent.llm.synth_LLM import SynthLLM
from synth_agent.config.config import Config
from synth_agent.memory.memory_tool import MemoryTool
from synth_agent.tool.tool_registry import ToolRegistry
import json
from synth_agent.tool.tool_list.bash_tool import BashTool
from synth_agent.tool.tool_list.write_tool import WriteTool
from synth_agent.tool.tool_list.read_tool import ReadTool
from synth_agent.config.memory_config import MemoryConfig
from synth_agent.rag.rag_tool import RAGTool
from synth_agent.tool.tool_list.web.baidu_search_tool import BaiduSearchTool
from synth_agent.tool.tool_list.web.url_search_tool import UrlSearchTool



DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，能够通过推理和使用工具来解决问题。

## 工作原则
1. 仔细分析用户的问题，确定需要什么信息或操作
2. 如果需要外部知识，优先使用RAG工具获取相关文档信息
3. 如果需要用户历史信息或上下文，使用记忆工具检索相关记忆
4. 根据工具返回的结果继续推理，直到能够回答用户的问题
5. 当你有足够的信息时，直接给出最终答案

## 工具使用策略
- **RAG工具**：用于获取结构化知识，特别是关于产品、服务、技术等方面的信息
- **记忆工具**：用于获取用户历史对话、偏好、之前提到的信息
- **优先顺序**：先尝试从记忆中获取上下文，再使用RAG获取外部知识
- 每次只调用必要的工具，避免冗余调用
- 根据工具返回的结果调整你的策略

## 记忆管理
- 当用户提供重要信息（如偏好、需求、个人信息等）时，应主动使用记忆工具存储
- 存储记忆时，选择合适的记忆类型（semantic/episodic/working）
- 为记忆添加适当的标签和重要性评分

## 回答要求
- 用中文回答
- 回答要简洁、准确、有帮助
- 基于工具返回的结果来回答，确保信息的准确性
- 当使用记忆或RAG信息时，明确引用来源，增强回答的可信度"""


class ReActAgent(Agent):
    """
    ReAct Agent - 推理与行动结合的智能体
    使用 LLM 原生 tool_calls 功能
    """

    def __init__(
        self,
        name: str,
        llm: SynthLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 100
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[Dict] = []
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReAct Agent
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            result = self._run_step(input_text)
            
            if result is not None:
                return result

        return "抱歉，我无法在限定步数内完成这个任务。"

    def _build_messages(self, input_text: str) -> List[Dict]:
        """
        构建消息列表
        """
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        system_content = (self.system_prompt or DEFAULT_SYSTEM_PROMPT)
        system_content = system_content.replace("{{current_time}}", current_time)

        messages = [
            {"role": "system", "content": system_content + f"\n\n当前时间: {current_time}"}
        ]
        
        messages.extend(self.current_history)
        
        if not self.current_history:
            messages.append({"role": "user", "content": input_text})
        
        return messages

    def _run_step(self, input_text: str) -> Optional[str]:
        """
        执行单步推理
        根据 finish_reason 判断：
        - "stop": 正常结束，返回内容
        - "tool_calls": 需要执行工具调用
        """
        tools_schema = self.tool_registry.get_tools_schema()
        messages = self._build_messages(input_text)

        llm_with_tools = SynthLLM(
            model=self.llm.model,
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
            timeout=self.llm.timeout,
            tools=tools_schema
        )
        
        response = llm_with_tools.think(messages, temperature=self.config.temperature)


        if not response:
            return "抱歉，我无法处理这个请求。"

        if response.get("full_reasoning"):
            print(f"🧠 思考: {response['full_reasoning']}")

        raw_chunks = response.get("raw_chunks", [])
        finish_reason = self._get_finish_reason(raw_chunks)
        
        print(f"📌 Finish Reason: {finish_reason}")

        if finish_reason == "stop":
            content = response.get("full_content", "")
            if content:
                print(f"📝 最终回答: {content}")
                self.current_history.append({"role": "assistant", "content": content})
                return content
            return None

        if finish_reason == "tool_calls":
            tool_calls = self._extract_tool_calls(raw_chunks)
            
            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "reasoning": response.get("full_reasoning") or None,
                    "content": response.get("full_content") or None,
                    "tool_calls": tool_calls
                }
                self.current_history.append(assistant_message)
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                    tool_call_id = tool_call.get("id", "")
                    
                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    print(f"🔧 工具调用: {tool_name}({tool_args})")
                    
                    # 直接传递字典参数，避免类型转换问题
                    observation = self.tool_registry.execute_tool(tool_name, tool_args)
                    print(f"👁️  观察: {observation}")
                    
                    tool_result_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(observation)
                    }
                    self.current_history.append(tool_result_message)
                
                return None

        return None

    def _get_finish_reason(self, raw_chunks: List[dict]) -> Optional[str]:
        """
        从原始响应块中获取 finish_reason
        """
        for chunk in reversed(raw_chunks):
            choices = chunk.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    return finish_reason
        return None

    def _extract_tool_calls(self, raw_chunks: List[dict]) -> List[dict]:
        """
        从原始响应块中提取 tool_calls
        """
        tool_calls = {}
        
        for chunk in raw_chunks:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            
            delta = choices[0].get("delta", {})
            chunk_tool_calls = delta.get("tool_calls", [])
            
            for tc in chunk_tool_calls:
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""}
                    }
                
                if tc.get("id"):
                    tool_calls[idx]["id"] = tc["id"]
                
                func = tc.get("function", {})
                if func.get("name"):
                    tool_calls[idx]["function"]["name"] = func["name"]
                if func.get("arguments"):
                    tool_calls[idx]["function"]["arguments"] += func["arguments"]
        
        return list(tool_calls.values())


if __name__ == "__main__":
    llm = SynthLLM()
    tool_registry = ToolRegistry()

    # bash_tool = BashTool()
    # read_tool = ReadTool()
    # write_tool = WriteTool()
    # memory_tool = MemoryTool(user_id="user_xiaohong", memory_config=memory_config)
    # rag_tool = RAGTool(user_id="user_xiaohong", rag_config=rag_config)
    baidu_search_tool = BaiduSearchTool()
    url_search_tool = UrlSearchTool()
    
    # tool_registry.register_tool(bash_tool)
    # tool_registry.register_tool(read_tool)
    # tool_registry.register_tool(write_tool)
    # tool_registry.register_tool(memory_tool)
    # tool_registry.register_tool(rag_tool)
    tool_registry.register_tool(baidu_search_tool)
    tool_registry.register_tool(url_search_tool)
    
    agent = ReActAgent(
        name="HelloAgent", 
        llm=llm, 
        tool_registry=tool_registry, 
        config=Config(),
        system_prompt="请借助网络搜索功能"
    )
    agent.run("张雪峰他死了吗")
