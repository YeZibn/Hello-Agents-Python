from datetime import datetime
from typing import Optional, List
from synth_agent.config.context_config import ContextConfig
from synth_agent.memory.memory_tool import MemoryTool
from synth_agent.rag.rag_tool import RAGTool
from synth_agent.context.context_packet import ContextPacket
from synth_agent.message.message import Message
from synth_agent.embedder.qwen_embedder import QwenEmbedder
from synth_agent.memory.memory_manager import MemoryManager
from synth_agent.config.memory_config import MemoryConfig


class ContextBuilder:
    """上下文构建器"""

    def __init__(
        self, 
        user_id: str,
        config: ContextConfig = None
        ):

        """初始化上下文构建器
        
        Args:
            user_id: 用户ID
            memory_tool: 记忆工具
            rag_tool: RAG工具
            config: 上下文构建配置
        """
        self.user_id = user_id
        self.config = config or ContextConfig()
        self.memory_manager = MemoryManager(
            config=MemoryConfig(
                database_path="travel_memory.db",
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None
            ), 
            user_id=user_id,
            enable_episodic=True,
            enable_semantic=True)
        self.rag_tool = RAGTool(user_id)
        self.embedder = self._create_embedding_model()

    def _create_embedding_model(self):
        return QwenEmbedder()

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
        max_tokens: int = 4096
        ) -> str:
        """构建完整的上下文

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            custom_packets: 自定义信息包
            max_tokens: 最大 token 限制

        Returns:
            str: 构建好的上下文字符串
        """
        # 1. 汇集所有候选信息
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets
        )

        # 2. 选择最相关的信息包
        selected_packets = self._select(
            packets=packets,
            user_query=user_query,
            available_tokens=max_tokens
        )

        # 3. 组织成结构化上下文
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query
        )

        # 4. 压缩超限的上下文
        compressed_context = self._compress(
            context=structured_context,
            max_tokens=max_tokens
        )

        return compressed_context

    def _gather(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None
        ) -> List[ContextPacket]:
        """汇集所有候选信息

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            custom_packets: 自定义信息包

        Returns:
            List[ContextPacket]: 候选信息列表
        """
        packets = []

        # 1. 添加系统指令(最高优先级,不参与评分)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,  # 系统指令始终保留
                metadata={"type": "system_instruction", "priority": "high"}
            ))

        # 2. 从记忆系统检索相关记忆
        if self.memory_manager:
            try:
                episodic_memory_results = self.memory_manager.retrieve(
                    query=user_query,
                    memory_types=["episodic"]
                )

                semantic_memory_results = self.memory_manager.retrieve(
                    query=user_query,
                    memory_types=["semantic"]
                )

                memory_results = {}

                memory_results["memories"] = episodic_memory_results.get("episodic", []) + semantic_memory_results.get("semantic", [])

                # 解析记忆结果并转换为 ContextPacket
                memory_packets = self._parse_memory_results(memory_results, user_query)
                packets.extend(memory_packets)
            except Exception as e:
                print(f"[WARNING] 记忆检索失败: {e}")

        # 3. 从 RAG 系统检索相关知识
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action":"search",
                    "query":user_query,
                    "top_k":5
                })
                # 解析 RAG 结果并转换为 ContextPacket
                rag_packets = self._parse_rag_results(rag_results, user_query)
                packets.extend(rag_packets)
            except Exception as e:
                print(f"[WARNING] RAG 检索失败: {e}")

        # 4. 添加对话历史(仅保留最近的 N 条)
        if conversation_history:
            recent_history = conversation_history[-5:]  # 默认保留最近 5 条
            for msg in recent_history:
                packets.append(ContextPacket(
                    content=f"{msg.role}: {msg.content}",
                    timestamp=msg.timestamp if hasattr(msg, 'timestamp') else datetime.now(),
                    token_count=self._count_tokens(msg.content),
                    relevance_score=0.6,  # 历史消息的基础相关性
                    metadata={"type": "conversation_history", "role": msg.role}
                ))

        # 5. 添加自定义信息包
        if custom_packets:
            packets.extend(custom_packets)

        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str,
        available_tokens: int
        ) -> List[ContextPacket]:
        """选择最相关的信息包

        Args:
            packets: 候选信息包列表
            user_query: 用户查询(用于计算相关性)
            available_tokens: 可用的 token 数量

        Returns:
            List[ContextPacket]: 选中的信息包列表
        """
        # 1. 分离系统指令和其他信息
        system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

        # 2. 计算系统指令占用的 token
        system_tokens = sum(p.token_count for p in system_packets)
        remaining_tokens = available_tokens - system_tokens

        if remaining_tokens <= 0:
            print("[WARNING] 系统指令已占满所有 token 预算")
            return system_packets

        # 3. 为其他信息计算综合分数
        scored_packets = []
        for packet in other_packets:
            # 计算相关性分数(如果尚未计算)
            if packet.relevance_score == 0.5:  # 默认值,需要重新计算
                relevance = self._calculate_relevance(packet.content, user_query)
                packet.relevance_score = relevance

            # 计算新近性分数
            recency = self._calculate_recency(packet.timestamp)

            # 综合分数 = 相关性权重 × 相关性 + 新近性权重 × 新近性
            combined_score = (
                self.config.relevance_weight * packet.relevance_score +
                self.config.recency_weight * recency
            )

            # 过滤低于最小相关性阈值的信息
            if packet.relevance_score >= self.config.min_relevance:
                scored_packets.append((combined_score, packet))

        # 4. 按分数降序排序
        scored_packets.sort(key=lambda x: x[0], reverse=True)

        # 5. 贪心选择:按分数从高到低填充,直到达到 token 上限
        selected = system_packets.copy()
        current_tokens = system_tokens

        for score, packet in scored_packets:
            if current_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                # Token 预算已满,停止选择
                break

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包,共 {current_tokens} tokens")
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        """计算内容与查询的相关性

        使用向量相似度计算。

        Args:
            content: 内容文本
            query: 查询文本

        Returns:
            float: 相关性分数(0.0-1.0)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 生成向量
            content_embedding = self.embedder.encode([content])
            query_embedding = self.embedder.encode([query])
            
            # 计算余弦相似度
            similarity = cosine_similarity(content_embedding, query_embedding)[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"[WARNING] 向量相似度计算失败,使用备用方法: {e}")
            # 备用方法: 简单的关键词重叠算法
            content_words = set(content.lower().split())
            query_words = set(query.lower().split())

            if not query_words:
                return 0.0

            # Jaccard 相似度
            intersection = content_words & query_words
            union = content_words | query_words

            return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """计算时间近因性分数

        使用指数衰减模型,24小时内保持高分,之后逐渐衰减。

        Args:
            timestamp: 信息的时间戳

        Returns:
            float: 新近性分数(0.0-1.0)
        """
        import math

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        age_hours = (datetime.now() - timestamp).total_seconds() / 3600

        # 指数衰减:24小时内保持高分,之后逐渐衰减
        decay_factor = 0.1  # 衰减系数
        recency_score = math.exp(-decay_factor * age_hours / 24)

        return max(0.1, min(1.0, recency_score))  # 限制在 [0.1, 1.0] 范围内

    def _structure(self, selected_packets: List[ContextPacket], user_query: str) -> str:
        """将选中的信息包组织成结构化的上下文模板

        Args:
            selected_packets: 选中的信息包列表
            user_query: 用户查询

        Returns:
            str: 结构化的上下文字符串
        """
        # 按类型分组
        system_instructions = []
        evidence = []
        context = []

        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")

            if packet_type == "system_instruction":
                system_instructions.append(packet.content)
            elif packet_type in ["rag_result", "knowledge"]:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        # 构建结构化模板
        sections = []

        # [Role & Policies]
        if system_instructions:
            sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

        # [Task]
        sections.append(f"[Task]\n{user_query}")

        # [Evidence]
        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))

        # [Context]
        if context:
            sections.append("[Context]\n" + "\n".join(context))

        # [Output]
        sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")

        return "\n\n".join(sections)

    def _compress(self, context: str, max_tokens: int) -> str:
        """压缩超限的上下文

        Args:
            context: 原始上下文
            max_tokens: 最大 token 限制

        Returns:
            str: 压缩后的上下文
        """
        current_tokens = self._count_tokens(context)

        if current_tokens <= max_tokens:
            return context  # 无需压缩

        print(f"[ContextBuilder] 上下文超限({current_tokens} > {max_tokens}),执行压缩")

        # 分区压缩:保持结构完整性
        sections = context.split("\n\n")
        compressed_sections = []
        current_total = 0

        for section in sections:
            section_tokens = self._count_tokens(section)

            if current_total + section_tokens <= max_tokens:
                # 完整保留
                compressed_sections.append(section)
                current_total += section_tokens
            else:
                # 部分保留
                remaining_tokens = max_tokens - current_total
                if remaining_tokens > 50:  # 至少保留 50 tokens
                    # 简单截断(生产环境中可以使用 LLM 摘要)
                    truncated = self._truncate_text(section, remaining_tokens)
                    compressed_sections.append(truncated + "\n[... 内容已压缩 ...]")
                break

        compressed_context = "\n\n".join(compressed_sections)
        final_tokens = self._count_tokens(compressed_context)
        print(f"[ContextBuilder] 压缩完成: {current_tokens} -> {final_tokens} tokens")

        return compressed_context

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数量

        Args:
            text: 原始文本
            max_tokens: 最大 token 数量

        Returns:
            str: 截断后的文本
        """
        # 简单实现:按字符比例估算
        # 生产环境中应该使用精确的 tokenizer
        char_per_token = len(text) / self._count_tokens(text) if self._count_tokens(text) > 0 else 4
        max_chars = int(max_tokens * char_per_token)

        return text[:max_chars]

    def _count_tokens(self, text: str) -> int:
        """估算文本的 token 数量

        Args:
            text: 文本内容

        Returns:
            int: token 数量
        """
        # 简单估算:中文 1 字符 ≈ 1 token,英文 1 单词 ≈ 1.3 tokens
        # 生产环境中应该使用实际的 tokenizer
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english_words = len([w for w in text.split() if w])

        return int(chinese_chars + english_words * 1.3)

    def _parse_memory_results(self, memory_results: dict, user_query: str) -> List[ContextPacket]:
        """解析记忆结果并转换为 ContextPacket

        Args:
            memory_results: 记忆工具返回的结果
            user_query: 用户查询

        Returns:
            List[ContextPacket]: 转换后的信息包列表
        """
        packets = []
        
        if not memory_results or "memories" not in memory_results:
            return packets
        
        for memory in memory_results.get("memories", []):

            content = memory.content
            timestamp = memory.metadata.get("timestamp", datetime.now())
            
            relevance = self._calculate_relevance(content, user_query)
            
            packets.append(ContextPacket(
                content=content,
                timestamp=timestamp,
                token_count=self._count_tokens(content),
                relevance_score=relevance,
                metadata={"type": "memory", "source": "memory_tool"}
            ))
        
        return packets

    def _parse_rag_results(self, rag_results: dict, user_query: str) -> List[ContextPacket]:
        """解析 RAG 结果并转换为 ContextPacket

        Args:
            rag_results: RAG 工具返回的结果
            user_query: 用户查询

        Returns:
            List[ContextPacket]: 转换后的信息包列表
        """
        packets = []
        
        if not rag_results or "results" not in rag_results:
            return packets
        
        for result in rag_results.get("results", []):
            content = result.get("content", "")
            relevance = self._calculate_recency(content, user_query)
            timestamp = result.get("metadata", {}).get("timestamp", datetime.now())
            
            packets.append(ContextPacket(
                content=content,
                timestamp=timestamp,
                token_count=self._count_tokens(content),
                relevance_score=relevance,  # 确保分数在 0-1 之间
                metadata={"type": "rag_result", "source": "rag_tool"}
            ))
        
        return packets
# 测试内容
if __name__ == "__main__":
    # 创建测试用的 ContextBuilder 实例
    builder = ContextBuilder(user_id="user_xiaoming")
    
    # 测试查询 - 日本旅游相关
    test_query = "请推荐一下京都的赏樱景点和最佳观赏时间"
    
    # 构建对话历史
    test_history = [
        Message(role="user", content="こんにちは"),
        Message(role="assistant", content="こんにちは！日本旅行について何かお手伝いできることはありますか？"),
        Message(role="user", content="京都に行きたいです"),
    ]
    
    # 测试系统指令
    test_system_instructions = "你是一个专业的日本旅游顾问，熟悉日本各地的旅游景点、美食、交通和文化，请提供详细实用的旅行建议。"
    
    # 执行上下文构建
    result = builder.build(
        user_query=test_query,
        conversation_history=test_history,
        system_instructions=test_system_instructions,
        max_tokens=2048
    )
    
    print("=" * 50)
    print("构建的上下文结果：")
    print("=" * 50)
    print(result)
    print("=" * 50)
    print(f"总token数: {builder._count_tokens(result)}")
