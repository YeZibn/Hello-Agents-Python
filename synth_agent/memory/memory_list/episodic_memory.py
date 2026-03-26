import time
from datetime import datetime
from typing import List, Dict, Optional, Set
from synth_agent.memory.sqlite.sqlite_document_store import SQLiteDocumentStore
from synth_agent.memory.qdrant.qdrant_vector_store import QdrantVectorStore
from synth_agent.memory.memory import MemoryItem, BaseMemory
from synth_agent.config.memory_config import MemoryConfig
from synth_agent.memory.memory_list.episodic import Episode
from synth_agent.embedder.qwen_embedder import QwenEmbedder


class EpisodicMemory(BaseMemory):
    """情景记忆实现
    特点：
    - SQLite+Qdrant混合存储架构
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    - 优化的评分算法
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key, config.qdrant_episodic_collection_name)
        self.embedder = self._create_embedding_model()
        self.sessions = {}
        self._session_cache = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600
        self._load_sessions()
    
    def _load_sessions(self):
        """从数据库加载会话索引"""
        try:
            all_ids = self.doc_store.get_all_ids()
            
            for memory_id in all_ids:
                episode = self.doc_store.get(memory_id)
                if episode:
                    session_id = episode.session_id
                    if session_id not in self.sessions:
                        self.sessions[session_id] = []
                    self.sessions[session_id].append(memory_id)
            
            for session_id in self.sessions:
                episodes = []
                for memory_id in self.sessions[session_id]:
                    episode = self.doc_store.get(memory_id)
                    if episode:
                        episodes.append((episode.timestamp, memory_id))
                
                episodes.sort(key=lambda x: x[0])
                self.sessions[session_id] = [mem_id for _, mem_id in episodes]
                
        except Exception as e:
            print(f"加载会话索引失败: {e}")
            self.sessions = {}
    
    def _create_embedding_model(self):
        return QwenEmbedder()
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        self._cleanup_sessions()
        
        session_id = memory_item.metadata.get("session_id", "default")
        episode = Episode(
            memory_id=memory_item.id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=memory_item.metadata,
        )
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        max_session_size = 1000
        if len(self.sessions[session_id]) >= max_session_size:
            oldest_memory_id = self.sessions[session_id].pop(0)
            self._remove_memory(oldest_memory_id)
        
        self.sessions[session_id].append(episode.memory_id)
        
        self._persist_episode(episode)
        
        if session_id in self._session_cache:
            del self._session_cache[session_id]
        
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：结构化过滤 + 语义向量检索"""
        candidate_ids = self._structured_filter(**kwargs)
        
        hits = self._vector_search(query, limit * 5, kwargs.get("user_id"))
        
        results = []
        for hit in hits:
            if self._should_include(hit, candidate_ids, kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item(hit)
                results.append((score, memory_item))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]
    
    def retrieve_by_session(self, session_id: str, limit: int = 10, **kwargs) -> List[MemoryItem]:
        """按会话检索记忆"""
        user_id = kwargs.get("user_id")
        
        if session_id in self._session_cache:
            cached_results = self._session_cache[session_id][:limit]
            # 过滤用户ID
            if user_id:
                cached_results = [m for m in cached_results if m.metadata.get("user_id") == user_id]
            return cached_results
        
        episodes = self.doc_store.get_by_session(session_id, limit)
        
        results = []
        for episode in episodes:
            # 过滤用户ID
            if user_id and episode.context.get("user_id") != user_id:
                continue
                
            memory_item = MemoryItem(
                content=episode.content,
                memory_id=episode.memory_id,
                timestamp=episode.timestamp,
                metadata=episode.context
            )
            results.append(memory_item)
        
        self._session_cache[session_id] = results
        
        return results
    
    def _persist_episode(self, episode: Episode) -> None:
        """持久化存储情景记忆"""
        self.doc_store.save(episode)
        
        try:
            vector = self.embedder.encode(episode.content)
            metadata = {
                "memory_id": episode.memory_id,
                "session_id": episode.session_id,
                "timestamp": episode.timestamp.isoformat(),
                "importance": episode.context.get("importance", 0.5),
                "content": episode.content[:200],
                "tags": episode.context.get("tags", []),
                "user_id": episode.context.get("user_id", None),
                "user_name": episode.context.get("user_name", None),
            }
            self.vector_store.add(episode.memory_id, vector.tolist() if hasattr(vector, 'tolist') else vector, metadata)
        except Exception as e:
            print(f"嵌入失败: {e}")
    
    def _remove_memory(self, memory_id: str) -> None:
        """移除情景记忆"""
        self.doc_store.delete(memory_id)
        
        try:
            self.vector_store.delete(memory_id)
        except Exception as e:
            print(f"删除向量失败: {e}")
    
    def _get_episode(self, memory_id: str) -> Optional[Episode]:
        """获取情景记忆"""
        return self.doc_store.get(memory_id)
    
    def _structured_filter(self, **kwargs) -> Set[str]:
        """结构化过滤"""
        return self.doc_store.get_all_ids()
    
    def _vector_search(self, query: str, limit: int, user_id: Optional[str]) -> List[Dict]:
        """向量搜索"""
        try:
            query_vector = self.embedder.encode(query)
            query_vector = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
            
            hits = self.vector_store.search(query_vector, limit)
            
            if user_id:
                hits = [hit for hit in hits if hit.get("metadata", {}).get("user_id") == user_id]
            
            return hits
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return []
    
    def _should_include(self, hit: Dict, candidate_ids: Set[str], kwargs: Dict) -> bool:
        """判断是否包含该结果"""
        memory_id = hit.get("id")
        if memory_id not in candidate_ids:
            return False
        
        if "start_time" in kwargs or "end_time" in kwargs:
            timestamp_str = hit["metadata"].get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if "start_time" in kwargs and timestamp < kwargs["start_time"]:
                        return False
                    if "end_time" in kwargs and timestamp > kwargs["end_time"]:
                        return False
                except Exception:
                    pass
        
        if "min_importance" in kwargs:
            importance = hit["metadata"].get("importance", 0.5)
            if importance < kwargs["min_importance"]:
                return False
        
        return True
    
    def _calculate_episode_score(self, hit: Dict) -> float:
        """情景记忆评分算法"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"].get("timestamp"))
        importance = hit["metadata"].get("importance", 0.5)
        
        base_relevance = vec_score * 0.7 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)
        
        return base_relevance * importance_weight
    
    def _calculate_recency(self, timestamp_str: Optional[str]) -> float:
        """计算时间近因性得分"""
        if not timestamp_str:
            return 0.5
        
        try:
            import math
            timestamp = datetime.fromisoformat(timestamp_str)
            current_time = datetime.now()
            age_hours = (current_time - timestamp).total_seconds() / 3600
            
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            
            return max(0.1, recency_score)
        except Exception:
            return 0.5
    
    def _create_memory_item(self, hit: Dict) -> MemoryItem:
        """从搜索结果创建MemoryItem"""
        metadata = hit["metadata"]
        return MemoryItem(
            content=metadata.get("content", ""),
            memory_id=hit.get("id"),
            timestamp=datetime.fromisoformat(metadata.get("timestamp")) if metadata.get("timestamp") else datetime.now(),
            metadata=metadata
        )
    
    def _cleanup_sessions(self) -> None:
        """清理过期会话和缓存"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        empty_sessions = [session_id for session_id, memory_ids in self.sessions.items() if not memory_ids]
        for session_id in empty_sessions:
            del self.sessions[session_id]
            if session_id in self._session_cache:
                del self._session_cache[session_id]
        
        expired_sessions = []
        for session_id, cached_items in self._session_cache.items():
            if not cached_items:
                expired_sessions.append(session_id)
        for session_id in expired_sessions:
            del self._session_cache[session_id]
        
        self._last_cleanup = current_time
    
    def get_session_stats(self, **kwargs) -> Dict:
        """获取会话统计信息"""
        user_id = kwargs.get("user_id")
        
        stats = {
            "total_sessions": len(self.sessions),
            "session_distribution": {},
            "total_memories": 0
        }
        
        for session_id, memory_ids in self.sessions.items():
            # 过滤用户记忆
            if user_id:
                user_memory_ids = []
                for memory_id in memory_ids:
                    episode = self.doc_store.get(memory_id)
                    if episode and episode.context.get("user_id") == user_id:
                        user_memory_ids.append(memory_id)
                count = len(user_memory_ids)
            else:
                count = len(memory_ids)
            
            stats["total_memories"] += count
            if count > 0:
                stats["session_distribution"][session_id] = count
        
        return stats


if __name__ == "__main__":
    # 模拟一个生活化的场景：用户的旅行规划助手记忆系统
    
    # 创建测试配置
    config = MemoryConfig(
        database_path="travel_memory.db",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None
    )
    
    # 初始化情景记忆
    memory = EpisodicMemory(config)
    
    # print("=== 🏠 情景记忆系统 - 智能旅行助手 ===\n")
    
    # # 场景1：用户第一次咨询 - 日本旅行规划
    # print("📅 场景1：用户小明咨询日本樱花季旅行")
    # session_tokyo = "session_tokyo_trip_20240315"
    
    # tokyo_memories = [
    #     {
    #         "content": "用户小明说：'我想今年4月初去东京看樱花，大概5-7天行程，预算1万5左右'",
    #         "importance": 0.9,
    #         "tags": ["需求", "预算", "时间"]
    #     },
    #     {
    #         "content": "助手推荐：新宿御苑和上野公园是东京最佳赏樱地点，建议提前预订酒店",
    #         "importance": 0.8,
    #         "tags": ["推荐", "景点"]
    #     },
    #     {
    #         "content": "用户询问：'我需要办理签证吗？' 助手回复：需要护照和在职证明，办理周期7-10个工作日",
    #         "importance": 0.85,
    #         "tags": ["签证", "手续"]
    #     },
    #     {
    #         "content": "用户提到对日式温泉很感兴趣，希望安排箱根一日游",
    #         "importance": 0.75,
    #         "tags": ["兴趣", "温泉"]
    #     }
    # ]
    
    # for i, mem in enumerate(tokyo_memories):
    #     item = MemoryItem(
    #         content=mem["content"],
    #         memory_id=f"tokyo_{i}_{int(time.time())}",
    #         timestamp=datetime(2024, 3, 15, 10, 0 + i*10),
    #         metadata={
    #             "session_id": session_tokyo,
    #             "user_id": "user_xiaoming",
    #             "user_name": "小明",
    #             "importance": mem["importance"],
    #             "tags": mem["tags"],
    #             "topic": "日本旅行"
    #         }
    #     )
    #     memory.add(item)
    #     print(f"  ✓ 记录: {mem['content'][:40]}...")
    
    # # 场景2：一周后用户再次咨询 - 行程细节确认
    # print("\n📅 场景2：一周后小明确认行程细节")
    # session_tokyo_followup = "session_tokyo_confirm_20240322"
    
    # followup_memories = [
    #     {
    #         "content": "用户说：'签证已经办好了，现在想确认具体的行程路线'",
    #         "importance": 0.9,
    #         "tags": ["进度更新", "行程"]
    #     },
    #     {
    #         "content": "用户特别提到：'我女朋友对购物很感兴趣，想多安排些时间在新宿和银座'",
    #         "importance": 0.8,
    #         "tags": ["同伴", "购物", "偏好"]
    #     },
    #     {
    #         "content": "最终确定行程：Day1抵达东京-Day2上野赏樱-Day3箱根温泉-Day4富士山-Day5购物-Day6返程",
    #         "importance": 0.95,
    #         "tags": ["最终方案", "行程"]
    #     }
    # ]
    
    # for i, mem in enumerate(followup_memories):
    #     item = MemoryItem(
    #         content=mem["content"],
    #         memory_id=f"tokyo_follow_{i}_{int(time.time())}",
    #         timestamp=datetime(2024, 3, 22, 14, 0 + i*15),
    #         metadata={
    #             "session_id": session_tokyo_followup,
    #             "user_id": "user_xiaoming",
    #             "user_name": "小明",
    #             "importance": mem["importance"],
    #             "tags": mem["tags"],
    #             "topic": "日本旅行确认",
    #             "related_session": session_tokyo
    #         }
    #     )
    #     memory.add(item)
    #     print(f"  ✓ 记录: {mem['content'][:40]}...")
    
    # # 场景3：另一个用户咨询 - 完全不同的话题
    # print("\n📅 场景3：新用户小红咨询国内亲子游")
    # session_family = "session_family_trip_20240320"
    
    # family_memories = [
    #     {
    #         "content": "用户小红说：'想带6岁的孩子暑假去海边，有什么推荐？'",
    #         "importance": 0.85,
    #         "tags": ["亲子", "海边", "暑假"]
    #     },
    #     {
    #         "content": "助手推荐三亚或青岛，三亚更适合小朋友玩水，青岛可以体验赶海",
    #         "importance": 0.8,
    #         "tags": ["推荐", "目的地"]
    #     },
    #     {
    #         "content": "用户担心：'孩子比较小，坐飞机会不会不适应？'",
    #         "importance": 0.75,
    #         "tags": ["担忧", "交通"]
    #     }
    # ]
    
    # for i, mem in enumerate(family_memories):
    #     item = MemoryItem(
    #         content=mem["content"],
    #         memory_id=f"family_{i}_{int(time.time())}",
    #         timestamp=datetime(2024, 3, 20, 9, 0 + i*20),
    #         metadata={
    #             "session_id": session_family,
    #             "user_id": "user_xiaohong",
    #             "user_name": "小红",
    #             "importance": mem["importance"],
    #             "tags": mem["tags"],
    #             "topic": "亲子游"
    #         }
    #     )
    #     memory.add(item)
    #     print(f"  ✓ 记录: {mem['content'][:40]}...")
    
    # 测试语义检索 - 模拟用户提出新问题
    print("\n🔍 === 测试语义检索 ===")
    print("用户问：'东京樱花赏樱地点推荐注意事项'\n")
    
    results = memory.retrieve("'请推荐一下京都的赏樱景点和最佳观赏时间'", limit=3, user_id="user_xiaoming")
    for i, result in enumerate(results, 1):
        user_name = result.metadata.get("user_name", "未知用户")
        print(f"  相关记忆{i} [{user_name}]: {result.content}")
        print(f"    → 重要性: {result.metadata.get('importance', 'N/A')}, 标签: {result.metadata.get('tags', [])}")
        print()
    
    # 测试按会话检索 - 查看某个用户的完整对话历史
    print("📋 === 测试按会话检索（小明的日本旅行规划）===")
    session_results = memory.retrieve_by_session(session_tokyo, limit=10)
    
    print(f"会话 {session_tokyo} 的历史记录：")
    for i, result in enumerate(session_results, 1):
        time_str = result.timestamp.strftime("%m月%d日 %H:%M") if result.timestamp else "未知时间"
        print(f"  {i}. [{time_str}] {result.content[:60]}...")
    
    # 测试跨会话检索 - 查找特定用户的所有相关记忆
    print("\n👤 === 测试用户偏好检索 ===")
    print("查找小明提到的所有兴趣和偏好：\n")
    
    preference_results = memory.retrieve("小明 女朋友 购物 温泉 兴趣偏好", limit=5, user_id="user_xiaoming")
    for i, result in enumerate(preference_results, 1):
        if result.metadata.get("user_id") == "user_xiaoming":
            print(f"  偏好{i}: {result.content}")
    
    # 测试获取会话统计
    print("\n📊 === 会话统计 ===")
    stats = memory.get_session_stats()
    print(f"总会话数: {stats['total_sessions']}")
    print("会话分布:")
    for session_id, count in stats['session_distribution'].items():
        # 简化会话ID显示
        short_id = session_id.replace("session_", "")
        print(f"  - {short_id}: {count} 条记忆")
    
    print("\n✅ 生活化场景测试完成！")
    print("这个测试模拟了真实的旅行助手场景，包括：")
    print("  • 多轮对话的上下文记忆")
    print("  • 不同用户的隔离存储")
    print("  • 语义检索理解用户意图")
    print("  • 按会话追溯完整对话历史")