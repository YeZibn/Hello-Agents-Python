import re
import time
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple

from synth_agent.memory.memory import MemoryItem, BaseMemory
from synth_agent.config.memory_config import MemoryConfig
from synth_agent.memory.qdrant.qdrant_vector_store import QdrantVectorStore
from synth_agent.memory.neo4j.neo4j_graph_store import Neo4jGraphStore, Entity, Relation
from synth_agent.llm.synth_LLM import SynthLLM
from synth_agent.embedder.qwen_embedder import QwenEmbedder



class SemanticMemory(BaseMemory):
    """语义记忆实现
    
    特点：
    - 使用HuggingFace中文预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 嵌入模型
        self.embedding_model = self._create_embedding_model()
        
        # 数据库连接
        neo4j_config = {
            "uri": config.neo4j_uri,
            "user": config.neo4j_user,
            "password": config.neo4j_password
        }
        
        self.vector_store = QdrantVectorStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_semantic_collection_name
        )
        self.graph_store = Neo4jGraphStore(**neo4j_config)
    
        # 实体和关系缓存
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        
        # NLP处理器
        self.nlp = self._init_nlp()
        
        # 缓存管理
        self._cache = {}
        self._cache_timeout = 3600
    
    def _create_embedding_model(self):
        return QwenEmbedder()
    
    def _init_nlp(self):
        """初始化NLP处理器"""
        try:
            llm = SynthLLM()
            return LLMBasedNLP(llm)
        except Exception as e:
            print(f"LLM初始化失败，使用规则提取: {e}")
            return RuleBasedNLP()
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        try:
            # 1. 生成文本嵌入
            vector = self.embedding_model.encode(memory_item.content)
            
            # 2. 提取实体和关系
            entities = self._extract_entities(memory_item.content)
            relations = self._extract_relations(memory_item.content, entities)
            
            # 3. 存储到Neo4j图数据库
            for entity in entities:
                self._add_entity_to_graph(entity, memory_item)
                self.entities[entity.entity_id] = entity
            
            for relation in relations:
                self._add_relation_to_graph(relation, memory_item)
                relation_key = f"{relation.source_id}_{relation.relation_type}_{relation.target_id}"
                self.relations[relation_key] = relation
            
            # 4. 构建丰富的元数据
            metadata = self._build_rich_metadata(memory_item, entities, relations)
            
            # 5. 存储到Qdrant向量数据库
            self.vector_store.add(
                memory_item.id, 
                vector.tolist() if hasattr(vector, 'tolist') else vector, 
                metadata
            )
            
            # 清除缓存
            self._clear_cache()
            
            print(f"添加语义记忆成功: {memory_item.id}")
            print(f"  - 提取实体: {len(entities)} 个")
            print(f"  - 提取关系: {len(relations)} 个")
            
            return memory_item.id
        except Exception as e:
            print(f"添加语义记忆失败: {e}")
            return memory_item.id
    
    def _build_rich_metadata(self, memory_item: MemoryItem, entities: List[Entity], relations: List[Relation]) -> Dict:
        """构建丰富的元数据"""
        metadata = {
            # 基本信息
            "semantic_id": memory_item.id,
            "content": memory_item.content[:500],  # 存储内容摘要
            "timestamp": memory_item.timestamp.isoformat(),
            "importance": memory_item.importance,
            
            # 实体信息
            "entities": [
                {
                    "id": e.entity_id,
                    "name": e.name,
                    "type": e.type,
                    "properties": e.properties
                } for e in entities
            ],
            "entity_count": len(entities),
            "entity_names": [e.name for e in entities],
            "entity_types": list(set([e.type for e in entities])),
            
            # 关系信息
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type,
                    "properties": r.properties
                } for r in relations
            ],
            "relation_count": len(relations),
            "relation_types": list(set([r.relation_type for r in relations])),
            
            # 用户信息
            "user_id": memory_item.metadata.get("user_id"),
            "user_name": memory_item.metadata.get("user_name"),
            "session_id": memory_item.metadata.get("session_id"),
            
            # 上下文信息
            "tags": memory_item.metadata.get("tags", []),
        }
        
        # 添加自定义元数据
        custom_fields = ["location", "topic", "intent", "keywords", "summary"]
        for field in custom_fields:
            if field in memory_item.metadata:
                metadata[field] = memory_item.metadata[field]
        
        return metadata
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆"""
        try:
            user_id = kwargs.get("user_id")
            
            # 1. 向量检索
            vector_results = self._vector_search(query, limit * 2, user_id)
            
            # 2. 图检索
            graph_results = self._graph_search(query, limit * 2, user_id)
            
            # 3. 混合排序
            combined_results = self._combine_and_rank_results(
                vector_results, graph_results, query, limit
            )
            
            # 4. 转换为MemoryItem
            memory_items = []
            for result in combined_results:
                memory_item = MemoryItem(
                    content=result.get("content", ""),
                    memory_id=result.get("semantic_id"),
                    timestamp=datetime.fromisoformat(result.get("timestamp")) if result.get("timestamp") else datetime.now(),
                    importance=result.get("importance", 0.5),
                    metadata=result
                )
                memory_items.append(memory_item)
            
            return memory_items[:limit]
        except Exception as e:
            print(f"检索语义记忆失败: {e}")
            return []
    
    def retrieve_by_entity(self, entity_name: str, limit: int = 10, **kwargs) -> List[MemoryItem]:
        """按实体检索相关记忆"""
        try:
            user_id = kwargs.get("user_id")
            
            # 搜索实体
            entities = self.graph_store.search_entities(entity_name, limit=1, user_id=user_id)
            if not entities:
                return []
            
            entity_id = entities[0]["entity_id"]
            
            # 获取实体的所有关系
            relations = self.graph_store.get_entity_relations(entity_id, limit * 2, user_id=user_id)
            
            # 收集相关的记忆ID
            memory_ids = set()
            for rel in relations:
                # 从关系中获取记忆ID（需要根据实际存储结构调整）
                pass
            
            # TODO: 根据记忆ID从向量库获取详细内容
            return []
        except Exception as e:
            print(f"按实体检索失败: {e}")
            return []
    
    def retrieve_by_relation(self, relation_type: str, limit: int = 10, **kwargs) -> List[MemoryItem]:
        """按关系类型检索相关记忆"""
        try:
            user_id = kwargs.get("user_id")
            relations = self.graph_store.search_relations(relation_type, limit, user_id=user_id)
            # TODO: 根据关系获取相关记忆
            return []
        except Exception as e:
            print(f"按关系检索失败: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """提取实体"""
        entities = []
        extracted = self.nlp.extract_entities(text)
        
        for name, entity_type in extracted:
            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            entity = Entity(
                entity_id=entity_id,
                name=name,
                type=entity_type,
                properties={"source_text": text[:100]}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """提取关系"""
        relations = []
        entity_names = {entity.name: entity.entity_id for entity in entities}
        
        # 提取关系
        extracted_relations = self.nlp.extract_relations(text, [(e.name, e.type) for e in entities])
        
        for source_name, relation_type, target_name in extracted_relations:
            if source_name in entity_names and target_name in entity_names:
                relation = Relation(
                    source_id=entity_names[source_name],
                    target_id=entity_names[target_name],
                    relation_type=relation_type,
                    properties={"source_text": text[:100]}
                )
                relations.append(relation)
        
        return relations
    
    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem):
        """添加实体到图数据库"""
        try:
            user_id = memory_item.metadata.get("user_id")
            self.graph_store.add_entity(entity, memory_item.id, user_id=user_id, content=memory_item.content)
        except Exception as e:
            print(f"添加实体到图数据库失败: {e}")
    
    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem):
        """添加关系到图数据库"""
        try:
            user_id = memory_item.metadata.get("user_id")
            self.graph_store.add_relation(relation, memory_item.id, user_id=user_id)
        except Exception as e:
            print(f"添加关系到图数据库失败: {e}")
    
    def _vector_search(self, query: str, limit: int, user_id: Optional[str]) -> List[Dict]:
        """向量搜索"""
        try:
            query_vector = self.embedding_model.encode(query)
            query_vector = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
            hits = self.vector_store.search(query_vector, limit)
            
            # 过滤用户ID
            if user_id:
                hits = [hit for hit in hits if hit.get("metadata", {}).get("user_id") == user_id]
            
            return hits
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return []
    
    def _graph_search(self, query: str, limit: int, user_id: Optional[str]) -> List[Dict]:
        """图搜索"""
        try:
            entity_results = self.graph_store.search_entities(query, limit, user_id=user_id)
            relation_results = self.graph_store.search_relations(query, limit, user_id=user_id)
            
            # 转换为统一格式
            results = []
            for entity in entity_results:
                results.append({
                    "semantic_id": f"entity_{entity['entity_id']}",
                    "similarity": 0.8,
                    "entity": entity
                })
            
            for relation in relation_results:
                results.append({
                    "semantic_id": f"relation_{relation['source_id']}_{relation['target_id']}",
                    "similarity": 0.7,
                    "relation": relation
                })
            
            return results[:limit]
        except Exception as e:
            print(f"图搜索失败: {e}")
            return []
    
    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        """混合排序结果"""
        combined = {}
        
        # 合并向量和图检索结果
        for result in vector_results:
            semantic_id = result.get("metadata", {}).get("semantic_id") or result.get("id")
            combined[semantic_id] = {
                **result.get("metadata", {}),
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0
            }
        
        for result in graph_results:
            semantic_id = result["semantic_id"]
            if semantic_id in combined:
                combined[semantic_id]["graph_score"] = result.get("similarity", 0.0)
            else:
                combined[semantic_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0)
                }
        
        # 计算混合分数
        for semantic_id, result in combined.items():
            vector_score = result.get("vector_score", 0.0)
            graph_score = result.get("graph_score", 0.0)
            importance = result.get("importance", 0.5)
            
            # 基础相似度得分
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            
            # 重要性权重 [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)
            
            # 最终得分：相似度 * 重要性权重
            combined_score = base_relevance * importance_weight
            result["combined_score"] = combined_score
        
        # 排序并返回
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def _clear_cache(self):
        """清除缓存"""
        self._cache.clear()
    
    def get_entity_stats(self, **kwargs) -> Dict:
        """获取实体统计信息"""
        user_id = kwargs.get("user_id")
        
        # 从图数据库获取统计信息（已包含用户隔离）
        graph_stats = self.get_graph_stats(user_id=user_id)
        
        # 始终使用图数据库的统计数据，避免缓存数据不准确
        entity_count = graph_stats.get("entity_count", 0)
        relation_count = graph_stats.get("relation_count", 0)
        
        stats = {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "entity_types": graph_stats.get("entity_types", {}),
            "relation_types": graph_stats.get("relation_types", {}),
            "total_memories": graph_stats.get("memory_count", 0)
        }
        
        return stats
    
    def get_graph_stats(self, user_id: Optional[str] = None) -> Dict:
        """获取图数据库统计信息"""
        try:
            return self.graph_store.get_stats(user_id=user_id)
        except Exception as e:
            print(f"获取图统计失败: {e}")
            return {"entity_count": 0, "relation_count": 0, "memory_count": 0, "entity_types": {}, "relation_types": {}}
    
    def close(self):
        """关闭资源"""
        if hasattr(self.graph_store, 'close'):
            self.graph_store.close()


class RuleBasedNLP:
    """基于规则的NLP处理器"""
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """使用正则表达式提取实体"""
        entities = []
        
        # 提取中文人名
        person_pattern = re.compile(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|博士|经理|总监)?')
        persons = person_pattern.findall(text)
        
        # 提取组织机构
        org_pattern = re.compile(r'[\u4e00-\u9fff]{2,}(?:公司|机构|大学|学院|研究所|部门|集团|银行|医院)')
        orgs = org_pattern.findall(text)
        
        # 提取地点
        location_pattern = re.compile(r'[\u4e00-\u9fff]{2,}(?:市|省|县|区|镇|村|路|街|楼)')
        locations = location_pattern.findall(text)
        
        # 提取时间
        time_pattern = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|昨天|今天|明天|上周|下周')
        times = time_pattern.findall(text)
        
        for person in persons:
            entities.append((person, "PERSON"))
        for org in orgs:
            entities.append((org, "ORG"))
        for location in locations:
            entities.append((location, "LOCATION"))
        for t in times:
            entities.append((t, "TIME"))
        
        return list(set(entities))
    
    def extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """使用规则提取关系"""
        relations = []
        entity_names = [e[0] for e in entities]
        
        # 定义关系模式
        relation_patterns = [
            (r'(.+?)是(.+?)的(.+)', '是'),
            (r'(.+?)在(.+?)工作', '工作于'),
            (r'(.+?)毕业于(.+)', '毕业于'),
            (r'(.+?)喜欢(.+)', '喜欢'),
            (r'(.+?)认识(.+)', '认识'),
            (r'(.+?)属于(.+)', '属于'),
        ]
        
        sentences = re.split(r'[。！？；\n]', text)
        for sentence in sentences:
            for entity1 in entity_names:
                if entity1 in sentence:
                    for entity2 in entity_names:
                        if entity1 != entity2 and entity2 in sentence:
                            # 检查关系模式
                            for pattern, rel_type in relation_patterns:
                                if re.search(pattern, sentence):
                                    relations.append((entity1, rel_type, entity2))
                                    break
        
        return list(set(relations))


class LLMBasedNLP:
    """基于LLM的NLP处理器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM并返回文本内容"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.think(messages)
        if response and isinstance(response, dict):
            return response.get("full_content", "")
        return ""
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """使用LLM提取实体"""
        prompt = f"""请从以下文本中提取命名实体，并以JSON格式返回。

实体类型包括：
- PERSON（人名）
- ORG（组织机构）
- LOCATION（地点）
- TIME（时间）
- PRODUCT（产品）
- EVENT（事件）

文本：{text}

请返回以下格式的JSON：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型"}},
        ...
    ]
}}
只返回JSON，不要其他解释。"""

        try:
            response = self._call_llm(prompt)
            if not response:
                return RuleBasedNLP().extract_entities(text)
            
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group()
            
            result = json.loads(response)
            entities = []
            for entity in result.get("entities", []):
                name = entity.get("name", "")
                entity_type = entity.get("type", "UNKNOWN")
                if name:
                    entities.append((name, entity_type))
            return entities
        except Exception as e:
            print(f"LLM实体提取失败: {e}")
            return RuleBasedNLP().extract_entities(text)
    
    def extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """使用LLM提取关系"""
        entity_names = [e[0] for e in entities]
        entities_str = ", ".join([f"{name}({type})" for name, type in entities])
        
        prompt = f"""请从以下文本中提取实体之间的关系，并以JSON格式返回。

文本：{text}

已识别的实体：{entities_str}

请返回以下格式的JSON：
{{
    "relations": [
        {{"source": "源实体名称", "target": "目标实体名称", "relation": "关系类型"}},
        ...
    ]
}}

关系类型可以是：是、在、喜欢、工作于、属于、拥有、认识、毕业于等。
只返回JSON，不要其他解释。"""

        try:
            response = self._call_llm(prompt)
            if not response:
                return RuleBasedNLP().extract_relations(text, entities)
            
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group()
            
            result = json.loads(response)
            relations = []
            for relation in result.get("relations", []):
                source = relation.get("source", "")
                target = relation.get("target", "")
                rel_type = relation.get("relation", "相关")
                if source and target and source in entity_names and target in entity_names:
                    relations.append((source, rel_type, target))
            return relations
        except Exception as e:
            print(f"LLM关系提取失败: {e}")
            return RuleBasedNLP().extract_relations(text, entities)


if __name__ == "__main__":
    from synth_agent.config.memory_config import MemoryConfig
    
    # 创建配置
    config = MemoryConfig(
        qdrant_url="http://localhost:6333",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="12345678"
    )
    
    # 初始化语义记忆
    semantic_memory = SemanticMemory(config)
    
    # # 测试添加记忆
    print("=== 测试添加语义记忆 ===")
    
    test_memories = [
        {
            "content": "张三是清华大学计算机系的教授，他专注于人工智能和机器学习研究。",
            "importance": 0.9,
            "metadata": {
                "session_id": "session_001",
                "user_id": "user_xiaoming",
                "user_name": "小明",
                "category": "人物介绍",
                "tags": ["教授", "AI", "清华"]
            }
        },
        {
            "content": "李四在北京大学读书，他喜欢打篮球和游泳。",
            "importance": 0.7,
            "metadata": {
                "session_id": "session_001",
                "user_id": "user_xiaoming",
                "user_name": "小明",
                "category": "人物介绍",
                "tags": ["学生", "运动"]
            }
        },
        {
            "content": "阿里巴巴集团在杭州成立了人工智能研究院。",
            "importance": 0.8,
            "metadata": {
                "session_id": "session_002",
                "user_id": "user_xiaoming",
                "user_name": "小明",
                "category": "新闻",
                "tags": ["公司", "AI", "杭州"]
            }
        }
    ]
    
    for i, mem_data in enumerate(test_memories):
        memory_item = MemoryItem(
            content=mem_data["content"],
            importance=mem_data["importance"],
            metadata=mem_data["metadata"]
        )
        semantic_memory.add(memory_item)
        print()
    
    # 测试检索
    print("\n=== 测试语义检索 ===")
    results = semantic_memory.retrieve("人工智能研究", limit=3, user_id="user_xiaoming")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.content[:50]}... (得分: {result.metadata.get('combined_score', 0):.3f})")
    
    # 测试统计信息
    print("\n=== 统计信息 ===")
    entity_stats = semantic_memory.get_entity_stats()
    print(f"实体数量: {entity_stats['entity_count']}")
    print(f"关系数量: {entity_stats['relation_count']}")
    print(f"实体类型分布: {entity_stats['entity_types']}")
    print(f"关系类型分布: {entity_stats['relation_types']}")
    
    # 关闭资源
    semantic_memory.close()
    print("\n测试完成!")



