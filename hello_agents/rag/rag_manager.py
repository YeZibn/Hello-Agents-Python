"""RAG管理器 - 提供完整的RAG管理能力"""
import os
from typing import Dict, Any, List, Optional
from hello_agents.config.rag_config import RAGConfig
from hello_agents.memory.qdrant.qdrant_vector_store import QdrantVectorStore
from qdrant_client.models import PointStruct
from markitdown import MarkItDown

from hello_agents.embedder.qwen_embedder import QwenEmbedder



class RAGManager:
    """RAG管理器
    
    提供完整的 RAG 管理能力：
    - 添加多格式文档（PDF、Office等）
    - 智能检索与召回
    - 知识库管理
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        config: Optional[RAGConfig] = None
    ):
        self.user_id = user_id
        self.config = config or RAGConfig()

        os.makedirs(config.knowledge_base_path, exist_ok=True)
        
        # 创建用户特定的目录结构
        self.user_knowledge_base_path = os.path.join(
            self.config.knowledge_base_path,
            f"user_{user_id}"
        )
        os.makedirs(self.user_knowledge_base_path, exist_ok=True)
        os.makedirs(os.path.join(self.user_knowledge_base_path, "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.user_knowledge_base_path, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(self.user_knowledge_base_path, "markdown"), exist_ok=True)
        
        # 使用用户ID作为命名空间的一部分
        self.user_namespace = f"{self.config.rag_namespace}_{user_id}"
        
        self.embedder = self._create_embedding_model()
        self.qdrant_client = QdrantVectorStore(
            config.qdrant_url, 
            config.qdrant_api_key, 
            config.collection_name
        )
        self.markitdown = self._init_markitdown()
        
    
    def _create_embedding_model(self):
        """创建嵌入模型"""
        return QwenEmbedder()
    
    def _init_markitdown(self):
        """初始化 MarkItDown 转换器"""
        try:
            return MarkItDown()
        except Exception as e:
            print(f"初始化 MarkItDown 失败: {e}")
            return None
    
    def index_document(self, file_path: str) -> str:
        """索引文档"""
        if not file_path:
            return "❌ 缺少必填参数: file_path"
        
        if not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"
        
        try:
            # 复制原始文档到用户目录
            import shutil
            doc_filename = os.path.basename(file_path)
            dest_doc_path = os.path.join(self.user_knowledge_base_path, "documents", doc_filename)
            shutil.copy2(file_path, dest_doc_path)
            print(f"✅ 文档已保存到: {dest_doc_path}")
            
            markdown_text = self._convert_to_markdown(file_path)
            if not markdown_text.strip():
                return f"❌ 文档转换失败或文档为空: {file_path}"
            
            chunks = self._chunk_text(markdown_text)
            
            # 保存分块到用户目录
            self._save_chunks(chunks, doc_filename)
            
            if not self.embedder:
                return "❌ 嵌入模型未初始化，无法索引文档"
            
            if not self.qdrant_client:
                return "❌ Qdrant客户端未初始化，无法索引文档"
            
            self._index_chunks(chunks, dest_doc_path)
            
            return f"✅ 文档索引成功: {file_path}，共生成 {len(chunks)} 个分块"
        except Exception as e:
            return f"❌ 索引文档失败: {str(e)}"
    
    def add_text(self, text: str, source: str = "手动输入") -> str:
        """添加文本"""
        if not text:
            return "❌ 缺少必填参数: text"
        
        try:
            chunks = self._chunk_text(text)
            
            # 保存文本内容和分块到用户目录
            import time
            timestamp = int(time.time())
            text_filename = f"text_{source}_{timestamp}.txt"
            text_path = os.path.join(self.user_knowledge_base_path, "documents", text_filename)
            
            # 保存原始文本
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ 文本已保存到: {text_path}")
            
            # 保存分块
            self._save_chunks(chunks, text_filename)
            
            if not self.embedder:
                return "❌ 嵌入模型未初始化，无法添加文本"
            
            if not self.qdrant_client:
                return "❌ Qdrant客户端未初始化，无法添加文本"
            
            self._index_chunks(chunks, text_path)
            
            return f"✅ 文本添加成功，共生成 {len(chunks)} 个分块"
        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"
    
    def list_documents(self) -> str:
        """列出文档"""
        try:
            if not self.qdrant_client:
                return "❌ Qdrant客户端未初始化"
            
            result = self.qdrant_client.scroll(
                limit=100,
                with_payload=True,
                scroll_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": self.user_id}}
                    ]
                }
            )
            
            points = result[0]
            
            sources = set()
            for point in points:
                if point.payload and "source" in point.payload:
                    sources.add(point.payload["source"])
            
            if not sources:
                return "📭 知识库为空"
            
            return f"📚 知识库包含 {len(sources)} 个文档:\n" + "\n".join(f"- {s}" for s in sorted(sources))
        except Exception as e:
            return f"❌ 列出文档失败: {str(e)}"
    
    def _convert_to_markdown(self, path: str) -> str:
        """将文档转换为Markdown格式并保存"""
        if not os.path.exists(path):
            return ""
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.txt', '.md', '.json', '.csv', '.xml', '.html']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    markdown_text = f.read()
            except Exception as e:
                print(f"⚠️ 读取文本文件失败: {e}")
                return ""
        
        elif self.markitdown:
            try:
                result = self.markitdown.convert(path)
                markdown_text = getattr(result, "text_content", None)
                if not isinstance(markdown_text, str) or not markdown_text.strip():
                    print(f"⚠️ MarkItDown转换失败: 无有效内容")
                    return ""
                print(f"✅ MarkItDown转换成功: {path}")
            except Exception as e:
                print(f"⚠️ MarkItDown转换失败: {e}")
                return ""
        else:
            return ""
        
        # 保存Markdown文件到markdown目录
        try:
            doc_filename = os.path.basename(path)
            md_filename = os.path.splitext(doc_filename)[0] + '.md'
            md_save_path = os.path.join(self.user_knowledge_base_path, "markdown", md_filename)
            
            with open(md_save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            print(f"✅ Markdown已保存到: {md_save_path}")
        except Exception as e:
            print(f"⚠️ 保存Markdown文件失败: {e}")
        
        return markdown_text
    
    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 20) -> List[Dict]:
        """将文本递归分块
        
        递归分块策略:
        1. 首先按标题和段落分割成语义单元
        2. 如果单元大小合适则保留
        3. 如果太大则按句子分割
        4. 如果句子仍太大则按单词递归分割
        """
        paragraphs = self._split_paragraphs(text)
        return self._recursive_chunk(paragraphs, chunk_size, overlap)
    
    def _recursive_chunk(self, elements: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
        """递归分块主逻辑
        
        Args:
            elements: 要分块的元素列表，每个元素包含content和heading_path
            chunk_size: 最大块大小（字符数）
            overlap: 块之间的重叠大小
        
        Returns:
            分块后的列表
        """
        chunks = []
        
        if not elements:
            return chunks
        
        current_group = []
        current_size = 0
        
        for element in elements:
            element_size = len(element["content"])
            
            if current_size + element_size > chunk_size and current_group:
                chunk_text = "\n\n".join(p["content"] for p in current_group)
                chunks.append({
                    "content": chunk_text,
                    "heading_path": current_group[0].get("heading_path"),
                    "start": current_group[0]["start"],
                    "end": current_group[-1]["end"]
                })
                
                if overlap > 0:
                    overlap_text = self._get_overlap_content(chunk_text, overlap)
                    if overlap_text:
                        overlap_element = {
                            "content": overlap_text,
                            "heading_path": current_group[-1].get("heading_path"),
                            "start": current_group[-1]["end"] - len(overlap_text),
                            "end": current_group[-1]["end"],
                            "is_overlap": True
                        }
                        current_group = [overlap_element]
                        current_size = len(overlap_text)
                    else:
                        current_group = []
                        current_size = 0
                else:
                    current_group = []
                    current_size = 0
            
            if element_size > chunk_size:
                if current_group:
                    chunk_text = "\n\n".join(p["content"] for p in current_group)
                    chunks.append({
                        "content": chunk_text,
                        "heading_path": current_group[0].get("heading_path"),
                        "start": current_group[0]["start"],
                        "end": current_group[-1]["end"]
                    })
                    current_group = []
                    current_size = 0
                
                sub_chunks = self._split_element(element, chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                current_group.append(element)
                current_size += element_size
        
        if current_group:
            chunk_text = "\n\n".join(p["content"] for p in current_group)
            chunks.append({
                "content": chunk_text,
                "heading_path": current_group[0].get("heading_path"),
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"]
            })
        
        return chunks
    
    def _get_overlap_content(self, text: str, overlap_size: int) -> str:
        """获取文本末尾的overlap内容
        
        优先在句子边界分割，避免截断句子。
        
        Args:
            text: 原始文本
            overlap_size: overlap大小（字符数）
        
        Returns:
            overlap文本
        """
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        
        sentence_endings = ['。', '！', '？', '.', '!', '?', '\n']
        
        for i, char in enumerate(overlap_text):
            if char in sentence_endings:
                return overlap_text[i+1:].strip()
        
        return overlap_text
    
    def _split_element(self, element: Dict, chunk_size: int, overlap: int) -> List[Dict]:
        """拆分过大的元素
        
        按句子级别进行拆分，保留标题上下文。
        
        Args:
            element: 要拆分的元素
            chunk_size: 最大块大小
            overlap: 重叠大小
        
        Returns:
            拆分后的块列表
        """
        content = element["content"]
        heading_path = element.get("heading_path")
        start_pos = element.get("start", 0)
        
        sentences = self._split_sentences(content)
        
        if len(sentences) <= 1:
            return self._split_by_words(element, chunk_size)
        
        sentence_elements = []
        for sentence in sentences:
            sentence_start = content.find(sentence)
            if sentence_start == -1:
                sentence_start = 0
            sentence_elements.append({
                "content": sentence,
                "heading_path": heading_path,
                "start": start_pos + sentence_start,
                "end": start_pos + sentence_start + len(sentence)
            })
        
        sub_chunks = self._recursive_chunk(sentence_elements, chunk_size, overlap)
        
        if not sub_chunks:
            return self._split_by_words(element, chunk_size)
        
        return sub_chunks
    
    def _split_by_words(self, element: Dict, chunk_size: int) -> List[Dict]:
        """按单词/字符级别拆分元素
        
        当无法按句子拆分时使用此方法。
        
        Args:
            element: 要拆分的元素
            chunk_size: 最大块大小
        
        Returns:
            拆分后的块列表
        """
        import re
        
        content = element["content"]
        heading_path = element.get("heading_path")
        start_pos = element.get("start", 0)
        
        chinese_chars = re.findall(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff]+', content)
        
        if not chinese_chars:
            chinese_chars = list(content)
        
        chunks = []
        current_chars = []
        current_size = 0
        
        for chars in chinese_chars:
            char_len = len(chars)
            
            if current_size + char_len > chunk_size and current_chars:
                chunk_text = ''.join(current_chars)
                chunks.append({
                    "content": chunk_text,
                    "heading_path": heading_path,
                    "start": start_pos,
                    "end": start_pos + len(chunk_text)
                })
                start_pos += len(chunk_text)
                current_chars = []
                current_size = 0
            
            current_chars.append(chars)
            current_size += char_len
        
        if current_chars:
            chunk_text = ''.join(current_chars)
            chunks.append({
                "content": chunk_text,
                "heading_path": heading_path,
                "start": start_pos,
                "end": start_pos + len(chunk_text)
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割文本
        
        支持中文和英文标点符号。
        
        Args:
            text: 要分割的文本
        
        Returns:
            句子列表
        """
        import re
        
        sentence_endings = r'[。！？.!?]+'
        parts = re.split(sentence_endings, text)
        
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            if i < len(parts) - 1:
                sentences.append(part)
            else:
                if part:
                    sentences.append(part)
        
        return sentences
    
    def _split_paragraphs(self, text: str) -> List[Dict]:
        """分割段落"""
        lines = text.splitlines()
        paragraphs = []
        current_para = []
        heading_stack = []
        char_pos = 0
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('#'):
                if current_para:
                    content = "\n".join(current_para).strip()
                    if content:
                        paragraphs.append({
                            "content": content,
                            "heading_path": " > ".join(heading_stack) if heading_stack else None,
                            "start": char_pos - len(content),
                            "end": char_pos
                        })
                    current_para = []
                
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.lstrip('#').strip()
                
                if level <= len(heading_stack):
                    heading_stack = heading_stack[:level-1]
                heading_stack.append(title)
            else:
                if stripped:
                    current_para.append(line)
                elif current_para:
                    content = "\n".join(current_para).strip()
                    if content:
                        paragraphs.append({
                            "content": content,
                            "heading_path": " > ".join(heading_stack) if heading_stack else None,
                            "start": char_pos - len(content),
                            "end": char_pos
                        })
                    current_para = []
            
            char_pos += len(line) + 1
        
        if current_para:
            content = "\n".join(current_para).strip()
            if content:
                paragraphs.append({
                    "content": content,
                    "heading_path": " > ".join(heading_stack) if heading_stack else None,
                    "start": char_pos - len(content),
                    "end": char_pos
                })
        
        return paragraphs if paragraphs else [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    
    def _save_chunks(self, chunks: List[Dict], doc_filename: str):
        """保存分块到用户目录"""
        import json
        chunk_filename = f"{os.path.splitext(doc_filename)[0]}_chunks.json"
        chunk_path = os.path.join(self.user_knowledge_base_path, "chunks", chunk_filename)
        
        # 保存分块信息
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分块已保存到: {chunk_path}")
    
    def _index_chunks(self, chunks: List[Dict], source: str):
        """索引文本块"""
        if not self.embedder or not self.qdrant_client:
            return
        
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.embedder.encode(chunk["content"])
            
            metadata = {
                    "content": chunk["content"],
                    "source": source,
                    "namespace": self.user_namespace,
                    "user_id": self.user_id,
                    "heading_path": chunk.get("heading_path"),
                    "start": chunk.get("start"),
                    "end": chunk.get("end")
                }
        
            points.append({"memory_id": f"{self.user_namespace}_{source}_{idx}","vector":vector,"metadata":metadata})
        
        self.qdrant_client.add_batch([p["memory_id"] for p in points], [p["vector"] for p in points], [p["metadata"] for p in points])
        
        print(f"✅ 索引了 {len(points)} 个文本块")

def main():
    """测试RAGManager的主要功能"""
    print("=" * 50)
    print("开始测试 RAGManager")
    print("=" * 50)
    
    # 创建测试用的RAGConfig
    
    config = RAGConfig()
    
    # 初始化RAGManager
    print("\n1. 初始化 RAGManager...")
    rag_manager = RAGManager(user_id="user_xiaoming", config=config)
    print("✅ RAGManager 初始化成功")
    
    # 测试添加文本
    print("\n2. 测试添加文本...")
    test_text = """
    # 人工智能简介
    
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    ## 机器学习
    
    机器学习是人工智能的一个子集，它使用统计技术使计算机能够从数据中学习，而无需明确编程。
    
    ## 深度学习
    
    深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的工作方式。
    """
    
    result = rag_manager.add_text(test_text, source="AI介绍文档")
    print(result)
    
    # 测试列出文档
    print("\n3. 测试列出文档...")
    result = rag_manager.list_documents()
    print(result)
    
    # 创建一个测试文本文件
    print("\n4. 测试索引文档...")
    test_file_path = "电子商务.md"
    
    result = rag_manager.index_document(test_file_path)
    print(result)
    
    # 再次列出文档
    print("\n5. 再次列出文档...")
    result = rag_manager.list_documents()
    print(result)
    
    # 清理测试文件
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\n✅ 已清理测试文件: {test_file_path}")
    
    print("\n" + "=" * 50)
    print("RAGManager 测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
