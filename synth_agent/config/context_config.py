import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ContextConfig(BaseModel):
    """上下文构建配置

    Attributes:
        max_tokens: 最大 token 数量
        reserve_ratio: 为系统指令预留的比例(0.0-1.0)
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重(0.0-1.0)
        relevance_weight: 相关性权重(0.0-1.0)
    """
    max_tokens: int = Field(default=3000, description="最大 token 数量，默认3000")
    reserve_ratio: float = Field(default=0.2, description="为系统指令预留的比例(0.0-1.0)，默认0.2")
    min_relevance: float = Field(default=0.5, description="最低相关性阈值，默认0.1")
    enable_compression: bool = Field(default=True, description="是否启用压缩，默认True")
    recency_weight: float = Field(default=0.3, description="最近性权重(0.0-1.0)，默认0.3")
    relevance_weight: float = Field(default=0.7, description="相关性权重(0.0-1.0)，默认0.7")

    def model_post_init(self, __context: Any) -> None:
        """验证配置参数"""
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"

    