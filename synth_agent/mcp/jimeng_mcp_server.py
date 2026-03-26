import httpx
import uuid
from typing import Dict, Any, List
import logging
import json
import os
import time
import hmac
import hashlib
import base64
from datetime import datetime
from fastmcp import FastMCP
from dotenv import load_dotenv


load_dotenv()

# ====================== 配置 ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jimeng-ai-mcp-server")


# 火山引擎配置
VOLCENGINE_ACCESS_KEY = os.environ.get("VOLCENGINE_ACCESS_KEY")
VOLCENGINE_SECRET_KEY = os.environ.get("VOLCENGINE_SECRET_KEY")
API_ENDPOINT = "https://visual.volcengineapi.com"

# 统一 MCP 实例
mcp = FastMCP(
    name="JimengAIServer",
    instructions="这个服务用来通过即梦 AI 3.1 生成图片。\n" 
                 "调用 generate_image_async 生成图片，返回图片 URL"
)

# ====================== 火山引擎签名认证 ======================
def hmac_sha256(key: bytes, msg: str) -> bytes:
    """使用 HMAC-SHA256 计算签名"""
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def generate_signature(method: str, uri: str, query_string: str, headers: Dict[str, str], body: str, secret_key: str) -> str:
    """
    生成火山引擎 API 签名
    参考: https://www.volcengine.com/docs/6369/67269
    """
    # 1. 构建规范化请求字符串
    # HTTPMethod + "\n" + CanonicalURI + "\n" + CanonicalQueryString + "\n" + CanonicalHeaders + "\n" + SignedHeaders + "\n" + HexEncode(Hash(RequestPayload))
    
    # 规范化 URI
    canonical_uri = uri if uri else "/"
    
    # 规范化 Query String
    canonical_query_string = query_string if query_string else ""
    
    # 规范化 Headers
    # 需要签名的 headers: host, x-date, x-content-sha256, content-type
    signed_headers_list = ["host", "x-date"]
    canonical_headers = ""
    for key in sorted(signed_headers_list):
        # 获取 header 值（不区分大小写）
        header_value = None
        for h_key, h_value in headers.items():
            if h_key.lower() == key:
                header_value = h_value.strip()
                break
        
        if header_value is None:
            raise ValueError(f"缺少必需的 header: {key}")
        
        canonical_headers += f"{key}:{header_value}\n"
    
    signed_headers = ";".join(sorted(signed_headers_list))
    
    # 请求体哈希
    payload_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    # 构建规范化请求字符串
    canonical_request = f"{method}\n{canonical_uri}\n{canonical_query_string}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    
    # 2. 构建签名字符串
    # Algorithm + "\n" + RequestDate + "\n" + CredentialScope + "\n" + HexEncode(Hash(CanonicalRequest))
    algorithm = "HMAC-SHA256"
    
    # 获取日期时间
    x_date = headers.get("X-Date") or headers.get("x-date")
    if not x_date:
        raise ValueError("缺少 X-Date header")
    
    request_date = x_date
    date_stamp = request_date[:8]  # YYYYMMDD
    region = "cn-north-1"
    service = "cv"
    credential_scope = f"{date_stamp}/{region}/{service}/request"
    
    string_to_sign = f"{algorithm}\n{request_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    
    # 3. 计算签名
    # 派生签名密钥
    k_date = hmac_sha256(secret_key.encode('utf-8'), date_stamp)
    k_region = hmac_sha256(k_date, region)
    k_service = hmac_sha256(k_region, service)
    k_signing = hmac_sha256(k_service, "request")
    
    # 计算最终签名
    signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    
    return signature


# ====================== 提交任务到即梦 AI ======================
def submit_task(prompt: str, width: int = 1024, height: int = 1024, seed: int = -1) -> Dict[str, Any]:
    """提交文生图任务"""
    method = "POST"
    uri = "/"
    query_string = "Action=CVSync2AsyncSubmitTask&Version=2022-08-31"
    
    # 构建请求体
    body = {
        "req_key": "jimeng_t2i_v31",
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "use_pre_llm": True
    }
    body_str = json.dumps(body, ensure_ascii=False)
    
    # 构建头部
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    headers = {
        "host": "visual.volcengineapi.com",
        "x-date": now,
        "content-type": "application/json"
    }
    
    # 生成签名
    signature = generate_signature(
        method=method,
        uri=uri,
        query_string=query_string,
        headers=headers,
        body=body_str,
        secret_key=VOLCENGINE_SECRET_KEY
    )
    
    # 构建 Authorization header
    credential = f"{VOLCENGINE_ACCESS_KEY}/{now[:8]}/cn-north-1/cv/request"
    signed_headers = "host;x-date"
    authorization = f"HMAC-SHA256 Credential={credential}, SignedHeaders={signed_headers}, Signature={signature}"
    
    # 添加 Authorization 到 headers
    headers["authorization"] = authorization
    
    try:
        # 准备请求
        url = f"{API_ENDPOINT}{uri}?{query_string}"
        
        # 发送请求
        response = httpx.post(
            url,
            content=body_str.encode('utf-8'),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
            
        if result.get("code") == 10000:
            return result.get("data", {})
        else:
            raise RuntimeError(f"API 错误: {result.get('message', 'Unknown error')}")
            
    except httpx.HTTPError as e:
        raise RuntimeError(f"请求失败: {e}")
    except Exception as e:
        raise RuntimeError(f"提交任务失败: {str(e)}")

# ====================== 查询任务状态 ======================
def query_task(task_id: str) -> Dict[str, Any]:
    """查询任务状态"""
    method = "POST"
    uri = "/"
    query_string = "Action=CVSync2AsyncGetResult&Version=2022-08-31"
    
    # 构建请求体
    body = {
        "req_key": "jimeng_t2i_v31",
        "task_id": task_id,
        "req_json": json.dumps({"return_url": True})
    }
    body_str = json.dumps(body, ensure_ascii=False)
    
    # 构建头部
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    headers = {
        "host": "visual.volcengineapi.com",
        "x-date": now,
        "content-type": "application/json"
    }
    
    # 生成签名
    signature = generate_signature(
        method=method,
        uri=uri,
        query_string=query_string,
        headers=headers,
        body=body_str,
        secret_key=VOLCENGINE_SECRET_KEY
    )
    
    # 构建 Authorization header
    credential = f"{VOLCENGINE_ACCESS_KEY}/{now[:8]}/cn-north-1/cv/request"
    signed_headers = "host;x-date"
    authorization = f"HMAC-SHA256 Credential={credential}, SignedHeaders={signed_headers}, Signature={signature}"
    
    # 添加 Authorization 到 headers
    headers["authorization"] = authorization
    
    try:
        # 准备请求
        url = f"{API_ENDPOINT}{uri}?{query_string}"
        
        # 发送请求
        response = httpx.post(
            url,
            content=body_str.encode('utf-8'),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
            
        if result.get("code") == 10000:
            return result.get("data", {})
        else:
            raise RuntimeError(f"API 错误: {result.get('message', 'Unknown error')}")
            
    except httpx.HTTPError as e:
        raise RuntimeError(f"请求失败: {e}")
    except Exception as e:
        raise RuntimeError(f"查询任务失败: {str(e)}")

# ====================== 生成图片并等待完成 ======================
def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = -1) -> List[str]:
    """生成图片并返回 URL 列表"""
    # 提交任务
    task_data = submit_task(prompt, width, height, seed)
    task_id = task_data.get("task_id")
    
    if not task_id:
        raise RuntimeError("未获取到 task_id")
    
    # 轮询查询状态
    max_retries = 60  # 最多轮询 60 次
    retry_interval = 2  # 每 2 秒查询一次
    
    for i in range(max_retries):
        time.sleep(retry_interval)
        result = query_task(task_id)
        status = result.get("status")
        
        if status == "done":
            image_urls = result.get("image_urls", [])
            if image_urls:
                return image_urls
            else:
                raise RuntimeError("生成失败: 未返回图片 URL")
        elif status in ["in_queue", "generating"]:
            logger.info(f"生成中... ({i+1}/{max_retries})")
        else:
            raise RuntimeError(f"任务失败: status={status}")
    
    raise RuntimeError("生成超时")

# ====================== MCP 工具：生成图片 ======================
@mcp.tool()
def generate_image_async(
    prompt: str = "a cute cat wearing a yellow hat",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1
) -> Dict[str, Any]:
    """使用即梦 AI 3.1 生成图片"""
    try:
        image_urls = generate_image(prompt, width, height, seed)
        return {
            "success": True,
            "image_urls": image_urls,
            "prompt": prompt
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ====================== MCP 工具：测试服务 ======================
@mcp.tool()
def ping() -> Dict[str, Any]:
    """测试 MCP 服务是否正常运行"""
    return {
        "success": True,
        "message": "服务正常运行",
        "service": "JimengAIServer",
        "version": "1.0.0"
    }

# ====================== 启动 MCP 服务 ======================
if __name__ == "__main__":
    logger.info("✅ 即梦 AI 3.1 MCP 服务启动：http://127.0.0.1:9000")
    mcp.run(transport="http", host="localhost", port=9000)
