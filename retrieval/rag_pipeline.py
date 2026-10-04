"""
rag_pipeline.py
===============
文档文本处理工具（记忆系统 v2 保留部分）。

旧的 Chroma 文档向量库 / BM25 混合检索 / RAGPipeline 已迁移至 Infinity
（见 memory_stack.InfinityStore），本模块只保留纯文本处理函数：
  - sentence_tokenize / chunk_sentences：中文句切分 + 分块（供文档索引用）
  - extract_text：按扩展名提取文本
  - save_upload：保存上传文件到用户目录
"""

import re
from pathlib import Path
from typing import List
from datetime import datetime

# ==========================================
# 0. 文件处理 & 上传目录
# ==========================================
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def extract_text(file_path: Path) -> str:
    """根据文件扩展名提取文本内容"""
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
               ".html", ".css", ".csv", ".xml", ".log"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return f"[需安装 PyPDF2] 无法解析 PDF: {file_path.name}"

    if ext == ".docx":
        try:
            from docx import Document as DocxDoc
            doc = DocxDoc(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return f"[需安装 python-docx] 无法解析 DOCX: {file_path.name}"

    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return (
            f"[图片文件: {file_path.name}]\n"
            f"图片已上传，当前环境未配置 OCR/视觉模型。"
            f"如需分析图片内容，请使用支持多模态的模型或手动描述图片内容。"
        )

    return f"[不支持的文件类型] {file_path.name}"


def save_upload(content: bytes, filename: str, user_id: int = None) -> Path:
    """保存上传文件到用户专属目录，返回路径。

    ⚠️ 用户隔离：文件保存到 ./uploads/{user_id}/ 子目录。
    user_id 为 None 时回退到 ./uploads/（兼容 CLI 模式）。
    """
    safe_name = re.sub(r"[^\w.\-]", "_", filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_id is not None:
        user_dir = UPLOAD_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / f"{timestamp}_{safe_name}"
    else:
        dest = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    dest.write_bytes(content)
    print(f"--> [Upload] 文件已保存: {dest} (user={user_id})")
    return dest


# ==========================================
# 1. 严格 NLP 句尾检测
# ==========================================
# 换行分隔（段落边界也是句子边界）
_NL_SEP = re.compile(r"\n{2,}")


def sentence_tokenize(text: str, max_sent_len: int = 300) -> List[str]:
    """严格 NLP 句尾检测：中英文混合句尾边界切分

    处理顺序：
    1. 先按段落（连续2+换行）粗切
    2. 每个段落内按句尾标点精切（中文标点后允许可选空白）
    3. 超长句子（> max_sent_len）强制按逗号/分号二次切分
    4. 仍超长的按字符数硬切（安全网）
    5. 合并过短句、过滤空句
    """
    if not text or not text.strip():
        return []

    # Step 1: 段落切分
    paragraphs = _NL_SEP.split(text.strip())

    # Step 2: 句尾标点精切
    # 中文：。！？；后允许可选空白（\s*），修复 PDF/OCR 文本中 "。\\n" 不切分的问题
    # 英文：.!? 后跟（空白+大写字母）或换行或结尾，或无空白直接跟非空白字符（兼容格式不佳的文本）
    _SENT_SPLIT = re.compile(
        r"(?<=[。！？；])\s*"
        r"|(?<=[.!?])(?=\s+\S|\n|$)"
    )

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        parts = _SENT_SPLIT.split(para)
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:
                sentences.append(part)

    # Step 3: 超长句子强制二次切分（安全网——先按逗号/分号，再按字符数）
    sentences = _force_split_long(sentences, max_sent_len)

    # Step 4: 合并过短的句子（< 5 字符）到前一句
    merged = []
    for s in sentences:
        if merged and len(s) < 5:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)

    return merged


def _force_split_long(sentences: List[str], max_len: int = 300) -> List[str]:
    """超长句子强制切分：先按逗号/分号切，仍超长则按字符数硬切。"""
    result = []
    for s in sentences:
        if len(s) <= max_len:
            result.append(s)
            continue

        # 第一级：按中文逗号、分号、冒号 + 英文逗号切分
        sub_parts = re.split(r"(?<=[，、：；,;:])\s*", s)
        for part in sub_parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_len:
                result.append(part)
            else:
                # 第二级：按字符数硬切（保留 max_len 字符一块，最后一块可能更短）
                for i in range(0, len(part), max_len):
                    chunk = part[i:i + max_len].strip()
                    if chunk:
                        result.append(chunk)
    return result


# ==========================================
# 2. Chunk 切分（20% overlap）
# ==========================================
def chunk_sentences(
    sentences: List[str],
    chunk_size: int = 8,
    overlap_ratio: float = 0.2,
) -> List[str]:
    """将句子列表切分为 chunk，相邻 chunk 之间重叠 20%

    Args:
        sentences: 句子列表
        chunk_size: 每个 chunk 包含的句子数
        overlap_ratio: 重叠比例（0.2 = 每次滑动 chunk_size * 0.2 句）

    Returns:
        chunk 字符串列表
    """
    if not sentences:
        return []

    overlap = max(1, int(chunk_size * overlap_ratio))  # 至少重叠 1 句
    step = chunk_size - overlap
    if step <= 0:
        step = 1

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i:i + chunk_size]
        if chunk_sents:
            chunks.append(" ".join(chunk_sents))
        i += step

    return chunks
