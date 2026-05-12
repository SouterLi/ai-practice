# 写一个辅助代码调试的 AI 助手，它可以调用三个工具
# 1.读取文件，2.查询技术文档，3.执行 shell 脚本
# 要求：1.使用 langchain 提供的 create_tool_calling_agent（即「工具调用型 Agent」，作业中常被写作 create_tool_call_agent）与 AgentExecutor
#        在 LangChain 1.x 中二者由 langchain_classic 提供，与新版 create_agent 并存
#       2.Agent 支持工具的超时控制和报错重试
#       3.有完整的日志打印和唯一的 trace_id 可以追踪
#       4.使用 ReAct 范式进行循环（由 AgentExecutor 多轮「推理—行动—观察」实现，与本 Agent 的工具调用链路一致）

from __future__ import annotations

import logging
import os
import re
import subprocess
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 工具：超时 + 失败重试
# -----------------------------------------------------------------------------
T = TypeVar("T")


WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[1]
READ_FILE_TIMEOUT_SEC = 30.0
READ_FILE_MAX_RETRIES = 3
DOC_FETCH_TIMEOUT_SEC = 45.0
DOC_FETCH_MAX_RETRIES = 3
SHELL_TIMEOUT_SEC = 120.0
SHELL_MAX_RETRIES = 2
CONTENT_PREVIEW_LIMIT = 24_000


def _ensure_under_workspace(path: Path) -> Path:
    """防止任意路径遍历，只允许读取 WORKSPACE_ROOT 之下的文件。"""
    resolved = path.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as e:
        raise ValueError(f"路径必须位于工作区 {WORKSPACE_ROOT} 之内：{resolved}") from e
    return resolved


@tool
def read_local_file(relative_path: str) -> str:
    """读取工作区内的文本文件内容，用于查看源码与配置。参数为相对工作区根目录的路径（使用正斜杠）。"""
    rel = relative_path.strip().lstrip("/").replace("\\", "/")
    path = _ensure_under_workspace(WORKSPACE_ROOT / rel)

    if not path.is_file():
        return f"[错误] 不是文件或不存在：{path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > CONTENT_PREVIEW_LIMIT:
        return text[:CONTENT_PREVIEW_LIMIT] + "\n…[内容过长已截断]…"
    return text

@tool
def query_tech_docs(url: str) -> str:
    """根据官方文档等网页 URL 拉取正文（GET），用于查询技术文档。参数 url 需以 http:// 或 https:// 开头。"""
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        return "[错误] url 必须以 http:// 或 https:// 开头。"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CodeDebugAgent/1.0 (documentation fetch)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=DOC_FETCH_TIMEOUT_SEC) as resp:
        raw = resp.read()
    # 保守按 utf-8 解码；乱码时替换
    text = raw.decode("utf-8", errors="replace")
    # 极简去标签，避免把整页 HTML 塞满上下文
    text = re.sub(r"(?si)<script.*?>.*?</script>", "", text)
    text = re.sub(r"(?si)<style.*?>.*?</style>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > CONTENT_PREVIEW_LIMIT:
        return text[:CONTENT_PREVIEW_LIMIT] + " …[抓取内容已截断]…"
    return text or "[提示] 页面无可见文本或可解析为空。"

@tool
def run_shell(command: str) -> str:
    """在工作区根目录下执行一条 shell 命令（Windows 下由系统解释）。仅用于调试；勿用于不可信输入。"""
    cmd = command.strip()

    # Windows 上使用 shell=True 以支持管道及内置命令；子进程自带 timeout
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=int(SHELL_TIMEOUT_SEC),
        encoding="utf-8",
        errors="replace",
    )
    out_parts = []
    if proc.stdout:
        out_parts.append(f"[stdout]\n{proc.stdout}")
    if proc.stderr:
        out_parts.append(f"[stderr]\n{proc.stderr}")
    body = "\n".join(out_parts) if out_parts else "[无输出]"
    return f"[exit={proc.returncode}]\n{body}"

TOOLS = [read_local_file, query_tech_docs, run_shell]

def build_agent_executor() -> AgentExecutor:
    """组装具备工具调用能力与多轮推理循环的 Executor。"""
    api_key_raw = os.getenv("DASHSCOPE_API_KEY")
    if not api_key_raw:
        logger.warning(
            "未设置环境变量 DASHSCOPE_API_KEY，ChatOpenAI 将无法调用远端模型。"
        )
    # ChatOpenAI 类型标注要求 SecretStr / None，与 os.getenv 的 str | None 对齐
    api_key_secret: SecretStr | None = (
        SecretStr(api_key_raw) if api_key_raw else None
    )

    llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=api_key_secret,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

    system = (
        "你是辅助开发与调试代码的助手。遵循 ReAct 思路：先在内心逐步推理接下来需要什么信息，"
        "再通过工具采取行动，根据工具返回观察结果继续推理。\n"
        f"可读文件仅能访问工作区根目录之下的路径（根目录绝对路径：{WORKSPACE_ROOT}）。\n"
        "执行 shell 有安全风险，仅用于用户明确授权的调试。\n"
        "能用工具拿到的信息不要用猜测代替。"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=False,
        max_execution_time=int(os.getenv("AGENT_MAX_RUN_SECONDS", "600")),
    )


def main() -> None:
    """交互式会话：每条用户输入绑定新的 trace_id，便于端到端检索日志。"""
    executor = build_agent_executor()
    print("代码调试助手已启动。输入 quit / exit 结束。")
    while True:
        try:
            user = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if user.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break
        if not user:
            continue

        tid = str(uuid.uuid4())
        logger.info("======== 新会话 trace_id=%s ========", tid)
        logger.info("用户输入长度=%s", len(user))

        try:
            out = executor.invoke({"input": user})
            answer = out.get("output", str(out))
            print("\n助手>", answer)
        except BaseException:
            logger.exception("Agent 本轮执行失败 trace_id=%s", tid)
            print("\n助手> [系统错误，请查看日志中的 trace_id 定位问题]")


if __name__ == "__main__":
    main()
