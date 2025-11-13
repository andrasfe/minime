import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


ROOT = Path(__file__).resolve().parents[1]


def test_extract_question_from_prompt():
    from scripts.minime_client import _extract_question_from_prompt

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helper."), ("human", "{question}")]
    )

    prompt_value = prompt.invoke({"question": "What is the meaning of life?"})
    question = _extract_question_from_prompt(prompt_value)

    assert question == "What is the meaning of life?"


@pytest.mark.asyncio
async def test_call_mcp_answer(monkeypatch):
    from scripts.minime_client import _call_mcp_answer

    class DummyResult:
        data = {"answer": "42"}
        structured_content = None
        content = None

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def call_tool(self, tool_name, payload):
            assert tool_name == "answer_question"
            assert payload == {"question": "Test question"}
            return DummyResult()

    monkeypatch.setattr("scripts.minime_client.Client", lambda url: DummyClient())

    result = await _call_mcp_answer("Test question", "http://localhost:9999")

    assert result == {"answer": "42"}


@pytest.mark.integration
def test_minime_client_end_to_end(tmp_path):
    port = 8123
    env = os.environ.copy()
    env["MCP_SERVER_PORT"] = str(port)
    env["MCP_SERVER_HOST"] = "127.0.0.1"

    subprocess.run(
        ["docker-compose", "up", "-d", "postgres"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )

    server = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for server to become ready
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                conn.request("GET", "/mcp")
                resp = conn.getresponse()
                if resp.status in (200, 405):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            stdout, stderr = server.communicate(timeout=5)
            raise RuntimeError(f"MCP server failed to start:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

        question = "What domains of quantum research is minime focused on?"
        cli_cmd = [
            sys.executable,
            "scripts/minime_client.py",
            "--question",
            question,
            "--url",
            f"http://127.0.0.1:{port}/mcp",
            "--format",
            "json",
        ]

        result = subprocess.run(
            cli_cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output = json.loads(result.stdout)
        assert output["question"] == question
        assert "result" in output
        assert output["result"], "Expected non-empty result from MCP server"

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

