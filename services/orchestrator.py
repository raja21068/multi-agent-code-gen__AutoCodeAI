"""
services/orchestrator.py — Central pipeline coordinator.

Wires together: memory retrieval → planning → agent execution loop
→ sandbox → critic review → memory persistence → streaming output.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable

from core.agents.agents import (
    CoderAgent,
    CriticAgent,
    DebuggerAgent,
    MemoryAgent,
    PlannerAgent,
    TesterAgent,
)
from core.tools.sandbox import DockerSandbox
from memory.repo_indexer import RepoIndexer

logger = logging.getLogger(__name__)

Callback = Callable[[str], Awaitable[None]] | None


class Orchestrator:
    def __init__(self, repo_path: str | None = None) -> None:
        self.planner  = PlannerAgent()
        self.coder    = CoderAgent()
        self.tester   = TesterAgent()
        self.debugger = DebuggerAgent()
        self.critic   = CriticAgent()
        self.memory   = MemoryAgent()
        self.sandbox  = DockerSandbox()
        self.indexer  = RepoIndexer(repo_path) if repo_path else None
        if self.indexer:
            self.indexer.start_watching()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_content(filepath: str) -> str:
        if not filepath:
            return ""
        try:
            return Path(filepath).read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            logger.warning("File not found: %s", filepath)
            return ""

    async def _notify(self, callback: Callback, msg: str) -> None:
        if callback:
            await callback(msg)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        context_files: list[str],
        callback: Callback = None,
    ) -> list[dict]:
        await self._notify(callback, "🧠 Retrieving memory …\n")
        memory_context = self.memory.retrieve(task)

        if self.indexer:
            snippets = self.indexer.retrieve_relevant(task)
            repo_ctx = "\n".join(
                s["metadata"]["content"][:300] for s in snippets
            )
            memory_context += f"\nRepo context:\n{repo_ctx}"

        await self._notify(callback, "📝 Creating plan …\n")
        plan = await self.planner.create_plan(task, memory_context, "")
        await self._notify(callback, f"✅ Plan: {plan.get('explanation', '')}\n\n")

        results: list[dict] = []
        latest_code = ""

        for step in plan.get("steps", []):
            agent = step.get("agent", "")
            desc  = step.get("description", "")
            await self._notify(callback, f"⚙️  {agent.capitalize()}: {desc}\n")

            if agent == "coder":
                existing = self._get_file_content(step.get("file", ""))
                latest_code = await self._stream_coder(
                    desc, context_files, results, memory_context, existing, callback
                )
                results.append({"step": desc, "output": latest_code, "type": "code"})

            elif agent == "tester" and latest_code:
                test_code = await self.tester.generate_tests(latest_code, desc)
                stdout, stderr = self.sandbox.run_code(latest_code, test_code)
                output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
                await self._notify(callback, f"🧪 Tests:\n{output[:800]}\n")
                results.append({"step": desc, "output": output, "type": "test"})

                # auto-debug on failure
                if stderr or "FAILED" in output or "ERROR" in output:
                    await self._notify(callback, "🔧 Debugging …\n")
                    latest_code = await self.debugger.fix(latest_code, output)
                    results.append(
                        {"step": "auto-debug", "output": latest_code, "type": "code"}
                    )

            elif agent == "debugger" and latest_code:
                error_ctx   = results[-1]["output"] if results else ""
                latest_code = await self.debugger.fix(latest_code, error_ctx)
                results.append({"step": desc, "output": latest_code, "type": "code"})

            elif agent == "critic":
                review = await self.critic.review(results, task)
                await self._notify(callback, f"📋 Review: {review}\n")
                results.append({"step": desc, "output": review, "type": "review"})

        # final critic pass
        final_review = await self.critic.review(results, task)
        await self._notify(callback, f"\n📋 Final review: {final_review}\n")
        if "PASS" in final_review and latest_code:
            self.memory.store(task, latest_code)

        await self._notify(callback, "\n✅ Done.\n")
        return results

    async def _stream_coder(
        self,
        subtask: str,
        context_files: list[str],
        results: list[dict],
        memory: str,
        existing_code: str,
        callback: Callback,
    ) -> str:
        full_code = ""
        async for token in self.coder.stream_code(
            subtask, context_files, results, memory, existing_code
        ):
            full_code += token
            await self._notify(callback, token)
        return full_code

    # ------------------------------------------------------------------
    # Streaming generator (for SSE / WebSocket)
    # ------------------------------------------------------------------

    async def run_streaming(
        self,
        task: str,
        context_files: list[str],
    ) -> AsyncGenerator[str, None]:
        """Yield string chunks suitable for SSE or WebSocket streaming."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _cb(msg: str) -> None:
            await queue.put(msg)

        async def _worker() -> None:
            try:
                await self.run(task, context_files, callback=_cb)
            finally:
                await queue.put(None)           # sentinel

        worker = asyncio.create_task(_worker())
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
        await worker

    # ------------------------------------------------------------------
    # Parallel execution helper
    # ------------------------------------------------------------------

    async def run_parallel(
        self,
        steps: list[dict],
        context_files: list[str],
    ) -> list:
        async def _exec(step: dict):
            if step["agent"] == "coder":
                return await self.coder.generate_code(
                    step["description"], context_files, [], "", ""
                )
            if step["agent"] == "tester":
                return await self.tester.generate_tests("", step["description"])
            return None

        return await asyncio.gather(
            *[_exec(s) for s in steps], return_exceptions=True
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self.indexer:
            self.indexer.stop()
