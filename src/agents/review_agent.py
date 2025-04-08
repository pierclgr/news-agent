import asyncio
from typing import Optional, List

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.lmstudio import LMStudio


class ReviewAgent(ReActAgent):
    def __init__(self,
                 name: str = 'ReviewAgent',
                 description: str = "Reviews the final report for correctness. Approves or requests changes.",
                 system_prompt: str = "You are the ReviewAgent. Read the report, provide feedback, and "
                                      "either approve or request revisions. If revisions are needed, handoff to "
                                      "WriteAgent.",
                 model: str = 'qwen2.5-7b-instruct-1m',
                 api_base: str = 'http://localhost:1234/v1',
                 timeout: int = 120,
                 verbose: bool = True,
                 can_handoff_to: Optional[List[str]] = None) -> None:
        if not can_handoff_to:
            can_handoff_to = ["WriteAgent"]

        # define the llm
        llm = LMStudio(
            model_name=model,
            base_url=api_base,
            timeout=timeout
        )

        # initialize the agent
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=[self.review_report],
            llm=llm,
            verbose=verbose,
            can_handoff_to=can_handoff_to
        )

    async def __run(self, user_msg: str) -> str:
        # perform the query
        response = await self.run(user_msg)
        return response

    def chat(self, user_msg: str) -> str:
        return asyncio.run(self.__run(user_msg))

    @staticmethod
    async def review_report(ctx: Context, review: str) -> str:
        """Review the report and store feedback in the shared context."""
        current_state = await ctx.get("state")
        current_state["review"] = review
        await ctx.set("state", current_state)
        return "Report reviewed."

