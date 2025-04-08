from llama_index.tools.agentql import AgentQLRestAPIToolSpec
import os
import asyncio


class BrowserTool:
    def __init__(self, agentql_api_key: str):
        self.agentql_api_key = agentql_api_key

        os.environ["AGENTQL_API_KEY"] = agentql_api_key

        self.agentql_rest_api_tool = self.__create_browser_tool()

    @classmethod
    def __create_browser_tool(cls):
        # Create the REST API tool (this doesn't need async)
        agentql_rest_api_tool = AgentQLRestAPIToolSpec(is_stealth_mode_enabled=True)

        return agentql_rest_api_tool

    async def __async_scrape_url(self, url: str, query: str = None, prompt: str = None):
        if query is None and prompt is None:
            raise ValueError("Please provide at least a query or a prompt.")

        # Use the REST API tool (with await since it's a coroutine)
        if prompt:
            result = await self.agentql_rest_api_tool.extract_web_data_with_rest_api(
                url=url,
                prompt=prompt
            )
        else:
            result = await self.agentql_rest_api_tool.extract_web_data_with_rest_api(
                url=url,
                query=query
            )

        return result

    def scrape_url(self, url: str, query: str = None, prompt: str = None):
        result = asyncio.run(self.__async_scrape_url(url, query, prompt))
        return result
