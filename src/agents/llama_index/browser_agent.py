from typing import Optional, List

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.lmstudio import LMStudio
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool
from pydantic import Field

from src.tools.llama_index.browser_tool import BrowserTool
import asyncio


class BrowserAgent(ReActAgent):
    count_search_calls: int = Field(default=0)
    max_search_calls: int = Field(default=10)

    """
    A browser agent that uses a tool to navigate the web to a specific URL and
    extract information requested by the user as a query or as a prompt. The agent
    also has access to a tool that allows to execute a search on the web.

    Parameters
    ----------
    agentql_api_key : str
        The API key to use to call the browser tool.
    name : str, optional
        The name of the agent. Defaults to 'BrowserAgent'.
    description : str, optional
        The description of the agent. Defaults to
        'A browser agent that uses a tool to navigate the web to a specific URL
        and extract information requested by the user as a query or as a prompt. The
        agent also has access to a tool that allows to execute a search on the web'.
    system_prompt : str, optional
        The system prompt to use to format the query. Defaults to
        'You are a browser agent that has access to a tool to navigate the web to a
        specific URL and extract information requested by the user as a query or as
        a prompt. You also have access to a tool that allows to execute a search on
        the web with a query given by the user.'
    model : str, optional
        The name of the model to use. Defaults to 'qwen2.5-7b-instruct-1m'.
    api_base : str, optional
        The base URL of the API. Defaults to 'http://localhost:1234/v1'.
    timeout : int, optional
        The timeout to use to call the API. Defaults to 120 seconds.
    verbose : bool, optional
        Whether to print debug information. Defaults to True.

    Attributes
    ----------
    name : str
        The name of the agent.
    description : str
        The description of the agent.
    system_prompt : str
        The system prompt to use to format the query.
    llm : LLM
        The LLM to use to generate code.
    tools : List[Tool]
        The tools available to the agent.
    """

    def __init__(self, agentql_api_key: str,
                 name: str = 'BrowserAgent',
                 description: str = 'A research agent that searches the web using DuckDuckGo and visits specific URL '
                                    'to then scrape and extract the content. It must not exceed 2 searches total, and '
                                    'must avoid repeating the same query. The user will input a query or a prompt '
                                    'specifying what content to search or what URL to visit and the type of '
                                    'information to extract from the results. Once sufficient information is '
                                    'collected, it should hand off to the WriteAgent.',
                 system_prompt: str = 'You are the ResearchAgent, a browser agent that has the capabilities to search '
                                      'the web or visit a specific URL to then scrape and extract the content. Your '
                                      'goal is to gather the information requested by the user from the web or from '
                                      'the specified URL. Only perform at most 2 distinct searches. If you have enough '
                                      'info or have reached 2 searches, handoff to the next agent. Avoid infinite '
                                      'loops!',
                 model: str = 'qwen2.5-7b-instruct-1m',
                 api_base: str = 'http://localhost:1234/v1',
                 timeout: int = 120,
                 verbose: bool = True,
                 can_handoff_to: Optional[List[str]] = None) -> None:
        """
        Constructor method of the BrowserAgent class.

        Parameters
        ----------
        agentql_api_key : str
            The API key to use to call the browser tool.
        name : str, optional
            The name of the agent. Defaults to 'BrowserAgent'.
        description : str, optional
            The description of the agent. Defaults to
            'A browser agent that uses a tool to navigate the web to a specific URL
            and extract information requested by the user as a query or as a prompt. The
            agent also has access to a tool that allows to execute a search on the web'.
        system_prompt : str, optional
            The system prompt to use to format the query. Defaults to
            'You are a browser agent that has access to a tool to navigate the web to a
            specific URL and extract information requested by the user as a query or as
            a prompt. You also have access to a tool that allows to execute a search on
            the web with a query given by the user.'
        model : str, optional
            The name of the model to use. Defaults to 'qwen2.5-7b-instruct-1m'.
        api_base : str, optional
            The base URL of the API. Defaults to 'http://localhost:1234/v1'.
        timeout : int, optional
            The timeout to use to call the API. Defaults to 120 seconds.
        verbose : bool, optional
            Whether to print debug information. Defaults to True.
        """

        if not can_handoff_to:
            can_handoff_to = ["WriteAgent"]

        # define a tool to browse a specific url
        browser_tool = BrowserTool(agentql_api_key=agentql_api_key)
        browse_url = browser_tool.agentql_rest_api_tool.to_tool_list()

        # define a tool to search the web
        search_web = FunctionTool.from_defaults(DuckDuckGoSearchToolSpec().duckduckgo_full_search)

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
            tools=browse_url + [search_web],
            llm=llm,
            verbose=verbose,
            can_handoff_to=can_handoff_to
        )

    async def __run(self, user_msg: str) -> str:
        """
        Run the agent.

        Parameters
        ----------
        user_msg : str
            The message from the user.

        Returns
        -------
        str
            The response from the agent.
        """

        if self.count_search_calls >= self.max_search_calls:
            return "Search limit reached, no more searches allowed."

        # perform the query
        response = await self.run(user_msg)
        self.count_search_calls += 1
        return response

    def chat(self, user_msg: str) -> str:
        """
        Chat with the agent.

        Parameters
        ----------
        user_msg : str
            The message from the user.

        Returns
        -------
        str
            The response from the agent.
        """

        return asyncio.run(self.__run(user_msg))
