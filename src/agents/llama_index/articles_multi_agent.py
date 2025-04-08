import asyncio
import logging
import sys
from typing import List, Any

from llama_index.core.tools import FunctionTool
from llama_index.llms.lmstudio import LMStudio

from src.config import Config
from src.tools.articles_database import ArticlesDatabase
from datetime import datetime
from src.utils import import_agent_class, to_camel_case, get_printable_articles_list
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCallResult, ToolCall, ReActAgent


class ArticlesMultiAgent(AgentWorkflow):
    DEFAULT_MODEL: str = "qwen2.5-7b-instruct-1m"
    DEFAULT_API_BASE: str = "http://localhost:1234/v1"
    DEFAULT_TIMEOUT: int = 120
    DEFAULT_NAME: str = "root_agent"
    DEFAULT_VERBOSE: bool = True
    DEFAULT_HANDOFFS: List[str] = [
        "BrowserAgent",
        "RetrieverAgent"
    ]
    DEFAULT_DESCRIPTION: str = ("The root agent that coordinates the other agents. It has the ability and the "
                                "responsibility to answer the questions of the user directly or otherwise, if help is "
                                "needed, to delegate and handoff the task to other agents. You can handoff to "
                                "RetrieverAgent and BrowserAgent."),
    DEFAULT_SYSTEM_PROMPT: str = ("You are a helpful assistant that coordinates the other agents. When the user asks "
                                  "for something, you can answer the question directly if you know the answer and "
                                  " don't need the help of additional agents, otherwise you will delegate the task and "
                                  "hand it off to either the BrowserAgent or the RetrieverAgent."),

    def __init__(self, articles_database: ArticlesDatabase, config: Config):
        self.articles_database = articles_database

        # create the agents
        agents = config.get("agents", [])
        manager_agent = [agent for agent in agents if agent.get("manager", False)][0]
        agents = [agent for agent in agents if not agent.get("manager", False)]

        managed_agents = []
        for agent in agents:
            agent_name = agent.get("name")
            agent_name_camel_case = to_camel_case(agent_name)
            agent_class = import_agent_class(agent_name)
            agent["name"] = agent_name_camel_case
            agent_instance = agent_class(
                **agent
            )

            managed_agents.append(agent_instance)

        model = manager_agent.get("model", self.DEFAULT_MODEL)
        name = manager_agent.get("name", self.DEFAULT_NAME)

        name = to_camel_case(name)

        # define the llm
        llm = LMStudio(
            model_name=model,
            base_url=manager_agent.get("api_base", self.DEFAULT_API_BASE),
            timeout=manager_agent.get("timeout", self.DEFAULT_TIMEOUT)
        )

        self.verbose = manager_agent.get("verbose", self.DEFAULT_VERBOSE)

        if self.verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            logging.getLogger("llama_index").setLevel(logging.DEBUG)

        # create the root agent
        root_agent = ReActAgent(
            name=name,
            description=manager_agent.get("description", self.DEFAULT_DESCRIPTION),
            system_prompt=manager_agent.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT),
            llm=llm,
            verbose=self.verbose,
            can_handoff_to=manager_agent.get("can_handoff_to", self.DEFAULT_HANDOFFS)
        )
        managed_agents.append(root_agent)

        super().__init__(
            agents=managed_agents,
            root_agent=name,
            verbose=self.verbose,
            initial_state={
                "report_content": "Not written yet.",
                "review": "Review required."
            },
        )

    def chat(self, user_msg: str) -> dict:
        return asyncio.run(self.__chat(user_msg))

    async def __chat(self, user_msg: str) -> dict:
        handler = self.run(
            user_msg=user_msg
        )

        current_agent = None
        response = ""
        async for event in handler.stream_events():
            if self.verbose:
                # Your existing logging code
                if (
                        hasattr(event, "current_agent_name")
                        and event.current_agent_name != current_agent
                ):
                    current_agent = event.current_agent_name
                    print(f"\n{'=' * 50}")
                    print(f"🤖 Agent: {current_agent}")
                    print(f"{'=' * 50}\n")
                elif isinstance(event, AgentOutput):
                    if event.response.content:
                        print("📤 Output:", event.response.content)
                        response = event.response.content
                    if event.tool_calls:
                        print(
                            "🛠️  Planning to use tools:",
                            [call.tool_name for call in event.tool_calls],
                        )
                elif isinstance(event, ToolCallResult):
                    print(f"🔧 Tool Result ({event.tool_name}):")
                    print(f"  Arguments: {event.tool_kwargs}")
                    print(f"  Output: {event.tool_output}")
                elif isinstance(event, ToolCall):
                    print(f"🔨 Calling Tool: {event.tool_name}")
                    print(f"  With arguments: {event.tool_kwargs}")

        final_state = await handler.ctx.get("state")
        if self.verbose:
            print("\n\n=============================")
            print("FINAL REPORT:\n")
            print(final_state["report_content"])
            print("=============================\n")

        report_content = ""
        if "report_content" in final_state:
            report_content = final_state["report_content"]

        # Review feedback (if any)
        review = ""
        if "review" in final_state and self.verbose:
            print("Review Feedback:", final_state["review"])
            review = final_state["review"]

        response = {
            "response": response,
            "report": report_content,
            "review": review
        }

        return response

    def start(self):
        current_hour = datetime.now().hour

        if 5 <= current_hour < 12:
            print("🤖 Good morning, Pier!")
        elif 12 <= current_hour < 18:
            print("🤖 Good afternoon, Pier!")
        else:
            print("🤖 Good evening, Pier!")

        # retrieve new articles
        _, new_articles = self.articles_database.get_new_articles()

        # print a list of articles
        if len(new_articles) > 0:
            print("🤖 There are some news! Here's the new articles:")
            articles = new_articles
        else:
            print("🤖 There are no new articles, here's a list of all the available articles:")
            _, articles = self.articles_database.get_all_articles()

        print(get_printable_articles_list(articles))

        while True:
            # open the interactive menu
            option = self.interactive_menu(
                {0: "Show recent articles",
                 1: "Show all articles",
                 2: "Explore a specific article"})

            # perform the selected option
            result = self.perform_selected_option(option,
                                                  articles=articles,
                                                  options={
                                                      0: self.articles_database.get_new_articles,
                                                      1: self.articles_database.get_all_articles,
                                                      2: lambda: self.explore_article(articles)}
                                                  )

            if result == "/bye":
                break

            print(result)

        print("🤖 Goodbye, Pier!")

    def perform_selected_option(self, option: Any, options: dict, articles: list,
                                force_response_code: str = "query_agent"):
        if isinstance(option, int):
            operation, result = options[option]()

            # if the operation executed returns an article list
            if operation == "articles_list":
                # if the obtained list is not empty
                if len(result) > 0:
                    print("🤖 Here's the list of articles you requested:")
                    # set the global articles list to the new one
                    articles[:] = result

                    # get the response of the agent as a printable articles list
                    response = get_printable_articles_list(articles)
                    response_code = "str"
                else:
                    # return as a response of the agent a message saying that there are no articles
                    response = "🤖 There's no article that matches your request."
                    response_code = "str"
            else:
                response = result
                response_code = operation
        else:
            # the user inserted a query in natural language that should be processed as requested by input
            response = option
            response_code = force_response_code

        # if the user inserted a query in natural language that is not bye
        if response_code == "query_agent" and response != "/bye":
            # chat with the agent for the response
            try:
                response = self.chat(response)
            except Exception as e:
                response = str(e)

        return response

    def explore_article(self, articles: list):
        # make the user select an article to explore from the list
        while True:
            option = input("🤖 Insert the number of the article you want to explore. If you don't know what to do, "
                           "simply ask me a question, otherwise, type /bye to exit!\n")
            try:
                # if the user inputted a valid article number from the list
                option = int(option)
                if 0 <= option < len(articles):
                    # break the loop to continue exploring the selected article
                    break
                else:
                    # reiterate the loop to ask the user to insert a valid article number
                    print("🤖 The option you selected is invalid, please select a valid option!")
            except ValueError:
                # in this case, the use has inserted a natural language query, so break the loop to return the query
                # itself that will be handled from the menu selection method
                break

        if isinstance(option, int):
            # extract the article selected by the user
            article = articles[option]

            # extract the article from the database using the url
            url = article[0]
            article_row = self.articles_database.articles_records[self.articles_database.articles_records["url"] == url]

            # extract the article content
            title = article_row["title"].values[0]
            source = article_row["source"].values[0]
            publish_date = article_row["publish_date"].values[0]
            has_text = article_row["has_text"].values[0]
            if not has_text:
                has_textual_content = ("browse the URL of the article to scrape and extract article information by "
                                       "handing off to BrowserAgent,")
            else:
                has_textual_content = ("retrieve the article and extract article information from the knowledge base "
                                       "by handing off to RetrieverAgent,")

            prompt_header = (
                f"Given this article with the title '{title}', published by {source} on "
                f"{publish_date} at the url '{url}',")

            # open the interactive menu for the current article
            article_option = self.interactive_menu(
                {0: "Summarize the article",
                 1: "Create a report",
                 2: "Search the web for more information"})

            # perform the selected option
            result = self.perform_selected_option(article_option,
                                                  articles=articles,
                                                  force_response_code="str",
                                                  options={
                                                      0: lambda: ("str",
                                                                  f"{prompt_header} {has_textual_content} then "
                                                                  f"summarize it."),
                                                      1: lambda: ("str", f"{prompt_header} {has_textual_content} then "
                                                                         f"create a "
                                                                         f"detailed report about it, with "
                                                                         f"specific insights on the conclusion "
                                                                         f"and the findings of the article, "
                                                                         f"plus explain if what is described in "
                                                                         f"the article is a game changer and, "
                                                                         f"if so how."),
                                                      2: lambda: ("str", f"{prompt_header} search on DuckDuckGo for "
                                                                         f"more information about the content of the "
                                                                         f"article by handing off to BrowserAgent.")
                                                  })
        else:
            result = option
        return "query_agent", result

    @staticmethod
    def interactive_menu(options: dict):
        print("🤖 What do you want me do to?")
        for key, value in options.items():
            print(f"\t{key}) {value}")

        while True:
            option = input("🤖 If you don't know what to do, simply ask me a question, otherwise, type /bye to exit!\n")
            try:
                option = int(option)
                if 0 <= option < len(options):
                    break
                else:
                    print("🤖 The option you selected is invalid, please select a valid option!")
            except ValueError:
                break

        return option

        # while True:
        #     query = input("\n🤖 What do you want me do to? \n")
        #     if "/bye" == query.lower():
        #         break
        #
        #     # perform a query
        #     try:
        #         response = self.chat(query)
        #     except Exception as e:
        #         response = str(e)
        #     print(response)
