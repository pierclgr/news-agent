from v1.src.agents.articles_multi_agent import ArticlesMultiAgent
from v1.src.config import Config
from v1.src.tools.articles_database import ArticlesDatabase
from v1.src.tools.news_retriever import NewsRetriever

if __name__ == "__main__":
    config = Config("config/config.json")
    retriever = NewsRetriever(config=config)
    articles_database = ArticlesDatabase(config=config)

    # fetch new articles
    new_articles = retriever.fetch_articles_from_sources(articles_database=articles_database)

    # update the articles database with the new articles
    articles_database.update(articles=new_articles)

    # initialize the agent
    agent = ArticlesMultiAgent(articles_database=articles_database, config=config)

    # run the agent and chat with it
    agent.start()
