from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
import nltk
import ssl
from playwright.sync_api import sync_playwright
import re
from v1.src.tools.articles_database import ArticlesDatabase
from v1.src.config import Config


class NewsRetriever:
    DEFAULT_MAX_ARTICLES_PER_SITE = 20
    DEFAULT_SEARCH_TIMEOUT = 60000
    DEFAULT_WAIT_TIME = 3000
    DEEPMIND_URL = "https://deepmind.google"
    ANTHROPIC_URL = "https://www.anthropic.com"
    OPENAI_URL = "https://openai.com"
    GOOGLE_RESEARCH_URL = "https://research.google"
    HUGGINGFACE_URL = "https://huggingface.co"
    VERGE_URL = "https://www.theverge.com"
    WIRED_URL = "https://www.wired.com"
    VENTUREBEAT_URL = "https://venturebeat.com"
    TECHCRUNCH_URL = "https://techcrunch.com"
    AIBUSINESS_URL = "https://aibusiness.com"
    ARXIV_URL = "https://arxiv.org"
    ILPOST_URL = "https://www.ilpost.it"
    MISTRAL_URL = "https://mistral.ai"
    PERPLEXITY_URL = "https://www.perplexity.ai"
    XAI_URL = "https://x.ai"
    GOOGLE_BLOG_URL = "https://blog.google"
    GOOGLE_DEVELOPERS_BLOG_URL = "https://developers.googleblog.com"
    META_AI_URL = "https://ai.meta.com"

    def __init__(self, config: Config, user_agent: str = None):
        """
        Initialize a NewsRetrieverTool with a configuration file and a user agent.

        Parameters
        ----------
        config : AgentConfig
            The configuration file to use for this tool.
        user_agent : str, optional
            The user agent to use for fetching articles. If not provided, a default user agent is used.

        Raises
        ------
        ValueError
            If the configuration file is invalid.
        """

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt_tab')

        if not user_agent:
            user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36')

        self.user_agent = user_agent

        self.config = config

        # Extract websites from config
        websites = self.config.get("search.web.sites", None)
        if not websites:
            raise ValueError("No websites found in config.")

        self.websites = websites

        # Extract max number of articles per site from config
        self.max_articles_per_site = self.config.get("search.max_articles_per_site",
                                                     self.DEFAULT_MAX_ARTICLES_PER_SITE)

        # Extract search timeout from config
        self.search_timeout = self.config.get("search.timeout", self.DEFAULT_SEARCH_TIMEOUT)

        # Extract wait time from config
        self.wait_time = self.config.get("search.wait_time", self.DEFAULT_WAIT_TIME)

    def __scrape_websites(self, articles_database: ArticlesDatabase) -> List[Dict[str, Any]]:
        """
        Scrape articles from a list of websites.

        This function takes the list of websites provided in the configuration and scrapes articles from them. It looks
        for common article link patterns and extracts the article content using the `get_article_content` method.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing the article data. Each dictionary contains the article title, text, URL,
            authors, publish date, keywords, summary, source, and fetch date.
        """

        # Limit the number of sites to scrape
        websites_to_scrape = self.websites
        print(f"Scraping {len(websites_to_scrape)} website{'s' if len(websites_to_scrape) > 1 else ''} for articles...")

        articles = []

        # initialize the browser for scraping
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Run in headful mode to appear more human-like
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )

            # scrape each website
            for website_url in websites_to_scrape:
                print(f"Scraping website: {website_url}")

                # scrape the website with playwright
                page = context.new_page()
                # Apply stealth plugin

                page.goto(website_url, timeout=self.search_timeout)
                page.wait_for_timeout(self.wait_time)
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')

                site_articles = []
                if self.DEEPMIND_URL in website_url:
                    # scrape deepmind page
                    site_articles = self.__parse_website_for_article_links(
                        soup,
                        self.DEEPMIND_URL,
                        source="deepmind",
                        article_class="glue-card card",
                        title_class="glue-headline",
                        article_html_tag="a",
                        title_html_tag="p",
                        date_html_tag="time",
                        time_html_attr="datetime",
                        articles_database=articles_database)
                elif self.ANTHROPIC_URL in website_url:
                    # scrape anthropic page
                    site_articles = self.__parse_website_for_article_links(
                        soup,
                        self.ANTHROPIC_URL,
                        source="anthropic",
                        article_class="PostCard_post-card__z_Sqq",
                        title_class="PostCard_post-heading__Ob1pu",
                        article_html_tag="a",
                        title_html_tag="h3",
                        date_html_tag="div",
                        time_class="PostList_post-date__djrOA",
                        articles_database=articles_database)
                elif self.OPENAI_URL in website_url:
                    # scrape openai page
                    news_content = soup.find("div",
                                             class_="grid @sm:grid-cols-2 @md:grid-cols-3 gap-x-sm gap-y-2xl")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.OPENAI_URL,
                        source="openai",
                        article_class=None,
                        title_class="text-h5",
                        article_html_tag="a",
                        title_html_tag="div",
                        date_html_tag="time",
                        time_html_attr="datetime",
                        articles_database=articles_database)
                elif self.GOOGLE_RESEARCH_URL in website_url:
                    # scrape google research page
                    news_content = soup.find("ul", class_="blog-posts-grid__cards")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.GOOGLE_RESEARCH_URL,
                        source="google_research",
                        article_class="glue-card",
                        title_class="headline-5",
                        article_html_tag="a",
                        title_html_tag="span",
                        date_html_tag="p",
                        time_class="glue-label",
                        articles_database=articles_database)
                elif self.HUGGINGFACE_URL in website_url:
                    # scrape huggingface page
                    news_content = soup.find("div", class_="col-span-1")
                    classes_to_find = ["flex", "flex-col"]
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.HUGGINGFACE_URL,
                        source="huggingface",
                        article_class=lambda class_list: all(
                            cls in class_list for cls in
                            classes_to_find),
                        title_class="font-semibold",
                        article_html_tag="a",
                        title_html_tag="h2",
                        date_html_tag="span",
                        time_class="truncate",
                        articles_database=articles_database)
                elif self.VERGE_URL in website_url:
                    # scrape huggingface page
                    site_articles = self.__parse_website_for_article_links(
                        soup,
                        self.VERGE_URL,
                        source="theverge",
                        article_class="_1lkmsmo1",
                        title_class="_1lkmsmo1",
                        article_html_tag="a",
                        title_html_tag="a",
                        date_html_tag="time",
                        time_html_attr="datetime",
                        article_wrapper_html_tag="div",
                        article_wrapper_class=["_184mfto4", "_1pm20r51", "_1dqvz267", "_1dqvz265"],
                        articles_database=articles_database)
                elif self.WIRED_URL in website_url:
                    # scrape wired page
                    site_articles = self.__parse_website_for_article_links(
                        soup,
                        self.WIRED_URL,
                        source="wired",
                        article_class="summary-item__hed-link",
                        title_class="summary-item__hed",
                        article_html_tag="a",
                        title_html_tag="h3",
                        date_html_tag="time",
                        time_class="summary-item__publish-date",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="summary-item__content",
                        articles_database=articles_database)
                elif self.VENTUREBEAT_URL in website_url:
                    # scrape wired page
                    news_content = soup.find("div", class_="story-river")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.VENTUREBEAT_URL,
                        source="venturebeat",
                        article_class="ArticleListing__title-link",
                        title_class="ArticleListing__title-link",
                        article_html_tag="a",
                        title_html_tag="a",
                        date_html_tag="time",
                        time_html_attr="datetime",
                        article_wrapper_html_tag="article",
                        article_wrapper_class="ArticleListing",
                        articles_database=articles_database)
                elif self.TECHCRUNCH_URL in website_url:
                    # scrape techcrunch page
                    site_articles = self.__parse_website_for_article_links(
                        soup,
                        self.TECHCRUNCH_URL,
                        source="techcrunch",
                        article_class="loop-card__title-link",
                        title_class="loop-card__title-link",
                        article_html_tag="a",
                        title_html_tag="a",
                        date_html_tag="time",
                        time_html_attr="datetime",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="loop-card__content",
                        articles_database=articles_database)
                elif self.AIBUSINESS_URL in website_url:
                    # scrape aibusiness page
                    news_content = soup.find("div", class_="LatestFeatured-ColumnList")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.AIBUSINESS_URL,
                        source="aibusiness",
                        article_class="ListPreview-Title",
                        title_class="ListPreview-Title",
                        article_html_tag="a",
                        title_html_tag="a",
                        date_html_tag="span",
                        time_class="ListPreview-Date",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="ListPreview-ContentWrapper",
                        articles_database=articles_database)
                elif self.ILPOST_URL in website_url:
                    news_content = soup.find("div", class_="index_home-left__ikJqd")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.ILPOST_URL,
                        source="ilpost",
                        article_class=None,
                        title_class="_article-title_vvjfb_7",
                        article_html_tag="a",
                        title_html_tag="h2",
                        date_html_tag="time",
                        time_class="_taxonomy-item__time_1moex_37",
                        article_wrapper_html_tag="article",
                        article_wrapper_class="_taxonomy-item_1moex_1",
                        articles_database=articles_database)
                elif self.MISTRAL_URL in website_url:
                    # scrape mistral page
                    news_content = soup.find("div", id="news-section")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.MISTRAL_URL,
                        source="mistral",
                        article_class=None,
                        title_class=None,
                        article_html_tag="a",
                        title_html_tag="h3",
                        date_html_tag="span",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="blog-fade-in",
                        time_tag_index=1,
                        articles_database=articles_database)
                elif self.PERPLEXITY_URL in website_url:
                    # scrape perplexity page
                    header_article = soup.find("div", class_="framer-1qu7j16-container")
                    header_link = header_article.find("a", class_="framer-text")
                    header_text = header_link.get_text(strip=True)
                    header_link = header_link.get("href")

                    url = self.PERPLEXITY_URL + header_link[1:]
                    if not articles_database.is_article_already_fetched(url):
                        articles.append({
                            'title': header_text,
                            'publish_date': None,
                            'url': url,
                            'source_url': self.PERPLEXITY_URL,
                            'source': "perplexity_ai"
                        })

                    news_content = soup.find("div", class_="framer-1pk4ise")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.PERPLEXITY_URL,
                        source="perplexity_ai",
                        article_class="framer-fkCik",
                        title_class="framer-text",
                        article_html_tag="a",
                        title_html_tag="h4",
                        date_html_tag="p",
                        time_class="framer-text",
                        articles_database=articles_database)
                elif self.XAI_URL in website_url:
                    # scrape xai page
                    news_content = soup.find("div", class_="sm:gap-6")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.XAI_URL,
                        source="xai",
                        article_class=None,
                        title_class="text-lg",
                        article_html_tag="a",
                        title_html_tag="h4",
                        date_html_tag="span",
                        time_class="mono-tag",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="flex-col",
                        articles_database=articles_database)
                elif self.META_AI_URL in website_url:
                    # scrape meta ai page
                    # extract heading article
                    header_article = soup.find("div", class_="_amc_")
                    header_link = header_article.find("a", class_="_amcw _amd2")
                    header_text = header_link.get_text(strip=True)
                    header_link = header_link.get("href")
                    publish_date = header_article.find("div", class_="_amun")

                    date_str = publish_date.get_text(strip=True)
                    publish_date = datetime.strptime(date_str, "%B %d, %Y").date()
                    publish_date = publish_date.strftime("%Y-%m-%d")

                    url = header_link
                    if not articles_database.is_article_already_fetched(url):
                        articles.append({
                            'title': header_text,
                            'publish_date': publish_date,
                            'url': url,
                            'source_url': self.META_AI_URL,
                            'source': "meta_ai"
                        })

                    news_content = soup.find("div", class_="_amd6")
                    site_articles = self.__parse_website_for_article_links(
                        news_content,
                        self.META_AI_URL,
                        source="meta_ai",
                        article_class="_amcw _amdf",
                        title_class="_amcw _amdf",
                        article_html_tag="a",
                        title_html_tag="a",
                        date_html_tag="div",
                        time_class="_amdj",
                        article_wrapper_html_tag="div",
                        article_wrapper_class="_amdc",
                        articles_database=articles_database,
                        time_tag_index=1
                    )

                elif self.ARXIV_URL in website_url:
                    # scrape arxiv page
                    site_articles = self.__search_arxiv(soup, source_url=website_url,
                                                        articles_database=articles_database)

                articles.extend(site_articles)

            # close the browser
            browser.close()

        return articles

    def __parse_website_for_article_links(self, soup: BeautifulSoup, source_url: str,
                                          articles_database: ArticlesDatabase,
                                          article_class: Optional[Union[str, list, callable]],
                                          title_class: Optional[Union[str, list, callable]],
                                          source: str,
                                          article_html_tag: str = "a",
                                          title_html_tag: Optional[Union[str, callable]] = "p",
                                          date_html_tag: str = "time",
                                          time_class: str = None,
                                          time_html_attr: str = None,
                                          article_wrapper_class: Optional[Union[str, list]] = None,
                                          article_wrapper_html_tag: str = None,
                                          time_tag_index: int = 0) -> list:
        articles = []

        # Look for articles in the page
        article_tag_to_extract = article_wrapper_html_tag if article_wrapper_html_tag else article_html_tag
        article_class_to_extract = article_wrapper_class if article_wrapper_class else article_class

        if article_class_to_extract:
            # Handle the case when article_class_to_extract is a list of classes
            if isinstance(article_class_to_extract, list):
                # Find all elements with the specified tag
                all_elements = soup.find_all(article_tag_to_extract)
                # Filter elements that have any of the classes in the list
                articles_in_site = []
                for element in all_elements:
                    if element.get('class'):
                        # Check if any class in the element matches any class in article_class_to_extract
                        if any(cls in element.get('class') for cls in article_class_to_extract):
                            articles_in_site.append(element)
            else:
                # Original behavior for a single class
                articles_in_site = soup.find_all(article_tag_to_extract, class_=article_class_to_extract)
        else:
            articles_in_site = soup.find_all(article_tag_to_extract)

        articles_to_extract = min(len(articles_in_site), self.max_articles_per_site)

        print(f"Extracting {articles_to_extract} articles from {source_url}...")

        for article_tag in articles_in_site[:articles_to_extract]:
            # extract the article link
            if article_wrapper_html_tag:
                if article_class:
                    # Handle article_class being a list
                    if isinstance(article_class, list):
                        article_link = None
                        for cls in article_class:
                            article_link = article_tag.find(article_html_tag, class_=cls)
                            if article_link:
                                break
                    else:
                        article_link = article_tag.find(article_html_tag, class_=article_class)
                else:
                    article_link = article_tag.find(article_html_tag)

                if article_link and 'href' in article_link.attrs:
                    link_url = article_link['href']
                else:
                    # Skip this article if link can't be found
                    continue
            else:
                if 'href' in article_tag.attrs:
                    link_url = article_tag['href']
                else:
                    # Skip this article if it doesn't have an href attribute
                    continue

            # append url if the link doesn't start with https
            if not link_url.startswith('https'):
                if link_url.startswith("."):
                    link_url = link_url[1:]
                elif not link_url.startswith("/"):
                    link_url = f"/{link_url}"
                link_url = source_url + link_url

            # extract the title of the link
            title = None
            if title_class:
                # Handle title_class being a list
                if isinstance(title_class, list):
                    title_tag = None
                    for cls in title_class:
                        title_tag = article_tag.find(title_html_tag, class_=cls)
                        if title_tag:
                            break
                else:
                    title_tag = article_tag.find(title_html_tag, class_=title_class)
            else:
                title_tag = article_tag.find(title_html_tag)

            if title_tag:
                title = title_tag.get_text(strip=True)

            # find the date of the publication
            if time_class:
                time_tag = article_tag.find_all(date_html_tag, class_=time_class)
            else:
                time_tag = article_tag.find_all(date_html_tag)

            publish_date = None
            if time_tag and time_tag_index < len(time_tag):
                time_tag = time_tag[time_tag_index]
                previous_time_class = time_class
                if time_tag_index > 0:
                    time_class = "mistral"

                if time_tag:
                    if not time_class:
                        if time_html_attr and time_html_attr in time_tag.attrs:
                            date_str = time_tag[time_html_attr]

                            # Remove time portion from date_str if present
                            time_index = date_str.find("T")
                            if time_index != -1:
                                date_str = date_str[:time_index]

                            publish_date = date_str
                    else:
                        date_str = time_tag.get_text(strip=True)
                        formats = ["%B %d, %Y", "%b %d, %Y", "%m.%d.%Y",
                                   "%d/%m/%Y"]  # Full month and abbreviated month formats

                        for fmt in formats:
                            try:
                                publish_date = datetime.strptime(date_str, fmt).date()
                                publish_date = publish_date.strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                days_match = re.search(r"\d+ giorni fa", date_str)
                                if days_match:
                                    days_str = days_match.group(0)
                                    days = int(re.search(r"\d+", days_str).group(0))
                                    publish_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                                    break
                                else:
                                    continue

                if time_class == "mistral":
                    time_class = previous_time_class

            if not articles_database.is_article_already_fetched(link_url):
                articles.append(
                    {
                        'title': title,
                        'publish_date': publish_date,
                        'url': link_url,
                        'source_url': source_url,
                        'source': source
                    }
                )

        return articles

    def __search_arxiv(self, soup: BeautifulSoup, source_url: str, articles_database: ArticlesDatabase) -> list:
        # find all articles sections
        articles = []
        news_sections = soup.find_all("dl", id="articles")

        for news_section in news_sections:
            # extract the date
            date = news_section.find("h3").text.strip()
            date_match = re.match(r"(\w{3}, \d{1,2} \w{3} \d{4})", date)
            date = None
            if date_match:
                date = datetime.strptime(date_match.group(1), "%a, %d %b %Y").date()
                date = date.strftime("%Y-%m-%d")

            # find all articles
            dt_elements = soup.find_all('dt')

            articles_to_extract = min(len(dt_elements), self.max_articles_per_site)

            print(f"Extracting {articles_to_extract} articles from {source_url}...")

            for dt in dt_elements[:articles_to_extract]:
                # extract the link element a with title "Abstract"
                link = dt.find('a', title="Abstract")
                link_url = None
                if link:
                    link_url = link['href']
                    if not link_url.startswith('https'):
                        link_url = self.ARXIV_URL + link_url

                # extract the link element a with title "Download PDF"
                pdf_link = dt.find('a', title="Download PDF")
                pdf_link_url = None
                if pdf_link:
                    pdf_link_url = pdf_link['href']
                    if not pdf_link_url.startswith('https'):
                        pdf_link_url = self.ARXIV_URL + pdf_link_url

                # Find the next dd element after this dt
                dd = dt.find_next_sibling('dd')
                title = None
                authors = []
                if dd:
                    # extract the title
                    title = dd.find("div", class_="list-title").get_text(strip=True)[6:]

                    # extract authors
                    authors_section = dd.find("div", class_="list-authors")
                    authors = [author.get_text(strip=True) for author in authors_section.find_all("a")]

                if not articles_database.is_article_already_fetched(link_url):
                    articles.append(
                        {
                            'title': title,
                            'authors': authors,
                            'publish_date': date,
                            'url': link_url,
                            'source_url': source_url,
                            'source': "arxiv",
                            'pdf_url': pdf_link_url
                        }
                    )

        return articles

    def fetch_articles_from_sources(self, articles_database: ArticlesDatabase) -> List[Dict[str, Any]]:
        """
        Fetches articles about the topic from all sources.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing the article data. Each dictionary contains the article title, text,
            URL, authors, publish date, keywords, summary, source, and fetch date.
        """

        all_articles = []
        print(f"Fetching articles about from all sources...")

        # Perform additional scraping of websites if we don't have enough articles yet
        website_articles = self.__scrape_websites(articles_database)
        all_articles.extend(website_articles)
        print(f"Found {len(website_articles)} additional articles from direct website scraping")

        # Remove any duplicates based on URL
        unique_articles = []
        seen_urls = set()

        # filter out unique articles by removing duplicates
        for article in all_articles:
            url = article.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

                # scrape the article
                article_text = self.__scrape_article(article)

                # set the text of the article to current article object
                article["text"] = article_text

        print(f"Total unique articles found: {len(unique_articles)}")

        return unique_articles

    def __scrape_article(self, article: dict):
        # initialize the browser for scraping
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Run in headful mode to appear more human-like
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )

            # scrape the article
            article_url = article.get("url", "")
            print(f"Scraping article: {article_url}")

            # scrape the website with playwright
            page = context.new_page()

            page.goto(article_url, timeout=self.search_timeout)
            page.wait_for_timeout(self.wait_time)
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')

            if self.ILPOST_URL in article_url:
                # scrape ilpost article
                article_text = self.__parse_article(soup, content_tag="div", content_class="contenuto")
            elif self.AIBUSINESS_URL in article_url:
                # scrape aibusiness article
                article_text = self.__parse_article(soup, content_tag="div", content_class="ArticleBase-BodyContent")
            elif self.TECHCRUNCH_URL in article_url:
                # scrape techcrunch article
                article_text = self.__parse_article(soup, content_tag="div", content_class="wp-block-post-content")
            elif self.VENTUREBEAT_URL in article_url:
                # scrape venturebeat article
                article_text = self.__parse_article(soup, content_tag="div", content_class="article-content")
            elif self.WIRED_URL in article_url:
                # scrape wired article
                article_text = self.__parse_article(soup, content_tag="div", content_class="ArticlePageChunks-fLyCVG")
            elif self.VERGE_URL in article_url:
                # scrape verge article
                article_text = self.__parse_article(soup, content_tag="div", content_class="duet--layout--entry-body")
            elif self.HUGGINGFACE_URL in article_url:
                # scrape huggingface article
                article_text = self.__parse_article(soup, content_tag="div", content_class="blog-content")
            elif self.GOOGLE_RESEARCH_URL in article_url:
                # scrape google research article
                article_text = self.__parse_article(soup, content_tag="div", content_class="glue-grid__col")
            elif self.OPENAI_URL in article_url:
                # scrape openai article
                article_text = self.__parse_article(soup, content_tag="article", content_class="flex flex-col")
            elif self.ANTHROPIC_URL in article_url:
                # scrape anthropic article
                article_text = self.__parse_article(soup, content_tag="div", content_class="Body_body__XEXq7")
            elif self.DEEPMIND_URL in article_url:
                # scrape deepmind article
                article_text = self.__parse_article(soup, content_tag="div", content_class="glue-page")
            elif self.GOOGLE_BLOG_URL in article_url:
                # scrape google blog article
                article_text = self.__parse_article(soup, content_tag="article", content_class="uni-article-wrapper")
            elif self.GOOGLE_DEVELOPERS_BLOG_URL in article_url:
                # scrape google developers blog article
                article_text = self.__parse_article(soup, content_tag="div", content_class="blog-detail-container")
            elif self.MISTRAL_URL in article_url:
                # scrape mistral article
                article_text = self.__parse_article(soup, content_tag="div", content_class="blog-rich-text")
            elif self.PERPLEXITY_URL in article_url:
                # scrape perplexity article
                article_text = self.__parse_article(soup, content_tag="div", content_class="framer-tef8j0")
            elif self.META_AI_URL in article_url:
                # scrape meta ai article
                article_text = self.__parse_article(soup, content_tag="div", content_class="_7h8s")
            elif self.XAI_URL in article_url:
                # scrape xai article
                article_text = self.__parse_article(soup, content_tag="section", content_class="py-16")
            elif self.ARXIV_URL in article_url:
                # scrape arxiv article
                article_text = self.__parse_article(soup, content_tag="blockquote", content_class="abstract")

            # close the browser
            browser.close()

        return article_text

    def __parse_article(self, soup: BeautifulSoup, content_tag: str, content_class: str = None):
        if content_class:
            content = soup.find(content_tag, class_=content_class)
        else:
            content = soup.find(content_tag)

        if content:
            text = content.get_text(separator=" ", strip=True)
            text = text.replace("\xa0", " ").replace("\n", " ").replace("\'", "'")
        else:
            text = None

        return text
