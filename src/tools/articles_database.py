from src.config import Config
import os
import pandas as pd
import requests
import string
import json
import fitz


class ArticlesDatabase:
    DEFAULT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    def __init__(self, config: Config):
        self.config = config

        # Extract websites from config
        websites = self.config.get("search.web.sites", None)
        if not websites:
            raise ValueError("No websites found in config.")

        self.websites = websites

        # the folder path
        self.data_folder = self.config.get("db.folder_path", self.DEFAULT_FOLDER_PATH)

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # then create a "docs" folder inside the data folder if it doesn't exist
        self.docs_folder = os.path.join(self.data_folder, "docs")

        if not os.path.exists(self.docs_folder):
            os.makedirs(self.docs_folder)

        # finally, create an empty .csv file named "articles.csv" inside the data folder if it doesn't exist
        self.articles_records_path = os.path.join(self.data_folder, "articles.csv")
        if not os.path.exists(self.articles_records_path):
            with open(self.articles_records_path, "w") as f:
                f.write("url,article_file_path,title,source,publish_date,has_text,is_new\n")

        self.articles_records = pd.read_csv(self.articles_records_path)

    def update(self, articles: list):
        # update all the rows in the articles records by setting is_new to False
        self.articles_records.loc[:, "is_new"] = False

        new_articles = 0
        for article in articles:
            url = article.get("url", "")

            # skip the article if it doesn't have an url
            if not url:
                continue

            # skip the article if it has already been fetched
            if self.is_article_already_fetched(url):
                continue

            # download the pdf of the article if it exists
            pdf_url = article.get("pdf_url", "")
            if pdf_url:
                article_file_path = self.__download_pdf_from_url(article)
            else:
                # otherwise, save the article content to a pdf
                article_file_path = self.__save_article_to_txt(article)

            # extract the title
            title = article.get("title", "")

            # extract the source
            source = article.get("source", "")

            # extract the publish date
            publish_date = article.get("publish_date", "")

            # extract the text
            text = article.get("text", "")
            has_text = True if text else False

            # now save the article in the articles_record
            self.articles_records.loc[len(self.articles_records)] = [url, article_file_path, title, source,
                                                                     publish_date, has_text, True]

            new_articles += 1

        # update articles record files
        self.articles_records.to_csv(self.articles_records_path, index=False)

        if new_articles > 0:
            print(f"Articles database updated with {new_articles} new article{"s" if new_articles > 1 else ""}.")
        else:
            print("Articles database already up to date.")

    def is_article_already_fetched(self, url):
        return url in self.articles_records["url"].values

    def get_new_articles(self):
        return "articles_list", self.articles_records[self.articles_records["is_new"]].values.tolist()

    def get_all_articles(self):
        return "articles_list", self.articles_records.values.tolist()

    def __download_pdf_from_url(self, article):
        # download the pdf
        url = article.get("pdf_url", "")
        article_name = article.get("title", "")
        article_source = article.get("source", "")
        article_date = article.get("publish_date", "")
        filename = f"{article_name}_{article_source}_{article_date}"
        filename = self.__safe_filename(filename).lower()
        pdf_filename = f"{filename}.pdf"
        txt_filename = f"{filename}.txt"
        pdf_file_path = os.path.join(self.docs_folder, pdf_filename)
        txt_file_path = os.path.join(self.docs_folder, txt_filename)

        if not os.path.exists(pdf_file_path):
            response = requests.get(url)
            with open(pdf_file_path, "wb") as f:
                f.write(response.content)

        # convert pdf to txt
        self.__pdf_to_txt(pdf_file_path, txt_file_path)

        # delete the pdf file
        os.remove(pdf_file_path)

        return txt_file_path

    @staticmethod
    def __pdf_to_txt(pdf_path, txt_path):
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    def __save_article_to_txt(self, article):
        # create the file path
        article_name = article.get("title", "")
        article_source = article.get("source", "")
        article_date = article.get("publish_date", "")
        filename = f"{article_name}_{article_source}_{article_date}.txt"
        filename = self.__safe_filename(filename).lower()
        txt_file_path = os.path.join(self.docs_folder, filename)

        # save the article content to the txt file
        with open(txt_file_path, 'w') as f:
            json.dump(article, f, ensure_ascii=False)

        return txt_file_path

    @staticmethod
    def __safe_filename(filename):
        # Handle common OS reserved characters
        invalid_chars = '<>:"/\\|?*'

        # Replace invalid characters with empty string
        for char in invalid_chars:
            filename = filename.replace(char, '')

        # Remove control characters
        filename = ''.join(c for c in filename if c in string.printable)

        # Replace spaces and hyphens with underscores
        filename = filename.replace(' ', '_').replace('-', '_')

        # Handle Windows reserved filenames
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL',
                          'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                          'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

        base_name = os.path.splitext(filename)[0].upper()
        if base_name in reserved_names:
            if '.' in filename:
                name, ext = os.path.splitext(filename)
                filename = name + '_' + ext
            else:
                filename = filename + '_'

        # Ensure filename doesn't begin or end with a space or period
        filename = filename.strip(' .')

        # If empty filename, provide a default
        if not filename:
            filename = "unnamed_file"

        return filename
