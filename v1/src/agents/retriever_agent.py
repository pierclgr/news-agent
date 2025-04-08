from pathlib import Path
from typing import Optional, List

from llama_index.core.agent.workflow import ReActAgent
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
import asyncio
import glob
import os
from pydantic import Field


class RetrieverAgent(ReActAgent):
    count_search_calls: int = Field(default=0)
    max_search_calls: int = Field(default=10)

    def __init__(self,
                 docs_folder: str,
                 name: str = "RetrieverAgent",
                 description: str = ("A research agent that searches and retrieves documents and extract information "
                                     "from a knowledge base. It must not exceed 2 searches total, and must avoid "
                                     "repeating the same query. The user will input a query or a prompt specifying "
                                     "what document/documents and information to look for. Once sufficient information "
                                     "is collected, it should hand off to the WriteAgent.",
                                     ),
                 system_prompt: str = ("You are the RetrieverAgent, a research agent that has the capabilities to "
                                       "search and retrieve documents and extract information from a knowledge base. "
                                       "Your goal is to gather the information requested by the user from the "
                                       "documents in the knowledge base. Only perform at most 2 distinct searches. If "
                                       "you have enough info or have reached 2 searches, handoff to the next agent. "
                                       "Avoid infinite loops!"),
                 tokenizer_embedding_model: str = "BAAI/bge-small-en-v1.5",
                 model: str = "qwen2.5-7b-instruct-1m",
                 timeout: int = 120,
                 api_base: str = "http://localhost:1234/v1",
                 verbose: bool = True,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200,
                 can_handoff_to: Optional[List[str]] = None):

        if not can_handoff_to:
            can_handoff_to = ["WriteAgent"]

        # create a vector store for our documents
        docs_collection = chromadb.PersistentClient(path=docs_folder)
        chroma_collection = docs_collection.get_or_create_collection("articles_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # create the llm to use
        llm = LMStudio(
            model_name=model,
            base_url=api_base,
            timeout=timeout
        )

        # create the embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=tokenizer_embedding_model
        )
        Settings.embed_model = embed_model

        # extract the documents in the docs folder
        documents = [f for f in glob.glob(docs_folder + '/*') if
                     os.path.isfile(f) and f.endswith('.pdf') or f.endswith(".txt")]

        # create tools for each document
        documents_to_tool_dict = {}
        for document in documents:
            if verbose:
                print(f"Getting tools for document: {document}")

            # get the tool for the current document
            vector_tool, summary_tool = self.__get_doc_tools(document, chunk_size, chunk_overlap, storage_context)

            # add the tools to the dictionary
            documents_to_tool_dict[document] = [vector_tool, summary_tool]

        # extract all tools of the documents
        all_tools = [t for document in documents for t in documents_to_tool_dict[document]]

        obj_index = ObjectIndex.from_objects(
            all_tools,
            index_cls=VectorStoreIndex,
        )

        # create document retriever tool
        obj_retriever = obj_index.as_retriever(similarity_top_k=3)

        # initialize the agent
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tool_retriever=obj_retriever,
            llm=llm,
            verbose=verbose,
            can_handoff_to=can_handoff_to
        )

    @staticmethod
    def __get_doc_tools(file_path: str, chunk_size: int, chunk_overlap: int, storage_context: StorageContext):
        # load document
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # split the document in nodes
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        # create a vector store index for the document
        vector_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        # create a summary index for the document
        summary_index = SummaryIndex(nodes)

        # extract the document name
        document_name = Path(file_path).stem.lower()

        # define vector query function
        def __vector_query(query: str, page_numbers: Optional[List[str]] = None):
            page_numbers = page_numbers or []
            metadata_dicts = [
                {"key": "page_label", "value": p} for p in page_numbers
            ]

            query_engine = vector_index.as_query_engine(
                similarity_top_k=2,
                filters=MetadataFilters.from_dicts(
                    metadata_dicts, condition=FilterCondition.OR
                ),
            )
            response = query_engine.query(query)
            return response

        # define vector query tool
        vector_query_tool = FunctionTool.from_defaults(
            name=f"vector_tool_{document_name}",
            description=f"A tool to query the document {document_name}",
            fn=__vector_query
        )

        # define the function to create a summary tool
        def __summary_query(query: str):
            summary_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )

            response = summary_engine.query(query)
            return response

        # define the summary tool
        summary_tool = FunctionTool.from_defaults(
            fn=__summary_query,
            name=f"summary_tool_{document_name}",
            description=f"A tool to summarize the document {document_name}"
        )

        return vector_query_tool, summary_tool

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
