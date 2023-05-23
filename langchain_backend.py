# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import asyncio
import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pinecone
import tiktoken
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (ConversationalRetrievalChain, ConversationChain,
                              LLMChain, RetrievalQA, SequentialChain)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import (ChatMessageHistory, CombinedMemory,
                              ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              ConversationSummaryBufferMemory,
                              ConversationSummaryMemory, SimpleMemory)
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate, PromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, LLMResult, SystemMessage
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter,
                                     TokenTextSplitter)
from langchain.vectorstores import FAISS, Chroma, Pinecone
from pydantic import BaseModel, validator

from parameters import (ACCURACY_TEMPERATURE_MAP, HISTORY_MAX_LENGTH,
                        HISTORY_MAX_TEXT, MODEL, system_prompts)

load_dotenv()
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    async def on_chat_model_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("zzzz....")
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


def create_chat(
    accuracy: str = "medium",
    stream: bool = True,
    model: str = "gpt-3.5-turbo",  # Replace with your model
    session_id: Optional[str] = None,
) -> str:
    chat = ChatOpenAI(
        model_name=model,
        temperature=ACCURACY_TEMPERATURE_MAP[accuracy],
        streaming=stream,
        # callbacks=[MyCustomAsyncHandler()],
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    return chat


def load_document_from_github_repo(folder_path: Path, filetype: str = ".py"):
    if not folder_path.is_dir():
        raise ValueError("folder_path must be a directory")
    documents = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            if file.endswith(filetype) and "/.venv/" not in dirpath:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    documents.extend(loader.load_and_split())
                except Exception as e:
                    pass
    return documents


def create_pinecone_index(texts, embeddings, index_name):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    active_indexes = pinecone.list_indexes()
    if index_name not in active_indexes:
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            pods=1,
            replicas=1,
            pod_type="Starter",
        )
    db = Pinecone.from_documents(texts, embeddings, index_name=index_name)


def use_vector_store():
    chat = create_chat(model="gpt-4")
    # loader = TextLoader("resources\example.txt", encoding="utf8")
    # loader = PyPDFLoader("resources\gpt4_explain.pdf")
    # loader = BSHTMLLoader("resources\gpt4_explain.html", open_encoding="utf8")
    # documents = loader.load()
    # documents = load_document_from_github_repo(
    #     Path("resources\langchain"), filetype=".py"
    # )

    # to load a vector store
    db = Chroma(
        persist_directory="resources\chroma", embedding_function=OpenAIEmbeddings()
    )
    db.persist()

    # do a similarity search on the database - returns a list of documents that are similar to a given query
    # query = "how to stream chat using callbacks??"
    # docs = db.similarity_search(query)
    # print(docs)

    # actually use it in a chain as data source
    # https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html
    # chat over documents
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    response = chain(
        {
            "question": "how to stream chat using callbacks?",
        }
    )

    print(response)

    print()
    # print(response["answer"])
    response = chain(
        {
            "question": "And how do I stop the stream prematuraly?",
        }
    )


def use_memory():
    chat = create_chat()
    # """
    # Recent conversations:
    # {chat_history_lines}
    # """
    conv_window_memory = ConversationBufferWindowMemory(
        memory_key="chat_history_lines",
        input_key="input",
        k=1,
        return_messages=True,
    )

    # """
    # summary of conversation:
    # {summary}
    # """
    summary_memory = ConversationSummaryMemory(
        llm=chat, input_key="input", memory_key="summary"
    )

    combined_memory = CombinedMemory(memories=[conv_window_memory, summary_memory])

    #  combine the summary and window memory
    conv_summary_buffer = ConversationSummaryBufferMemory(
        llm=chat, max_token_limit=40, memory_key="summary"
    )

    # the memory is inserted using the system message prompt by memory_key
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant that answers user's question. Keep the answer precise.
    {summary}
    """
    )

    human_message_prompt_template = HumanMessagePromptTemplate.from_template(
        """Human: {input}\nAI: """
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template]
    )

    # resp = chat(chat_prompt.format_prompt(text="I love programming?").to_messages())
    chain = ConversationChain(
        llm=chat,
        prompt=chat_prompt_template,
        verbose=True,
        memory=conv_summary_buffer,
    )
    # additional memory that is not used in the chain but can be used to save the context
    backup_memory = ConversationBufferMemory()

    response = chain({"input": "Who discovered whispering gallery effect?"})
    backup_memory.save_context(
        {"input": response["input"]}, {"outputs": response["response"]}
    )
    print(response)
    response = chain({"input": "And next 4?"})
    backup_memory.save_context(
        {"input": response["input"]}, {"outputs": response["response"]}
    )
    print(response)
    response = chain({"input": "Pink?"})
    print(response)


def use_agent():
    chat = create_chat()
    # https://www.youtube.com/watch?v=KerHlb8nuVc
    tools = load_tools(["google-search", "python_repl"], llm=chat, max_iterations=3)
    agent = initialize_agent(
        tools,
        llm=chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    response = agent.run("BItcoin price in rmb and usd for 2023 May 17th ?")
    print(response)


# use_agent()
use_vector_store()


# chain = LLMChain(llm=chat, memory=memory)
# chain.run(prompt="I love programming.")

# if __name__ == "__main__":
#     chat = create_chat()

#     async def main():
#         # Start a task for chat.agenerate()
#         await chat.agenerate([[HumanMessage(content="Tell me a joke about icecream")]])
#         # Wait for a while
#         await asyncio.sleep(2)

#     # Run the main function
#     asyncio.run(main())
#     # Here you would stop your chat if the stop() method is available.'
