# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import asyncio
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


import openai
import tiktoken
from dotenv import load_dotenv
from langchain import (
    OpenAI,
    PromptTemplate,
)
from langchain.chains import SequentialChain, LLMChain, ConversationChain
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    SimpleMemory,
    ConversationSummaryMemory,
    CombinedMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)

from langchain.schema import AIMessage, HumanMessage, LLMResult, SystemMessage
from pydantic import BaseModel, validator
from parameters import (
    HISTORY_MAX_LENGTH,
    HISTORY_MAX_TEXT,
    MODEL,
    ACCURACY_TEMPERATURE_MAP,
    system_prompts,
)

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


chat = create_chat()
conv_window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history_lines", input_key="input", k=1, return_messages=True
)
# """
# Recent conversations:
# {chat_history_lines}
# """

summary_memory = ConversationSummaryMemory(
    llm=chat, input_key="input", memory_key="summary"
)
# """
# summary of conversation:
# {summary}
# """

combined_memory = CombinedMemory(memories=[conv_window_memory, summary_memory])

conv_summary_buffer = ConversationSummaryBufferMemory(
    llm=chat, max_token_limit=40, memory_key="summary"
)
# """
# summary of conversation:
# {summary}
# """


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

backup_memory = ConversationBufferMemory()

response = chain({"input": "First 3 colors of the rainbow?"})
backup_memory.save_context(
    {"input": response["input"]}, {"outputs": response["response"]}
)
print(response)
response = chain({"input": "And next 4?"})
print(response)
response = chain({"input": "Pink?"})
print(response)

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
