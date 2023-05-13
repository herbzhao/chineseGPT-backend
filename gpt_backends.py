# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import datetime
import os
from pathlib import Path
from typing import List, Optional

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import openai
import tiktoken
from dotenv import load_dotenv

from parameters import (
    HISTORY_MAX_LENGTH,
    HISTORY_MAX_TEXT,
    MODEL,
    accuracy_temperatures_map,
    system_prompts,
)

load_dotenv()
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(
    prompt: str,
    history: Optional[List[dict]],
    actor: str = "personal assistant",
    max_tokens: int = 1024,
    accuracy: str = "medium",
    stream: bool = False,
    model: str = MODEL,
    session_id: Optional[str] = None,
) -> str:
    """This function is used to send a user input to the OpenAI API and return a response.

    This function is used to send a user input to the OpenAI API and return a response.
    The function takes a prompt, a list of previous messages, and some additional parameters
    as input and returns the response from the API.

    Args:
        prompt (str): The user input to send to the API
        history (list): A list of previous messages to send to the API
        actor (str): The name of the actor who is having the conversation with the user
        max_tokens (int): The maximum number of tokens to return from the API
        accuracy (str): The accuracy of the response to return from the API

    Returns:
        str: The response from the API
    """
    system_prompt = system_prompts[actor]
    if len(history) > 0:
        # shorten the history to HISTORY_MAX_LENGTH messages,
        history = history[-HISTORY_MAX_LENGTH:]
        # also shorten the text of each historic message to HISTORY_MAX_TEXT characters
        for message in history:
            message["content"] = message["content"][-HISTORY_MAX_TEXT:]
            # replace the key "author" with "role", and remove "loading" key
            message["role"] = message.pop("author")
            message.pop("loading")
            message.pop("time")

        # add the system prompt to the history
        prompt_messages = history
        prompt_messages.append({"role": "user", "content": prompt})
    else:
        prompt_messages = [
            # use user as the role as model doesnt use system role yet
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    prompt_token_number = calculate_token_number(prompt_messages)

    time_start = datetime.datetime.now()
    # https://platform.openai.com/docs/api-reference/chat/create
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt_messages,
        temperature=accuracy_temperatures_map[accuracy],
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stream=stream,
    )

    if not stream:
        response["role"] = response.choices[0].message.role
        response["content"] = response.choices[0].message.content
        return response, prompt_token_number

    if stream:
        return response, prompt_token_number


def record_chat_history(
    session_id,
    prompt_messages,
    response_message,
    time_start,
    time_lapsed,
):
    # calculate the number of tokens used for the prompt and the response and total
    num_tokens_prompt = calculate_token_number(prompt_messages)
    num_tokens_response = calculate_token_number([response_message])
    num_tokens_total = num_tokens_prompt + num_tokens_response
    # last message is the response, 2nd last message is the new prompt
    if len(prompt_messages) > 1:
        history_messages = prompt_messages[:-1]
    prompt_message = prompt_messages[-1]

    with open(
        Path("output") / Path(f"reponse_{session_id}.md"),
        "a+",
        encoding="utf-8",
    ) as file:
        file.write("___\n")
        file.write(f"### message time: {time_start}\n")
        file.write(
            f"### Token usage: total: {num_tokens_total}, prompt: {num_tokens_prompt}, response: {num_tokens_response}\n"
        )
        file.write(f"### time: {time_lapsed}\n")
        file.write(
            f"### length of history: {len(prompt_messages)}, history_text_limit: {HISTORY_MAX_TEXT}\n"
        )
        # prompt_messages contains the history of the conversation
        if len(history_messages) > 0:
            file.write("### included history:\n```\n")
            for message in history_messages:
                file.write(f"{message['role']}:   \n")
                file.write(f"content: {message['content']}  \n")
            file.write("```\n")

        file.write("### new prompt:\n```\n")
        file.write(f"{prompt_message['role']}:  \n")
        file.write(f"{prompt_message['content']}  \n")
        file.write("```\n")

        file.write("### response:\n```\n")
        file.write(f"{response_message['role']}:  \n")
        file.write(f"{response_message['content']}  \n")
        file.write("```\n")

        file.write("___\n")


def calculate_token_number(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        # print(
        #     "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        # )
        return calculate_token_number(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print(
        #     "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        # )
        return calculate_token_number(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314" or model == "gpt-4":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def voice_to_text(
    audio_file: str, accuracy: str = "medium", language: Optional[str] = None
) -> str:
    """Converts an audio file to text. Returns a string of the text.

    audio_file : str
        The path to the audio file.
    accuracy : str, optional
        The accuracy of the transcription. Must be one of "high", "medium",
        or "low". Defaults to "medium".
    language : Optional[str], optional
        The language of the audio file. Defaults to None.
    """
    # if audiofile has no name attribute, assign an arbitrary name
    transcript = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
        temperature=accuracy_temperatures_map[accuracy],
        language=language,
    )["text"]

    return transcript


async def voice_to_text_async(
    audio_file: str, accuracy: str = "medium", language: Optional[str] = None
) -> str:
    """Converts an audio file to text. Returns a string of the text.

    audio_file : str
        The path to the audio file.
    accuracy : str, optional
        The accuracy of the transcription. Must be one of "high", "medium",
        or "low". Defaults to "medium".
    language : Optional[str], optional
        The language of the audio file. Defaults to None.
    """
    # if audiofile has no name attribute, assign an arbitrary name
    transcript = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
        temperature=accuracy_temperatures_map[accuracy],
        language=language,
    )["text"]

    return transcript


def chat_langchain(
    prompt: list[str],
    history: Optional[List[dict]],
    actor: str = "personal assistant",
    max_tokens: int = 1024,
    accuracy: str = "medium",
    stream: bool = False,
    model: str = MODEL,
    session_id: Optional[str] = None,
) -> str:
    chat = ChatOpenAI(
        model_name=model,
        temperature=accuracy_temperatures_map[accuracy],
        streaming=stream,
    )
    # chat_result = chat.generate(prompt)
    print(chat_result)
    return chat_result


if __name__ == "__main__":
    example_messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
    ]
    example_messages = [
        [
            SystemMessage(
                content="You are a helpful assistant that translates English to French."
            ),
            HumanMessage(
                content="Translate this sentence from English to French. I love programming."
            ),
        ]
    ]

    prompt = ["hello how are you?"]

    chat_result = chat_langchain(prompt=example_messages, history=[])
    print(chat_result.llm_output)
