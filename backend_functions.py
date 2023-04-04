# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
from dotenv import load_dotenv
import datetime
from pathlib import Path
from typing import List, Optional
import tiktoken
from parameters import (
    accuracy_temperatures_map,
    system_prompts,
    HISTORY_MAX_LENGTH,
    HISTORY_MAX_TEXT,
    MODEL,
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(
    prompt: str,
    history: Optional[List[dict]],
    actor: str = "personal assistant",
    max_tokens: int = 1024,
    accuracy: str = "medium",
    stream: bool = False,
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
        prompt_messages = history
        prompt_messages.append({"role": "user", "content": prompt})
    else:
        prompt_messages = [
            # use user as the role as model doesnt use system role yet
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    # shorten the history to HISTORY_MAX_LENGTH messages,
    prompt_messages = prompt_messages[-HISTORY_MAX_LENGTH:]
    # also shorten the text of each message up to X characters, except for the last message
    for index, response_message in enumerate(prompt_messages):
        if index < len(prompt_messages) - 1:
            response_message["content"] = response_message["content"][:HISTORY_MAX_TEXT]

    time_start = datetime.datetime.now()
    # https://platform.openai.com/docs/api-reference/chat/create
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_messages,
        temperature=accuracy_temperatures_map[accuracy],
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stream=stream,
    )
    if not stream:
        response_message["role"] = response.choices[0].message.role
        response_message["content"] = response.choices[0].message.content

    if stream:
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_chunk_messages = []
        response_message = {"role": "assistant", "content": []}
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_chunk_messages.append(chunk_message)  # save the message
            if "content" in chunk_message:
                response_message["content"].append(chunk_message.content)
                print(chunk_message.content, end="", flush=True)
            elif "role" in chunk_message:
                response_message["role"] = chunk_message.role
                print(f"\r{chunk_message.role}: ", end="", flush=True)
        print()

        # convert the stream of chunks to a single string
        response_message["content"] = "".join(response_message["content"])

    time_lapsed = datetime.datetime.now() - time_start
    print(f"time: {time_lapsed.seconds}s")

    if session_id is None:
        session_id = time_start.strftime("%Y-%m-%d_%H-%M-%S")

    record_chat_history(
        session_id, prompt_messages, response_message, time_start, time_lapsed
    )

    return response_message


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
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


if __name__ == "__main__":
    prompt = "name 3 animals"
    response = chat(
        prompt=prompt,
        history=[],
        actor="personal assistant",
        max_tokens=200,
        accuracy="medium",
        stream=True,
        session_id="test",
    )
