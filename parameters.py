# GPT related
MODEL = "gpt-3.5-turbo-0301"
accuracy_temperatures_map = {"low": 1, "medium": 0.5, "high": 0.2}
system_prompts = {
    "personal assistant": "I want you to act as a personal assistant. I will provide you with questions related to daily lives.",
    "web app software architecture": " I want you to act as a software developer. I will provide some specific information about a web app requirements, and it will be your job to come up with an architecture and code for developing secure app.",
    "web app software developer": "I want you to act as a software developer. I will provide some specific information about a web app requirements, and it will be your job to write key part of codes",
}

# transcription related
HISTORY_MAX_LENGTH = 10
HISTORY_MAX_TEXT = 100
TRANSCRIBE_TIMEOUT_LENGTH = 1.5
INITIAL_TIMEOUT_LENGTH = 3600
# synthesise related
DELIMITERS = ["。", "！", "？", "，", "。", ",", ".", "\n"]
# self.delimiters = "[！？。 . \n]"
TEXT_RECEIVE_TIMEOUT_LENGTH = 0.5
SYNTHESIS_TIMEOUT_LENGTH = 0.5
LANGUAGE_VOICE_MAP = {"en-GB": "en-US-NancyNeural", "zh-CN": "zh-CN-XiaochenNeural"}

EXPECTED_LANGUAGE = ["en-GB", "zh-CN"]
MP3_SENDING_CHUNK_SIZE = 2048
MP3_SENDING_TIMEOUT_LENGTH = 5
