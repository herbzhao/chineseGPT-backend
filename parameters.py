accuracy_temperatures_map = {"low": 1, "medium": 0.5, "high": 0.2}
HISTORY_MAX_LENGTH = 10
HISTORY_MAX_TEXT = 100
LANGUAGE_DETECT_CONFIDENCE = 0.5
SENTENCE_BREAK_DURATION = 2000
SILENCE_THRESHOLD = -30
SILENCE_SPLIT_DURATION = 300
EXPECTED_LANGUAGE = ["en", "zh"]
MODEL = "gpt-3.5-turbo-0301"
system_prompts = {
    "personal assistant": "I want you to act as a personal assistant. I will provide you with questions related to daily lives.",
    "web app software architecture": " I want you to act as a software developer. I will provide some specific information about a web app requirements, and it will be your job to come up with an architecture and code for developing secure app.",
    "web app software developer": "I want you to act as a software developer. I will provide some specific information about a web app requirements, and it will be your job to write key part of codes",
}
