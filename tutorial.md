### Audio processing and segmentation

* <https://github.com/jiaaro/pydub/blob/master/API.markdown#silence>

### Azure

  * <https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/speech_sample.py>
  * <https://learn.microsoft.com/en-gb/azure/cognitive-services/speech-service/how-to-use-codec-compressed-audio-input-streams?tabs=windows%2Cdebian%2Cjava-android%2Cterminal&pivots=programming-language-python#example>
* Documentation for the Speech SDK for Python
  * <https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.audio.pushaudioinputstream?view=azure-python>

* For language detection:
  * https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-identification?pivots=programming-language-python&tabs=once#at-start-and-continuous-language-identification
* For streaming without using wave
  * https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/how-to-use-codec-compressed-audio-input-streams?tabs=windows%2Cdebian%2Cjava-android%2Cterminal&pivots=programming-language-python
  
* For continuous streaming
  * https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/513c0d8e4370f47dcf241c0682265c2b2fa37db6/samples/python/console/speech_sample.py#L356
  * https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/513c0d8e4370f47dcf241c0682265c2b2fa37db6/samples/python/console/speech_sample.py#L285
* Async
  * https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/how-to-async-conversation-transcription?pivots=programming-language-csharp


### Threading vs Async
* Thread is for working in parallel, while async is for waiting in parallel, so for waiting for a response from a server, async is better
* thread means you can run multiple of the same function at the same time.
* sync is basically sequential code 
* async means we dont need to wait for a previous step to complete before we can start the next 
* async function returns a coroutine object (not the actual function), hence, you need to use await (to execute the async co-routine). Await also blocks the execution of the code until the async function is completed.
* if everything is done using await without create_task, you are basically  running the code synchronously.
* task: asyncio.create_task() is used to create a task from a coroutine object. This will run out of order, but will not block the execution of the code. If any part of the code is waiting, the task will be executed.

* a event-loop is started by asyncio.run() or asyncio.create_task()

* Async code does not run out of order1. An async method will initially run synchronously until it hits an await keyword2. This is when the asynchronous execution will begin2. Now, everything in the async block runs in order but doesn’t halt program execution to do so1. Anything coming after an await will still run out of order, as it is simply starting the execution of an async function1.