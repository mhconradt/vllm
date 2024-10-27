## GOAL

Roughly OpenAI-compatible /v1/audio/transcriptions endpoint

Goals:

- file / model: basically just rawdog the transcription for the model used by vllm serve
- language: sets the language token used to indicate the output language, i.e. "en", "ko", "ja", "zh".
- prompt: conditions the output. Could say something like "use カタカナ" or "use ひらがな".
- temperature: temperature to use for auto-regressive decoding

Non-goals:

- response_format
- timestamp_granularities

Components:

- Implement Whisper model (basically copy huggingface transformers impl and remove parts only used for training)
- Implement /v1/audio/transcriptions
- Ensure /v1/chat/completions does something sensible
- Implement audio multi-modal processing (maximum 1 audio per prompt, maximum 30 seconds audio, etc.)
- Figure out how to jam the audio input through. The encoder input is not represented as text / tokens.