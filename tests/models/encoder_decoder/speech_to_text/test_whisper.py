from typing import List, Optional, Tuple, Type
import numpy as np
import pytest
from transformers import AutoTokenizer, AutoModel, BatchEncoding

from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.inputs import ExplicitEncoderDecoderPrompt
from vllm import TextPrompt
from ....conftest import HfRunner, VllmRunner
from ...utils import check_logprobs_close

MODELS = ["openai/whisper-tiny"]  # Fast failures with smallest model
AUDIO_SAMPLE_RATE = 16000


@pytest.fixture(scope="session")
def audio_assets():
    from vllm.assets.audio import AudioAsset
    return [AudioAsset("mary_had_lamb")]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]], model: str):
    output_ids, output_str, out_logprobs = vllm_output
    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = output_ids[:]
    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(hf_runner: Type[HfRunner],
             vllm_runner: Type[VllmRunner],
             audio_data: Tuple[np.ndarray, int],
             model: str,
             prompts: List[str],
             *,
             dtype: str = "half",
             max_tokens: int = 128):
    """Compare vLLM and HF outputs with minimal ceremony"""

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1}
    ) as vllm_model:
        vllm_outputs = []
        for prompt in prompts:
            # Force encoder-decoder pathway
            output = vllm_model.generate(
                ExplicitEncoderDecoderPrompt(
                    encoder_prompt=TextPrompt(
                        text=prompt,
                        multi_modal_data={"audio": [audio_data]}
                    ),
                    decoder_prompt=TextPrompt(text=prompt)
                ),
                max_tokens=max_tokens
            )
            vllm_outputs.append(output)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        import librosa
        audio, sr = audio_data
        if sr != AUDIO_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)

        hf_outputs = []
        for prompt in prompts:
            output = hf_model.generate(
                [prompt],
                max_tokens,
                audios=[(audio, AUDIO_SAMPLE_RATE)]
            )
            hf_outputs.append(output)

    for hf_out, vllm_out in zip(hf_outputs, vllm_outputs):
        check_logprobs_close(
            outputs_0_lst=hf_out,
            outputs_1_lst=[vllm_to_hf_output(out, model) for out in vllm_out],
            name_0="hf",
            name_1="vllm"
        )


def test_whisper_core(hf_runner, vllm_runner, audio_assets):
    """Core test focusing on potentially problematic cases"""
    audio = audio_assets[0].audio_and_sample_rate

    edge_case_prompts = [
        # Basic transcription
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",

        # Force timestamp handling
        "<|startoftranscript|><|en|><|transcribe|><|timestamp|>",

        # Test language switching mid-stream
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|><|fr|>",

        # Empty prompt to test encoder-only behavior
        ""
    ]

    run_test(hf_runner, vllm_runner, audio, MODELS[0], edge_case_prompts)


# def test_whisper_empty_audio(hf_runner, vllm_runner):
#     """Test handling of empty/invalid audio input"""
#     empty_audio = (np.array([0.0] * 100), AUDIO_SAMPLE_RATE)
#     prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
#
#     run_test(hf_runner, vllm_runner, empty_audio, MODELS[0], [prompt])
