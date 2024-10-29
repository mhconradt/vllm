from typing import Tuple, List, Optional, Type

import numpy as np
import pytest
from transformers import AutoTokenizer, AutoModel, BatchEncoding

from tests.conftest import HfRunner, VllmRunner
from tests.models.utils import check_logprobs_close
from vllm import TextPrompt
from vllm.inputs import EncoderDecoderInputs, ExplicitEncoderDecoderPrompt
from vllm.multimodal import MultiModalDataDict
from vllm.sequence import SampleLogprobs
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

MODELS = ["openai/whisper-large-v3", "whisper-tiny"]

AudioTuple = Tuple[np.ndarray, int]


@pytest.fixture(scope="session")
def audio_assets():
    from vllm.assets.audio import AudioAsset
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture(scope="module", params=("mary_had_lamb", "winning_call"))
def audio(request):
    from vllm.assets.audio import AudioAsset
    return AudioAsset(request.param)


def vllm_to_hf_output(vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = output_ids[:]
    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(
        hf_runner: Type[HfRunner],
        vllm_runner: Type[VllmRunner],
        prompts_and_audios: List[Tuple[str, str, AudioTuple]],
        model: str,
        *,
        dtype: str,
        max_tokens: int,
        num_logprobs: int,
        tensor_parallel_size: int,
        distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm."""
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_model.model.generate(
            ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    text="",
                    multi_modal_data={
                        "audio": ...,
                    }
                ),
                decoder_prompt=...
            )
        )
        vllm_outputs_per_audio = [
            vllm_model.generate_greedy_logprobs([vllm_prompt],
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                audios=[audio])
            for vllm_prompt, _, audio in prompts_and_audios
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["audio_values"] = hf_inputs["audio_values"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModel) as hf_model:
        import librosa

        hf_outputs_per_audio = [
            hf_model.generate_greedy_logprobs_limit(
                [hf_prompt],
                max_tokens,
                num_logprobs=num_logprobs,
                audios=[(librosa.resample(audio[0],
                                          orig_sr=audio[1],
                                          target_sr=16000), 16000)])
            for _, hf_prompt, audio in prompts_and_audios
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_audio,
                                        vllm_outputs_per_audio):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("model_name", MODELS)
def test_models(
        hf_runner,
        vllm_runner,
        audio,
        dtype: str,
        max_tokens: int,
        num_logprobs: int,
        model_name: str,
) -> None:
    pfx = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    run_test(
        hf_runner,
        vllm_runner,
        [(pfx, pfx, audio.audio_and_sample_rate)],
        model_name,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
