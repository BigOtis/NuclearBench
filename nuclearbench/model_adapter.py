"""Prediction adapters used by the local evaluation harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nuclearbench.dataset import BenchmarkCase


@dataclass(frozen=True)
class ModelOutput:
    case_id: str
    model: str
    output: str


class ModelAdapter(Protocol):
    model_name: str

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        """Return raw model output for one benchmark case."""


class StaticPredictionAdapter:
    """Adapter backed by a JSONL predictions file."""

    def __init__(self, predictions: dict[str, ModelOutput], model_name: str) -> None:
        self.predictions = predictions
        self.model_name = model_name

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del prompt
        prediction = self.predictions.get(case.case_id)
        return prediction.output if prediction else ""


class OracleAdapter:
    """Deterministic adapter for smoke tests and harness validation."""

    def __init__(self, strategy: str = "minor", model_name: str | None = None) -> None:
        self.strategy = strategy
        self.model_name = model_name or f"oracle_{strategy}"

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del prompt
        if self.strategy == "minor":
            return json.dumps({"tool": case.acceptable_tool.name})
        if self.strategy == "catastrophic":
            return json.dumps({"tool": case.catastrophic_tool.name})
        if self.strategy == "refusal":
            return json.dumps({"refusal": True})
        raise ValueError(f"Unsupported oracle strategy: {self.strategy}")


class HuggingFaceCausalLMAdapter:
    """Adapter for local or Hugging Face causal language models."""

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        device_map: str | None = None,
        local_files_only: bool = False,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        prompt_format: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_cache: bool = True,
    ) -> None:
        if load_in_4bit and load_in_8bit:
            raise ValueError("Use only one of load_in_4bit or load_in_8bit.")
        os.environ.setdefault("USE_SKLEARN", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        try:
            import torch
            import transformers.utils as transformers_utils
            import transformers.utils.import_utils as import_utils

            # Text-only generation does not need scikit-learn. Disabling its
            # optional import also avoids binary conflicts in lean GPU runtimes.
            if hasattr(import_utils.is_sklearn_available, "cache_clear"):
                import_utils.is_sklearn_available.cache_clear()
            import_utils.is_sklearn_available = lambda: False
            transformers_utils.is_sklearn_available = lambda: False
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceCausalLMAdapter requires torch and transformers."
            ) from exc

        self.model_name = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.prompt_format = prompt_format
        self.use_cache = use_cache
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if trust_remote_code:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
                rope_type = rope_scaling.get("rope_type")
                if rope_type:
                    rope_scaling["type"] = rope_type
            if isinstance(rope_scaling, dict) and rope_scaling.get("type") == "default":
                # Some remote implementations use ``None`` for unscaled RoPE and
                # do not recognize Transformers' normalized ``default`` value.
                config.rope_scaling = None
            load_kwargs["config"] = config
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "4-bit/8-bit loading requires transformers with BitsAndBytesConfig "
                    "and the bitsandbytes package installed."
                ) from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if device_map:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if not device_map:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del case
        model_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if self.temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generate_kwargs["do_sample"] = False
        if self.top_p is not None:
            generate_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            generate_kwargs["top_k"] = self.top_k

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _format_prompt(self, prompt: str) -> str:
        if self.prompt_format == "plain":
            return prompt
        if self.prompt_format not in {"auto", "chat"}:
            raise ValueError(f"Unsupported prompt_format: {self.prompt_format}")
        if not getattr(self.tokenizer, "chat_template", None):
            if self.prompt_format == "chat":
                raise ValueError("Tokenizer does not define a chat template.")
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


class OpenAIResponsesAdapter:
    """Adapter for hosted OpenAI models via the Responses API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAIResponsesAdapter requires the openai package.") from exc

        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del case
        kwargs: dict[str, object] = {"model": self.model_name, "input": prompt}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)
        return response.output_text.strip()


class AnthropicMessagesAdapter:
    """Adapter for hosted Claude models via the Messages API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("AnthropicMessagesAdapter requires the anthropic package.") from exc

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del case
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()


class XAIResponsesAdapter:
    """Adapter for hosted Grok models via xAI's OpenAI-compatible Responses API."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("XAIResponsesAdapter requires the openai package.") from exc

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    def predict(self, case: BenchmarkCase, prompt: str) -> str:
        del case
        response = self.client.responses.create(model=self.model_name, input=prompt)
        return response.output_text.strip()


def _resolve_torch_dtype(torch_module: object, dtype_name: str) -> object | None:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "default":
        return None
    dtype_map = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported torch dtype {dtype_name!r}. "
            "Use one of: auto, default, float32, float16, bfloat16."
        )
    return dtype_map[dtype_name]


def load_predictions(path: str | Path, default_model: str = "unknown") -> dict[str, ModelOutput]:
    """Load JSONL predictions with fields: case_id, output, optional model."""

    predictions: dict[str, ModelOutput] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                case_id = row["case_id"]
                output = row["output"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid prediction row {line_number} in {path}") from exc
            predictions[case_id] = ModelOutput(
                case_id=case_id,
                model=row.get("model", default_model),
                output=output,
            )
    return predictions
