"""Unified LLM client with text and vision support."""

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

log = logging.getLogger(__name__)


@dataclass
class UsageStats:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    successful_calls: int = 0
    failed_calls: int = 0


class LLMClient:
    def __init__(self, cfg, api_key: str):
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.client = OpenAI(api_key=api_key)
        self.stats = UsageStats()

    def _encode_image(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_image_content(self, images: list[Path]) -> list[dict]:
        parts = []
        for img_path in images:
            b64 = self._encode_image(img_path)
            suffix = Path(img_path).suffix.lstrip(".")
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                }
            )
        return parts

    def _attach_images(self, messages: list[dict], images: list[Path]) -> list[dict]:
        messages = [m.copy() for m in messages]
        last_user = next(m for m in reversed(messages) if m["role"] == "user")
        last_user["content"] = [
            {"type": "text", "text": last_user["content"]},
            *self._build_image_content(images),
        ]
        return messages

    def query_text(self, messages: list[dict]) -> str:
        return self._call(messages)

    def query_with_images(self, messages: list[dict], images: list[Path]) -> str:
        return self._call(self._attach_images(messages, images))

    def query_json(self, messages: list[dict], images: list[Path] | None = None) -> dict:
        """Call LLM and parse the response as JSON."""
        raw = self.query_with_images(messages, images) if images else self.query_text(messages)
        return self._parse_json(raw)

    def _parse_json(self, response: str) -> dict:
    # Strip markdown code fences
        match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
        raw = match.group(1).strip() if match else response.strip()

        # If no fences, try to isolate the JSON object
        if not match:
            obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
            raw = obj_match.group(0) if obj_match else raw

        def _sanitize(s: str) -> str:
            """Fix bare backslashes and literal newlines/tabs inside JSON string values."""
            result = []
            in_string = False
            i = 0
            while i < len(s):
                c = s[i]
                if c == '"' and (i == 0 or s[i-1] != '\\'):
                    num_backslashes = 0
                    j = i - 1
                    while j >= 0 and s[j] == '\\':
                        num_backslashes += 1
                        j -= 1
                    if num_backslashes % 2 == 0:  # even = not escaped
                        in_string = not in_string
                    result.append(c)
                    i += 1
                elif in_string and c == '\n':
                    result.append('\\n')
                    i += 1
                elif in_string and c == '\t':
                    result.append('\\t')
                    i += 1
                elif in_string and c == '\\':
                    # Check if this is a valid JSON escape sequence
                    next_c = s[i+1] if i+1 < len(s) else ''
                    if next_c in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                        result.append(c)
                        result.append(next_c)
                        i += 2
                    else:
                        # Bare backslash — escape it
                        result.append('\\\\')
                        i += 1
                else:
                    result.append(c)
                    i += 1
            return ''.join(result)

        try:
            return json.loads(_sanitize(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from LLM response:\n{response[:300]}") from e

    def _call(self, messages: list[dict]) -> str:
        self.stats.total_calls += 1
        # try-catch approved: OpenAI API is external, need to track failed_calls before re-raising
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self.stats.total_input_tokens += response.usage.prompt_tokens
            self.stats.total_output_tokens += response.usage.completion_tokens
            self.stats.successful_calls += 1
            return response.choices[0].message.content.strip()
        except Exception:
            self.stats.failed_calls += 1
            log.exception("LLM call failed")
            raise

    def usage_summary(self) -> dict:
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "total_input_tokens": self.stats.total_input_tokens,
            "total_output_tokens": self.stats.total_output_tokens,
        }
