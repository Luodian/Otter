"""Unit tests for the configurable evaluation LLM client."""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from pipeline.benchmarks.utils.eval_llm import EvalLLMClient, get_eval_llm_client, PROVIDER_CONFIGS


class TestProviderConfigs(unittest.TestCase):
    """Test provider configuration constants."""

    def test_openai_config_exists(self):
        self.assertIn("openai", PROVIDER_CONFIGS)
        self.assertEqual(PROVIDER_CONFIGS["openai"]["api_base"], "https://api.openai.com/v1")
        self.assertEqual(PROVIDER_CONFIGS["openai"]["api_key_env"], "OPENAI_API_KEY")

    def test_minimax_config_exists(self):
        self.assertIn("minimax", PROVIDER_CONFIGS)
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["api_base"], "https://api.minimax.io/v1")
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["default_model"], "MiniMax-M3")
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["api_key_env"], "MINIMAX_API_KEY")


class TestEvalLLMClientInit(unittest.TestCase):
    """Test EvalLLMClient initialization."""

    def test_explicit_openai_provider(self):
        client = EvalLLMClient(provider="openai", api_key="test-key")
        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.api_base, "https://api.openai.com/v1")
        self.assertEqual(client.model, "gpt-4-0613")
        self.assertEqual(client.api_key, "test-key")

    def test_explicit_minimax_provider(self):
        client = EvalLLMClient(provider="minimax", api_key="test-key")
        self.assertEqual(client.provider, "minimax")
        self.assertEqual(client.api_base, "https://api.minimax.io/v1")
        self.assertEqual(client.model, "MiniMax-M3")
        self.assertEqual(client.api_key, "test-key")

    def test_custom_model_override(self):
        client = EvalLLMClient(provider="minimax", api_key="key", model="MiniMax-M2.7")
        self.assertEqual(client.model, "MiniMax-M2.7")

    def test_custom_api_base_override(self):
        client = EvalLLMClient(provider="openai", api_key="key", api_base="https://custom.api.com/v1")
        self.assertEqual(client.api_base, "https://custom.api.com/v1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}, clear=False)
    def test_auto_detect_minimax_from_env(self):
        client = EvalLLMClient()
        self.assertEqual(client.provider, "minimax")
        self.assertEqual(client.api_key, "env-minimax-key")

    @patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "env-key"}, clear=False)
    def test_explicit_env_provider(self):
        client = EvalLLMClient()
        self.assertEqual(client.provider, "minimax")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=False)
    def test_default_to_openai(self):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        env.pop("EVAL_LLM_PROVIDER", None)
        with patch.dict(os.environ, env, clear=True):
            client = EvalLLMClient()
            self.assertEqual(client.provider, "openai")


class TestTemperatureClamping(unittest.TestCase):
    """Test temperature clamping for MiniMax."""

    def test_minimax_clamps_zero_temperature(self):
        client = EvalLLMClient(provider="minimax", api_key="key")
        self.assertEqual(client._clamp_temperature(0.0), 0.01)

    def test_minimax_preserves_nonzero_temperature(self):
        client = EvalLLMClient(provider="minimax", api_key="key")
        self.assertEqual(client._clamp_temperature(0.7), 0.7)

    def test_openai_preserves_zero_temperature(self):
        client = EvalLLMClient(provider="openai", api_key="key")
        self.assertEqual(client._clamp_temperature(0.0), 0.0)


class TestThinkTagStripping(unittest.TestCase):
    """Test <think>...</think> tag stripping for MiniMax."""

    def test_minimax_strips_think_tags(self):
        client = EvalLLMClient(provider="minimax", api_key="key")
        content = "<think>Let me think about this...</think>\nThe answer is yes."
        self.assertEqual(client._strip_think_tags(content), "The answer is yes.")

    def test_minimax_strips_multiline_think_tags(self):
        client = EvalLLMClient(provider="minimax", api_key="key")
        content = "<think>\nStep 1: analyze\nStep 2: conclude\n</think>\n0.8"
        self.assertEqual(client._strip_think_tags(content), "0.8")

    def test_minimax_preserves_content_without_think_tags(self):
        client = EvalLLMClient(provider="minimax", api_key="key")
        content = "The answer is yes."
        self.assertEqual(client._strip_think_tags(content), "The answer is yes.")

    def test_openai_preserves_all_content(self):
        client = EvalLLMClient(provider="openai", api_key="key")
        content = "<think>some content</think>\nThe answer is yes."
        self.assertEqual(client._strip_think_tags(content), content)


class TestChatCompletion(unittest.TestCase):
    """Test chat completion with mocked HTTP responses."""

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    def test_successful_openai_completion(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "yes"}}],
            "model": "gpt-4-0613",
        }
        mock_post.return_value = mock_response

        client = EvalLLMClient(provider="openai", api_key="test-key")
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Is this correct?"}],
            temperature=0,
            max_tokens=256,
        )

        self.assertEqual(result, "yes")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn("api.openai.com", call_args[0][0])

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    def test_successful_minimax_completion(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "<think>analyzing...</think>\n0.8"}}],
            "model": "MiniMax-M3",
        }
        mock_post.return_value = mock_response

        client = EvalLLMClient(provider="minimax", api_key="test-key")
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Score this answer"}],
            temperature=0,
            max_tokens=3,
        )

        self.assertEqual(result, "0.8")
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        self.assertEqual(payload["temperature"], 0.01)  # clamped
        self.assertIn("api.minimax.io", call_args[0][0])

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    @patch("pipeline.benchmarks.utils.eval_llm.time.sleep")
    def test_retry_on_failure(self, mock_sleep, mock_post):
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = Exception("Rate limit exceeded")

        mock_success = MagicMock()
        mock_success.raise_for_status = MagicMock()
        mock_success.json.return_value = {
            "choices": [{"message": {"content": "yes"}}],
        }

        mock_post.side_effect = [mock_fail, mock_success]

        client = EvalLLMClient(provider="openai", api_key="test-key")
        result = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            patience=3,
            sleep_time=1,
        )

        self.assertEqual(result, "yes")
        self.assertEqual(mock_post.call_count, 2)

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    @patch("pipeline.benchmarks.utils.eval_llm.time.sleep")
    def test_returns_empty_on_exhausted_retries(self, mock_sleep, mock_post):
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = Exception("Server error")
        mock_post.return_value = mock_fail

        client = EvalLLMClient(provider="openai", api_key="test-key")
        result = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            patience=2,
            sleep_time=0,
        )

        self.assertEqual(result, "")
        self.assertEqual(mock_post.call_count, 2)


class TestChatCompletionRaw(unittest.TestCase):
    """Test raw chat completion that returns response dict."""

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    def test_returns_content_and_response_data(self, mock_post):
        response_data = {
            "choices": [{"message": {"content": "0.7"}}],
            "model": "gpt-4-0613",
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = response_data
        mock_post.return_value = mock_response

        client = EvalLLMClient(provider="openai", api_key="test-key")
        content, raw = client.chat_completion_raw(
            messages=[{"role": "user", "content": "test"}],
        )

        self.assertEqual(content, "0.7")
        self.assertEqual(raw["model"], "gpt-4-0613")

    @patch("pipeline.benchmarks.utils.eval_llm.requests.post")
    def test_minimax_strips_think_tags_in_raw(self, mock_post):
        response_data = {
            "choices": [{"message": {"content": "<think>thinking</think>\n0.9"}}],
            "model": "MiniMax-M3",
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = response_data
        mock_post.return_value = mock_response

        client = EvalLLMClient(provider="minimax", api_key="test-key")
        content, raw = client.chat_completion_raw(
            messages=[{"role": "user", "content": "test"}],
        )

        self.assertEqual(content, "0.9")


class TestGetEvalLLMClient(unittest.TestCase):
    """Test factory function."""

    def test_creates_client_with_defaults(self):
        client = get_eval_llm_client(provider="openai", api_key="key")
        self.assertIsInstance(client, EvalLLMClient)
        self.assertEqual(client.provider, "openai")

    def test_creates_minimax_client(self):
        client = get_eval_llm_client(provider="minimax", api_key="key", model="MiniMax-M2.7")
        self.assertEqual(client.provider, "minimax")
        self.assertEqual(client.model, "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
