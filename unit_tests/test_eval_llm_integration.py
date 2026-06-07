"""Integration tests for MiniMax evaluation LLM provider.

These tests make real API calls to the MiniMax API.
Set MINIMAX_API_KEY environment variable to run.

Usage:
    MINIMAX_API_KEY=your-key python -m pytest unit_tests/test_eval_llm_integration.py -v
"""

import os
import unittest

from pipeline.benchmarks.utils.eval_llm import EvalLLMClient, get_eval_llm_client


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests against the live MiniMax API."""

    def setUp(self):
        self.client = EvalLLMClient(
            provider="minimax",
            api_key=MINIMAX_API_KEY,
            model="MiniMax-M3",
        )

    def test_basic_chat_completion(self):
        result = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                {"role": "user", "content": "What is 2 + 2? Answer with just the number."},
            ],
            temperature=0.01,
            max_tokens=256,
        )
        self.assertIn("4", result)

    def test_evaluation_judge_yes_no(self):
        result = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Your task is to judge whether the model response is correct to answer the given question or not."},
                {"role": "user", "content": "Question: What color is the sky?\nModel Response: The sky is blue.\nGround Truth: blue\nWill the model response be considered correct? You should only answer yes or no."},
            ],
            temperature=0.01,
            max_tokens=256,
        )
        self.assertIn("yes", result.lower())

    def test_scoring_correctness(self):
        result = self.client.chat_completion(
            messages=[
                {"role": "user", "content": "Compare the ground truth and prediction, give a correctness score from 0.0 to 1.0.\n\nQuestion: What is 2+2?\nGround Truth: 4\nPrediction: 4\n\nJust output the score number."},
            ],
            temperature=0.01,
            max_tokens=256,
        )
        self.assertTrue(len(result) > 0, "Response should not be empty")
        # Should contain a high score
        self.assertTrue(
            any(s in result for s in ["1.0", "1", "0.9", "0.8"]),
            f"Expected high score in response: {result}",
        )


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxAutoDetect(unittest.TestCase):
    """Test auto-detection of MiniMax provider."""

    def test_auto_detect_creates_minimax_client(self):
        client = get_eval_llm_client()
        self.assertEqual(client.provider, "minimax")
        self.assertEqual(client.api_base, "https://api.minimax.io/v1")


if __name__ == "__main__":
    unittest.main()
