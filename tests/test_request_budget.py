import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_client import CompletionError, DeepSeekClient
from request_budget import TokenBudget, TokenBudgetExhausted, budget_scope, reserve_request


class RequestBudgetTests(unittest.TestCase):
    def client(self):
        return DeepSeekClient(SimpleNamespace(
            api_key="fixture", model="test", base_url="https://budget-tests.invalid",
            max_output_tokens=100, request_retries=1, retry_backoff_seconds=0,
        ))

    def test_failed_attempt_usage_reduces_the_next_output_cap(self):
        client = self.client()
        budget = TokenBudget("task", 80)
        output_caps = []

        def stream(request, **_options):
            output_caps.append(json.loads(request.data)["max_tokens"])
            packet = "data: " + json.dumps({
                "choices": [{"delta": {"content": "partial or final"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 13},
            }) + "\n"
            if len(output_caps) == 2:
                packet += "data: [DONE]\n"
            return io.BytesIO(packet.encode())

        with budget_scope(budget), patch("http_transport.open_url", side_effect=stream):
            answer = client.complete([], on_content=lambda _chunk: None, purpose="summary")
        self.assertEqual(answer["content"], "partial or final")
        self.assertEqual(output_caps[0] - output_caps[1], 20)
        self.assertEqual(budget.snapshot()["measured"], 40)
        self.assertEqual(budget.snapshot()["requests"], 2)
        self.assertEqual(budget.snapshot()["reserved"], 0)

    def test_impossible_real_request_never_reaches_transport(self):
        client = self.client()
        budget = TokenBudget("task", 10)
        with budget_scope(budget), patch("http_transport.open_url") as transport:
            with self.assertRaises(CompletionError) as failure:
                client.complete([{"role": "user", "content": "large " * 50}])
        self.assertEqual(failure.exception.code, "budget_exhausted")
        transport.assert_not_called()
        self.assertEqual(budget.snapshot()["requests"], 0)

    def test_partial_failed_usage_is_charged_before_retry(self):
        client = self.client()
        budget = TokenBudget("task", 100)
        caps = []

        def stream(request, **_options):
            caps.append(json.loads(request.data)["max_tokens"])
            usage = {"prompt_tokens": 60} if len(caps) == 1 else {
                "prompt_tokens": 7, "completion_tokens": 3,
            }
            packet = "data: " + json.dumps({
                "choices": [{"delta": {"content": "answer"}}], "usage": usage,
            }) + "\n"
            if len(caps) == 2:
                packet += "data: [DONE]\n"
            return io.BytesIO(packet.encode())

        with budget_scope(budget), patch("http_transport.open_url", side_effect=stream):
            client.complete([], on_content=lambda _chunk: None)
        self.assertEqual(caps[0] - caps[1], 60)
        self.assertEqual(budget.snapshot()["measured"], 70)
        self.assertEqual(budget.snapshot()["unknown"], 1)
        self.assertEqual(budget.remaining(), 30)

    def test_caps_output_and_settles_measured_usage(self):
        budget = TokenBudget("task", 50)
        with budget_scope(budget), reserve_request(20, 100) as reservation:
            self.assertEqual(reservation.output_cap, 30)
            reservation.dispatched = True
            reservation.measured = 31
        self.assertEqual(budget.snapshot()["reserved"], 0)
        self.assertEqual(budget.snapshot()["measured"], 31)
        self.assertEqual(budget.remaining(), 19)

    def test_minimum_valid_output_must_fit_every_scope(self):
        task = TokenBudget("task", 100)
        goal = TokenBudget("goal", 200, measured=160)
        with budget_scope(task, goal), self.assertRaises(TokenBudgetExhausted):
            with reserve_request(40, 100):
                self.fail("An impossible request was admitted")
        self.assertEqual(task.snapshot()["reserved"], 0)

    def test_concurrent_reservations_cannot_spend_the_same_budget(self):
        budget = TokenBudget("session", 70)
        ready = threading.Event()
        release = threading.Event()

        def hold():
            with budget_scope(budget), reserve_request(30, 30):
                ready.set()
                release.wait(3)

        thread = threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(ready.wait(3))
            with budget_scope(budget), self.assertRaises(TokenBudgetExhausted):
                with reserve_request(12, 3):
                    self.fail("Double reservation")
        finally:
            release.set()
            thread.join()
        self.assertEqual(budget.remaining(), 70)
        self.assertEqual(budget.snapshot()["requests"], 0)

    def test_unobserved_failure_and_cancellation_release_reservations(self):
        budget = TokenBudget("task", 100)
        with budget_scope(budget):
            with self.assertRaises(KeyboardInterrupt), reserve_request(30, 50) as reservation:
                reservation.dispatched = True
                raise KeyboardInterrupt
            with self.assertRaises(ValueError), reserve_request(30, 50):
                raise ValueError("input not even dispatched")
        self.assertEqual(budget.snapshot()["unknown"], 1)
        self.assertEqual(budget.snapshot()["measured"], 0)
        self.assertEqual(budget.snapshot()["requests"], 1)
        self.assertEqual(budget.remaining(), 100)
