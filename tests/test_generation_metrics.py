import unittest
from unittest.mock import Mock

from execution_trace import TraceContext, trace_scope
from generation_metrics import GenerationMetrics
from usage import UsageAccumulator


def chunk(**delta):
    return {"choices": [{"delta": delta}]}


class GenerationMetricsTests(unittest.TestCase):
    def test_retry_generations_share_request_identity_but_have_distinct_metric_ids(self):
        trace = TraceContext.task("session").request()
        with trace_scope(trace):
            first = GenerationMetrics(UsageAccumulator())
            second = GenerationMetrics(UsageAccumulator())
        self.assertEqual(first.request_id, trace.request_id)
        self.assertEqual(second.request_id, trace.request_id)
        self.assertNotEqual(first.generation_id, second.generation_id)
        self.assertEqual(first.snapshot()["trace"], trace.as_dict())

    def test_tps_excludes_connection_and_first_output_wait(self):
        usage = UsageAccumulator()
        events = []
        metrics = GenerationMetrics(usage, events.append, clock=Mock(side_effect=[0, 30, 32]))
        metrics.observe(chunk(role="assistant"))
        metrics.observe(chunk(content="first"))
        metrics.observe(chunk(content="second"))
        usage.add({"usage": {"completion_tokens": 120}})
        metrics.observe({"choices": [], "usage": {"completion_tokens": 120}})
        metrics.publish(finished=True)
        self.assertEqual(events[-1]["tps"], 60)
        self.assertEqual(events[-1]["generation_seconds"], 2)
        self.assertEqual(events[-1]["excluded_wait_seconds"], 30)
        self.assertEqual(events[-1]["source"], "usage")
        self.assertTrue(events[-1]["finished"])

    def test_estimation_counts_reasoning_and_tool_output_without_duplicate_aliases(self):
        metrics = GenerationMetrics(UsageAccumulator(), clock=Mock(side_effect=[0, 10, 11]))
        metrics.observe(chunk(reasoning_content="abc", reasoning="abc", content="abc"))
        metrics.observe(chunk(tool_calls=[{"function": {"name": "read", "arguments": "{}"}}]))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["tokens"], 4)
        self.assertEqual(snapshot["tps"], 4)
        self.assertEqual(snapshot["source"], "estimate")

    def test_single_chunk_does_not_invent_a_generation_speed(self):
        metrics = GenerationMetrics(UsageAccumulator(), clock=Mock(side_effect=[0, 10]))
        metrics.observe(chunk(content="all tokens in one chunk"))
        self.assertIsNone(metrics.snapshot()["tps"])
        self.assertEqual(metrics.snapshot()["generation_seconds"], 0)

    def test_empty_stream_and_zero_duration_do_not_divide_by_zero(self):
        metrics = GenerationMetrics(UsageAccumulator(), clock=Mock(side_effect=[0, 10, 10]))
        self.assertIsNone(metrics.snapshot()["tps"])
        metrics.observe(chunk(content="one"))
        metrics.observe(chunk(content="two"))
        self.assertIsNone(metrics.snapshot()["tps"])

    def test_chunk_boundaries_do_not_inflate_estimated_token_counts(self):
        metrics = GenerationMetrics(UsageAccumulator(), clock=Mock(side_effect=[0, 1, 2, 3]))
        for character in "abc":
            metrics.observe(chunk(content=character))
        self.assertEqual(metrics.snapshot()["tokens"], 1)


if __name__ == "__main__":
    unittest.main()
