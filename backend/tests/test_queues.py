import unittest

from floodmark_pipeline.queues import CAPTURE_QUEUE_NAME, SCHEDULER_QUEUE_NAME


class QueueTests(unittest.TestCase):
    def test_scheduler_and_capture_workers_use_distinct_queues(self):
        self.assertEqual(CAPTURE_QUEUE_NAME, "arq:queue")
        self.assertEqual(SCHEDULER_QUEUE_NAME, "arq:scheduler")
        self.assertNotEqual(CAPTURE_QUEUE_NAME, SCHEDULER_QUEUE_NAME)
