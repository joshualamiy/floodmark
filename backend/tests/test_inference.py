import unittest

from floodmark_pipeline.inference import model_run


class InferenceTests(unittest.TestCase):
    def test_mvp_prediction_is_stable_for_retries(self):
        image = b"normalized jpeg bytes"

        first = model_run(image)
        second = model_run(image)

        self.assertEqual(first, second)
        self.assertIn(first.status, {"dry", "wet", "flooded"})
        self.assertEqual(first.model_version, {"data_version": "mvp-1", "stage_a_run_id": "stub-a", "stage_b_run_id": "stub-b"})
