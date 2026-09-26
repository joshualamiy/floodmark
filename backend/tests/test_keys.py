from datetime import datetime, timezone
import unittest

from floodmark_pipeline.keys import capture_job_id, capture_slot, object_key, skipped_key


class KeyTests(unittest.TestCase):
    def test_capture_job_id_uses_the_five_minute_utc_slot(self):
        scheduled_at = datetime(2026, 9, 26, 21, 19, 42, tzinfo=timezone.utc)

        self.assertEqual(capture_slot(scheduled_at), datetime(2026, 9, 26, 21, 15, tzinfo=timezone.utc))
        self.assertEqual(capture_job_id("GA511", "10651", scheduled_at), "capture:ga511:10651:2026-09-26T21:15Z")

    def test_object_key_uses_view_id_and_scheduled_utc_slot(self):
        captured_at = datetime(2026, 9, 26, 21, 15, tzinfo=timezone.utc)

        self.assertEqual(object_key("captures", "18558", captured_at), "captures/18558/20260926T2115Z.jpg")
        self.assertEqual(object_key("heatmaps", "18558", captured_at), "heatmaps/18558/20260926T2115Z.png")
        self.assertEqual(skipped_key("18558", captured_at), "skipped/18558/20260926T2115Z.jpg")

    def test_invalid_view_id_is_rejected(self):
        with self.assertRaises(ValueError):
            object_key("captures", "camera/18558", datetime.now(timezone.utc))
