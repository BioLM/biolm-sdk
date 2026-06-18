"""Tests for biolmai.progress (shared Rich progress helper)."""
import unittest

from biolm.progress import rich_progress


class TestRichProgress(unittest.TestCase):
    def test_context_yields_callback(self):
        with rich_progress(total_items=10, description="Test") as callback:
            self.assertTrue(callable(callback))

    def test_callback_accepts_completed_total(self):
        with rich_progress(total_items=5, description="Test") as callback:
            callback(0, 5)
            callback(3, 5)
            callback(5, 5)

    def test_zero_total_yields_noop_callback(self):
        with rich_progress(total_items=0, description="Test") as callback:
            self.assertTrue(callable(callback))
            callback(0, 0)
