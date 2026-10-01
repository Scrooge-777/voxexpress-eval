"""
Unit tests for CLI and Pipeline end-to-end execution.
"""

import os
import unittest
import pandas as pd

from expresseval.pipeline import EvaluationPipeline
from expresseval.io.manifest import load_manifest


class TestCLIAndPipeline(unittest.TestCase):
    def setUp(self):
        self.manifest_path = "data/manifests/sample_manifest.csv"
        self.output_path = "reports/test_eval_output.csv"

    def test_load_manifest(self):
        items = load_manifest(self.manifest_path)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].language, "en")
        self.assertIn("fox", items[0].text)

    def test_pipeline_execution(self):
        pipeline = EvaluationPipeline()
        df = pipeline.run(self.manifest_path, output_path=self.output_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("overall_score", df.columns)
        self.assertIn("prosody_score", df.columns)
        self.assertIn("speaker_sim_score", df.columns)
        self.assertTrue(os.path.exists(self.output_path))

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)


if __name__ == "__main__":
    unittest.main()
