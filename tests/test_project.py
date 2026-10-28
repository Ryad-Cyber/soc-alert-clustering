"""Basic checks for the data generator and the clustering pipeline.

Run with:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cluster_alerts, generate_data  # noqa: E402

REQUIRED_COLUMNS = {
    "alert_id", "timestamp", "src_ip", "dst_ip", "dst_port",
    "protocol", "alert_type", "severity", "bytes_transferred", "event_count",
}


class GenerateDataTests(unittest.TestCase):
    def test_row_count_and_columns(self):
        df = generate_data.generate(rows=150, seed=1)
        self.assertEqual(len(df), 150)
        self.assertTrue(REQUIRED_COLUMNS.issubset(df.columns))

    def test_deterministic_with_fixed_seed(self):
        df1 = generate_data.generate(rows=100, seed=7)
        df2 = generate_data.generate(rows=100, seed=7)
        self.assertTrue(df1.equals(df2))

    def test_no_missing_values(self):
        df = generate_data.generate(rows=100, seed=3)
        self.assertFalse(df.isna().any().any())

    def test_severity_values_are_known(self):
        df = generate_data.generate(rows=200, seed=5)
        self.assertTrue(set(df["severity"].unique()) <= {"Low", "Medium", "High"})


class ClusterPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate_data.generate(rows=120, seed=42)

    def test_load_alerts_from_dataframe_roundtrip(self):
        # load_alerts expects a CSV path; write the generated frame to a
        # temporary file and make sure it comes back clean.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "alerts.csv"
            self.raw.to_csv(csv_path, index=False)
            df = cluster_alerts.load_alerts(csv_path)
            self.assertEqual(len(df), len(self.raw))
            self.assertIn("severity_rank", df.columns)
            self.assertFalse(df["severity_rank"].isna().any())

    def test_build_features_is_numeric(self):
        df = cluster_alerts.load_alerts(self._write_csv())
        features = cluster_alerts.build_features(df)
        self.assertTrue(all(features.dtypes.apply(lambda d: d.kind in "biuf")))
        self.assertEqual(len(features), len(df))

    def test_clustering_produces_expected_number_of_labels(self):
        from sklearn.preprocessing import StandardScaler

        df = cluster_alerts.load_alerts(self._write_csv())
        features = cluster_alerts.build_features(df)
        X = StandardScaler().fit_transform(features)
        k, scores = cluster_alerts.choose_k(X, k_min=2, k_max=6, seed=42)
        self.assertIn(k, scores)
        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 6)

    def test_summarize_clusters_shape(self):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = cluster_alerts.load_alerts(self._write_csv())
        features = cluster_alerts.build_features(df)
        X = StandardScaler().fit_transform(features)
        df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
        summary = cluster_alerts.summarize_clusters(df)
        self.assertEqual(summary["alert_count"].sum(), len(df))
        self.assertEqual(len(summary), 4)

    def _write_csv(self) -> Path:
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        csv_path = tmp_dir / "alerts.csv"
        self.raw.to_csv(csv_path, index=False)
        return csv_path


if __name__ == "__main__":
    unittest.main()
