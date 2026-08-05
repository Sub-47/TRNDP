import unittest
from pathlib import Path

from gravity.model import compute_od_matrix
from graph.loader import load_graph


class GravityModelTests(unittest.TestCase):
    def test_load_graph_and_compute_od_matrix(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        graph = load_graph(str(project_root / "data" / "nodes.csv"), str(project_root / "data" / "edges.csv"))
        od_matrix = compute_od_matrix(graph)

        self.assertTrue(len(od_matrix) > 0)
        self.assertIn("origin_id", od_matrix.columns)
        self.assertIn("destination_id", od_matrix.columns)
        self.assertIn("trips", od_matrix.columns)


if __name__ == "__main__":
    unittest.main()
