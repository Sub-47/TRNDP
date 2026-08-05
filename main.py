from pathlib import Path

from graph.loader import load_graph
from gravity.model import compute_od_matrix


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    graph = load_graph(str(data_dir / "nodes.csv"), str(data_dir / "edges.csv"))
    od_matrix = compute_od_matrix(graph)

    output_path = data_dir / "od_matrix.csv"
    od_matrix.to_csv(output_path, index=False)

    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Computed {len(od_matrix)} OD pairs.")
    print(f"OD matrix written to {output_path}")


if __name__ == "__main__":
    main()
