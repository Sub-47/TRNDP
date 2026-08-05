from pathlib import Path

import config
from city.managers.map_manager import MapManager
from city.models.world import World
from graph.loader import load_graph
from gravity.model import compute_od_matrix
from zoning.allocator import allocate_zones, write_zone_csvs
from zoning.visualize import render_zone_map


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    nodes_path = data_dir / "generated_nodes.csv"
    edges_path = data_dir / "generated_edges.csv"
    od_matrix_path = data_dir / "generated_od_matrix.csv"
    world_map_path = root / config.OUTPUT_DIR / "world_map.png"

    world = World(MapManager())
    world.generate()
    world.save()

    nodes_df, edges_df = allocate_zones(world)
    write_zone_csvs(nodes_df, edges_df, nodes_path, edges_path)
    print(f"Placed {len(nodes_df)} zones and {len(edges_df)} roads.")

    render_zone_map(world, nodes_df, edges_df, world_map_path)
    print(f"Zone/road overlay written to {world_map_path}")

    graph = load_graph(str(nodes_path), str(edges_path))
    od_matrix = compute_od_matrix(graph)
    od_matrix.to_csv(od_matrix_path, index=False)
    print(f"Computed {len(od_matrix)} OD pairs -> {od_matrix_path}")


if __name__ == "__main__":
    main()
