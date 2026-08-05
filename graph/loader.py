import pandas as pd
import networkx as nx


def load_graph(nodes_path: str, edges_path: str) -> nx.Graph:
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    graph = nx.Graph()

    for row in nodes_df.itertuples(index=False):
        graph.add_node(row.node_id, population=row.population, jobs=row.jobs)

    for row in edges_df.itertuples(index=False):
        travel_time_h = row.length_km / row.speed_kmh
        graph.add_edge(
            row.source_id,
            row.target_id,
            length_km=row.length_km,
            speed_kmh=row.speed_kmh,
            travel_time_h=travel_time_h,
        )

    return graph