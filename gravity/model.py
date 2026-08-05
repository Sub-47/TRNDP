import numpy as np
import pandas as pd
import networkx as nx


def compute_od_matrix(graph: nx.Graph, beta: float = 2.0, k: float = 1.0) -> pd.DataFrame:
    travel_times = dict(nx.all_pairs_dijkstra_path_length(graph, weight="travel_time_h"))

    records = []
    for origin_id, destinations in travel_times.items():
        origin_population = graph.nodes[origin_id]["population"]
        for destination_id, travel_time_h in destinations.items():
            if origin_id == destination_id:
                continue

            destination_jobs = graph.nodes[destination_id]["jobs"]
            trips = k * origin_population * destination_jobs / (travel_time_h ** beta)
            records.append(
                {
                    "origin_id": origin_id,
                    "destination_id": destination_id,
                    "trips": trips,
                }
            )

    return pd.DataFrame(records, columns=["origin_id", "destination_id", "trips"])