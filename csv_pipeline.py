import random
import csv
import os

def generate_graph(num_nodes: int, num_edges: int):
    # sanity bound
    max_possible = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_possible:
        raise ValueError(f"Too many edges. Max possible for {num_nodes} nodes is {max_possible}")

    nodes = list(range(num_nodes))
    edges = set()

    # ensure connectivity by building a spanning tree first
    available = nodes.copy()
    random.shuffle(available)

    base = available.pop()
    while available:
        nxt = available.pop()
        edges.add(tuple(sorted((base, nxt))))
        base = nxt

    # add remaining edges randomly
    remaining = num_edges - len(edges)
    while remaining > 0:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v:
            continue
        e = tuple(sorted((u, v)))
        if e not in edges:
            edges.add(e)
            remaining -= 1

    return sorted(list(edges))


def export_csv(edges, path="custom_graph.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for u, v in edges:
            writer.writerow([u, v])
    return os.path.abspath(path)


if __name__ == "__main__":
    # user input (adjust as needed)
    num_nodes = int(input("Nodes: "))
    num_edges = int(input("Edges: "))

    edges = generate_graph(num_nodes, num_edges)
    file_path = export_csv(edges)

    print(f"Generated: {file_path}")
