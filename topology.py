"""
CS 422 Assignment 4: topology design that maximizes concurrent flow.
 
Given n nodes, each with d outgoing and d incoming unit-capacity links,
and a traffic matrix T, choose the links (topology) AND the routing so
that the concurrent flow theta is maximized.
 
MILP formulation (source-aggregated multicommodity flow):
  Variables
    x[i,j]   integer >= 0   number of links i -> j (i != j)
    f[s,i,j] >= 0           flow on link i -> j of traffic that originates at s
    theta    >= 0           concurrent flow (fraction of every demand delivered)
  Objective
    maximize theta
  Constraints
    1: out-degree: sum_j x[i,j] = d             for every node i
    2: in-degree:  sum_i x[i,j] = d             for every node j
    3: capacity:   sum_s f[s,i,j] <= x[i,j]     for every link (i,j)
    4: conservation for each source s:
          at s:      out - in = theta * sum_t T[s,t]
          at v != s: in - out = theta * T[s,v]
 
Usage
  python3 topology.py --uniform 7 3          # part 1 experiment
  python3 topology.py --hose-random --seed 1 # part 2, random hose TM (n=8,d=4)
  python3 topology.py --tm tm.csv --d 4      # part 2, your own TM
"""
 
import argparse
import itertools
import random
 
import gurobipy as gp
from gurobipy import GRB
 
 
def uniform_tm(n, d):
    # Uniform TM: every off-diagonal entry is d / (n - 1).
    T = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                row.append(d / (n - 1))
        T.append(row)
    return T
 
 
def random_hose_tm(n, d, seed):
    """Random TM in the hose set: row sums and column sums equal d.
 
    A sum of d random permutation matrices (no fixed points) with random
    weights, scaled so every row and column sums to exactly d.
    """
    rng = random.Random(seed)
    T = [[0.0] * n for _ in range(n)]
    k = 6  # number of permutations to mix
    weights = [rng.random() for _ in range(k)]
    total = sum(weights)
    for w in weights:
        # random derangement
        while True:
            p = list(range(n))
            rng.shuffle(p)
            if all(p[i] != i for i in range(n)):
                break
        for i in range(n):
            T[i][p[i]] += d * w / total
    return T
 
 
def read_tm(path):
    T = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                T.append([float(v) for v in line.split(",")])
    return T
 
 
def check_hose(T, d, tol=1e-6):
    n = len(T)
    for i in range(n):
        if T[i][i] != 0:
            raise ValueError(f"T[{i}][{i}] must be 0")
        if sum(T[i]) > d + tol:
            raise ValueError(f"row {i} sums to {sum(T[i]):.4f} > {d}")
    for j in range(n):
        col = sum(T[i][j] for i in range(n))
        if col > d + tol:
            raise ValueError(f"column {j} sums to {col:.4f} > {d}")
 
 
def solve(T, d, allow_parallel=True, verbose=False):
    n = len(T)
    nodes = range(n)
    links = [(i, j) for i in nodes for j in nodes if i != j]
 
    m = gp.Model("max_concurrent_flow_topology")
    m.Params.OutputFlag = 1 if verbose else 0
 
    # Decision variables
    ub = d if allow_parallel else 1
    x = m.addVars(links, vtype=GRB.INTEGER, lb=0, ub=ub, name="x")
    f = m.addVars(nodes, links, lb=0.0, name="f")
    theta = m.addVar(lb=0.0, name="theta")
 
    # Objective
    m.setObjective(theta, GRB.MAXIMIZE)
 
    # 1, 2: degree constraints
    for i in nodes:
        m.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == d, name=f"out_{i}")
        m.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == d, name=f"in_{i}")
 
    # 3: capacity: total flow on a link <= number of links placed there
    for (i, j) in links:
        m.addConstr(gp.quicksum(f[s, i, j] for s in nodes) <= x[i, j], name=f"cap_{i}_{j}")
 
    # 4: flow conservation, one commodity per source
    for s in nodes:
        for v in nodes:
            out_flow = gp.quicksum(f[s, v, w] for w in nodes if w != v)
            in_flow = gp.quicksum(f[s, w, v] for w in nodes if w != v)
            if v == s:
                m.addConstr(out_flow - in_flow == theta * sum(T[s]), name=f"src_{s}")
            else:
                m.addConstr(in_flow - out_flow == theta * T[s][v], name=f"cons_{s}_{v}")
 
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status}")
 
    X = [[0] * n for _ in nodes]
    for (i, j) in links:
        X[i][j] = int(round(x[i, j].X))
    return theta.X, X
 
 
def avg_shortest_path(X):
    #Average hop distance over all ordered pairs (BFS from each node).
    n = len(X)
    total = 0
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in range(n):
                    if X[u][v] > 0 and v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        if len(dist) < n:
            return float("inf")
        total += sum(dist.values())
    return total / (n * (n - 1))
 
 
def moore_avg_path(n, d):
    """Smallest possible average distance: d nodes at 1 hop, d^2 at 2, ..."""
    remaining = n - 1
    total = 0
    k = 1
    while remaining > 0:
        take = min(d ** k, remaining)
        total += k * take
        remaining -= take
        k += 1
    return total / (n - 1)
 
 
def print_matrix(name, M, fmt="{:5.2f}"):
    print(name)
    for row in M:
        print("  " + " ".join(fmt.format(v) for v in row))
 
 
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--uniform", nargs=2, type=int, metavar=("N", "D"))
    g.add_argument("--hose-random", action="store_true")
    g.add_argument("--tm", help="CSV file, one row per line")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-parallel", action="store_true", help="forbid parallel links")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
 
    if args.uniform:
        n, d = args.uniform
        T = uniform_tm(n, d)
    elif args.hose_random:
        n, d = args.n, args.d
        T = random_hose_tm(n, d, args.seed)
    else:
        T = read_tm(args.tm)
        n, d = len(T), args.d
 
    check_hose(T, d)
    theta, X = solve(T, d, allow_parallel=not args.no_parallel, verbose=args.verbose)
 
    print(f"n={n} d={d}")
    print_matrix("Traffic matrix T:", T)
    print_matrix("Topology x (number of links i->j):", X, fmt="{:2d}")
    print(f"theta = {theta:.4f}")
 
    if args.uniform:
        aspl = avg_shortest_path(X)
        print(f"avg shortest path of chosen graph = {aspl:.4f}  (1/aspl = {1 / aspl:.4f})")
        print(f"Moore lower bound on avg path     = {moore_avg_path(n, d):.4f}  "
              f"(theta upper bound = {1 / moore_avg_path(n, d):.4f})")
 
 
if __name__ == "__main__":
    main()
