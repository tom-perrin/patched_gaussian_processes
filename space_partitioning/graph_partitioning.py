import numpy as np
import networkx as nx

from tree_partitioning import PartitioningTree

def get_constraints(node: PartitioningTree):
    '''
    Returns a list of all geometric constraints (v, threshold, sign) for this node
    '''
    constraints = []
    curr = node
    while curr.parent is not None:
        p = curr.parent
        sign = 1 if curr == p.left else -1
        constraints.append((p.pdir, p.threshold, sign))
        curr = p
    return constraints

def get_clipped_frontier(
        node: PartitioningTree, 
        padding: float = 10.0
        ):
    '''
    Computes the frontier segment of a node's split, clipped by the previous frontiers
    '''
    if node.pdir is None:
        return None
    
    v = node.pdir
    c = node.threshold
    v_orth = np.array([-v[1], v[0]])
    anchor = v * c # reference point on the segment

    # line : L(t) = anchor + t*v_orth
    # constraints : sign_i * ((anchor + t*v_orth) @ v_i) <= sign_i * c_i
    #               t * sign_i * v_orth @ v_i <= sign_i * (c_i - anchor @ v_i)
    t_min, t_max = -padding, padding

    constraints = get_constraints(node)
    for v_i, c_i, sign_i in constraints:
        dot_orth = v_orth @ v_i
        dot_anchor = anchor @ v_i

        # left hand side, right hand side of inequality
        lhs = sign_i * dot_orth
        rhs = sign_i * (c_i - dot_anchor)

        if abs(lhs) > 1e-9: # check that line is not vertical
            val = rhs / lhs
            if lhs > 0:
                t_max = min(t_max, val)
            else:
                t_min = max(t_min, val)
    
    if t_min > t_max:
        return None
    return anchor + t_min * v_orth, anchor + t_max * v_orth

def are_neighbors(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree
        ):
    
    path_a = []
    curr = leaf_a
    while curr:
        path_a.append(curr)
        curr = curr.parent

    lca = None
    curr = leaf_b
    while curr:
        if curr in path_a:
            lca = curr
            break
        curr = curr.parent
    
    if lca is None or lca.pdir is None:
        return False
    
    # constraints from both leaves
    combined_constraints = get_constraints(leaf_a) + get_constraints(leaf_b)
    segment = get_clipped_frontier(lca.pdir, lca.threshold, combined_constraints)

    if segment is not None:
        p1, p2 = segment
        return np.linalg.norm(p1 - p2) > 1e-5
    return False

class PartitioningGraph:
    '''
    Docstring for PartitioningGraph
    '''
    def __init__(
            self, 
            tree: PartitioningTree
            ):
        leaves = tree.get_leaves()
        G = nx.Graph()

        for i in range(len(leaves)):
            G.add_node(i)
            for j in range(i+1, len(leaves)):
                if are_neighbors(leaves[i], leaves[j]):
                    G.add_edge(i,j)
        return G