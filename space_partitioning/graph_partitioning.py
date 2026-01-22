import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tree_partitioning import PartitioningTree

def get_constraints(node: PartitioningTree):
    '''
    Returns a list of all geometric constraints (v, threshold, sign) for a given node (partitioning tree).
    '''
    # Initialization
    constraints = []
    curr = node

    # Browsing parents recursively to get all anterior constraints
    while curr.parent is not None:
        p = curr.parent
        sign = 1 if curr == p.left else -1
        constraints.append((p.pdir, p.threshold, sign))
        curr = p
    
    return constraints

def get_clipped_frontier(
        node: PartitioningTree, 
        padding: float = 10.0,
        external_constraints: list = None
        ):
    '''
    Computes the frontier segment of a node's split, clipped by the previous frontiers
    '''
    if node.pdir is None:
        return None
    
    # Frontier segment equation
    v = node.pdir
    c = node.threshold
    v_orth = np.array([-v[1], v[0]])
    anchor = v * c # reference point on the segment

    # Parametrization of the end points of the frontier segment
    t_min, t_max = -padding, padding

    # Checks for external constraints already defined
    if external_constraints is not None:
        constraints = external_constraints
    else:
        constraints = get_constraints(node)

    # line : L(t) = anchor + t*v_orth
    # constraints : sign_i * ((anchor + t*v_orth) @ v_i) <= sign_i * c_i
    #               t * sign_i * v_orth @ v_i <= sign_i * (c_i - anchor @ v_i)
    for v_i, c_i, sign_i in constraints:
        dot_orth = v_orth @ v_i
        dot_anchor = anchor @ v_i

        # left hand side, right hand side of inequality
        lhs = sign_i * dot_orth
        rhs = sign_i * (c_i - dot_anchor)

        if abs(lhs) > 1e-9: # checks that line is not vertical
            val = rhs / lhs # potential new value of t_max / t_min
            if lhs > 0: # segment potentially clips in the right side
                t_max = min(t_max, val)
            else: # segment potentially clips in the left side
                t_min = max(t_min, val)
    
    if t_min > t_max: # no frontier
        return None
    return anchor + t_min * v_orth, anchor + t_max * v_orth # coordinates of the 2 end points

def are_neighbors(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree
        ):
    
    # Finds path to leaf_a from tree root
    path_a = []
    curr = leaf_a
    while curr:
        path_a.append(curr)
        curr = curr.parent

    # Finds lowest common ancestor of leaf_a and leaf_b
    lca = None
    curr = leaf_b
    while curr:
        if curr in path_a:
            lca = curr
            break
        curr = curr.parent
    
    if lca is None or lca.pdir is None:
        return False
    
    # Gets consraints from both leaves and creates associated frontier
    combined_constraints = get_constraints(leaf_a) + get_constraints(leaf_b)
    segment = get_clipped_frontier(lca, external_constraints=combined_constraints)

    # Frontier segment non-empty => shared border => leaves are neighbors
    if segment is not None:
        p1, p2 = segment
        return np.linalg.norm(p1 - p2) > 1e-5
    return False

class PartitioningGraph:
    '''
    Creates leaves adjacency graph for a fully extended partitioning tree.
    '''
    def __init__(
            self, 
            tree: PartitioningTree
            ):
        # Fully extends tree if not done already
        tree.fully_extend()
        
        # Creates leaves = nodes from partitioning tree
        leaves = tree.get_leaves()
        self.graph = nx.Graph()

        # Finds all neighbors = edges
        for i in range(len(leaves)):
            self.graph.add_node(i)
            for j in range(i+1, len(leaves)):
                if are_neighbors(leaves[i], leaves[j]):
                    self.graph.add_edge(i,j)
    
    def display(self):
        plt.figure()
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color='purple',
            edge_color='grey',
            node_size=500,
            font_size=10
            )
        plt.title('Leaf Adjacency Graph')
        plt.show()