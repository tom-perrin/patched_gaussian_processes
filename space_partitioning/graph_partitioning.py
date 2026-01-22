import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tree_partitioning import PartitioningTree

# --------------------------------------------------

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
    Computes the 2 end points of the frontier segment of a node's split, clipped by the previous frontiers
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


def get_shared_frontier(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree,
        ):
    '''
    Computes the 2 end points of the shared frontier between two regions (if it exists)
    '''
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
    
    if lca is None or lca.pdir is None: # no common ancestor
        return None
    
    # Gets consraints from both leaves and creates associated frontier
    combined_constraints = get_constraints(leaf_a) + get_constraints(leaf_b)
    segment = get_clipped_frontier(lca, external_constraints=combined_constraints)

    # Returns end points
    if segment is not None:
        p1, p2 = segment
        if np.linalg.norm(p1 - p2) > 1e-5:
            return segment
    return None


def are_neighbors(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree
        ):
    '''
    Tells if the regions corresponding to two leaves are neighbors i.e. share a common frontier
    '''
    return get_shared_frontier(leaf_a, leaf_b) is not None


class PartitioningGraph:
    '''
    Leaves adjacency graph for a fully extended partitioning tree.

    Parameters
    ----------
    tree : PartitioningTree
        Corresponding partitioning tree.

    Attributes
    ----------
    nodes : list[PartitioningTree]
        List of all leaves of the initial tree.
    G : networkx.Graph
        Adjacency graph of the leaves.
    
    Methods
    ----------
    frontier : Computes the 2 end points of the frontier between i and j (if they are neighbors).
    adjacency_matrix : Computes the adjacency matrix of the regions.
    frontier_matrix : Computes the matrix of all frontiers between regions (both end points if they exist).
    '''
    def __init__(
            self, 
            tree: PartitioningTree
            ):
        # Fully extends tree if not done already
        tree.fully_extend()
        
        # Creates nodes list and empty graph
        self.nodes = tree.get_leaves()
        self.G = nx.Graph()

        # Finds all neighbors = edges for each node 
        for i in range(len(self.nodes)):
            self.G.add_node(i)
            for j in range(i+1, len(self.nodes)):
                segment = get_shared_frontier(self.nodes[i], self.nodes[j])
                if segment is not None:
                    self.G.add_edge(i, j, geometry=segment) # end points of frontier segment stored in G[i][j]['geometry']
    

    def frontier(self, i:int, j:int):
        '''
        Computes the 2 end points of the frontier between i and j (if they are neighbors)
        '''
        if self.G.has_edge(i, j):
            return self.G[i][j]['geometry']
        return None
    
    @property
    def adjacency_matrix(self):
        '''
        Computes the adjacency matrix of the regions
        '''
        return nx.to_numpy_array(self.G)
    

    @property
    def frontier_matrix(self):
        '''
        Computes the matrix of all frontiers between regions (both end points if they exist)
        '''
        N = len(self.nodes)
        matrix = [
            [self.G[i][j]['geometry'] if self.G.has_edge(i, j) else None 
            for j in range(N)] 
            for i in range(N)]
        return matrix