from queue import Queue
import networkx as nx
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy import rand

def get_simple_test_graph():
    G = nx.DiGraph()
    capacity = 20
    size = 2
    total_size = 4 + size
    G.add_nodes_from(np.arange(total_size))
    G.add_edges_from([  (0, 1, {"capacity": capacity}),
                         (total_size-2, total_size-1, {"capacity": capacity})])
                        

    G.add_edges_from([(1, i + 2, {"capacity": capacity}) for i in np.arange(size)])
    G.add_edges_from([(i + 2, total_size-2, {"capacity": capacity}) for i in np.arange(size)])
                        
    return G

def get_multiple_components_test_graph(show_graph=False, number_of_branches = 10):
    G = nx.DiGraph()
    capacity = 20
    branch_size = 2
    
    total_size = number_of_branches + 3 + number_of_branches * branch_size
    G.add_nodes_from(np.arange(total_size))
    G.add_edges_from([  (0, 1, {"capacity": capacity}),
                         (total_size-2, total_size-1, {"capacity": capacity})])


    for j in np.arange(number_of_branches):
        before_node = 1 + j * (branch_size + 1)
        after_node = 1 + (j + 1) * (branch_size + 1)
        G.add_edges_from([(before_node, before_node + i + 1, {"capacity": capacity}) for i in np.arange(branch_size)])
        G.add_edges_from([(before_node + i + 1, after_node, {"capacity": capacity}) for i in np.arange(branch_size)])

    
    if show_graph:
        print_entire_graph(G)
    


    return G


def get_full_graph(size, capacity, show_graph=True):
    G = nx.DiGraph()
    
    G.add_nodes_from(np.arange(size))
    for i in np.arange(size):
        for j in np.arange(size):
            if i != j:
                G.add_edge(i, j, capacity=capacity)
    
    if show_graph:
        print_entire_graph(G)
    


    return G


def get_full_one_sided_graph(size, capacity, show_graph=True):
    G = nx.DiGraph()
    
    G.add_nodes_from(np.arange(size))
    for i in np.arange(size):
        for j in np.arange(size):
            if i < j:
                if r.random() > 0.5:
                    G.add_edge(i, j, capacity=capacity)
            elif i > j and (j, i) not in G.edges():
                G.add_edge(i, j, capacity=capacity)
    
    if show_graph:
        print_entire_graph(G)
    
    return G


def get_2_cycles_graph(cycle_length=10):
    G = nx.DiGraph()
    line_length = 5
    size = cycle_length * 2 + line_length
    G.add_nodes_from(np.arange(size))
    for i in np.arange(cycle_length):
        G.add_edge(i, (i + 1) % cycle_length, capacity = 20)
        G.add_edge(i + cycle_length, (i + 1) % cycle_length + cycle_length, capacity = 20)
    G.add_edge(0, cycle_length, capacity = 20)
    G.add_edge(cycle_length, 0, capacity = 20)

    for i in np.arange(line_length):
        n = cycle_length * 2 + i
        G.add_edge(n, n-1, capacity = 20)
        G.add_edge(n-1, n, capacity = 20)
    
    return G, size


def get_simple_test_graph2():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1,   2,3,4,    5,6])
    G.add_edges_from([  (0, 1, {"capacity": 30}),
                        (1, 2, {"capacity": 5}), 
                        (1, 3, {"capacity": 10}), 
                        (1, 4, {"capacity": 5}), 


                        (2, 5, {"capacity": 5}), 
                        (3, 5, {"capacity": 10}), 
                        (4, 5, {"capacity": 5}), 


                        (5, 6, {"capacity": 14})])
    return G


def get_rand_graph(size, avg_degree = 3, randomize_capacity= False):
    p = avg_degree / size
    G = nx.gnp_random_graph(p=p, n=size).to_directed()
    
    for e in G.edges():

        c = 10 if e[1] != 0 else 0
        if randomize_capacity:
            c = r.randint(1, 10) if e[1] != 0 else 0
        
        G.edges[e]["capacity"] = c

    return G

def get_rand_p_graph(size, p, capacity=None):
    
    G = nx.gnp_random_graph(p=p, n=size, directed=True)
    
    if capacity != None:
        for e in G.edges():
            G.edges[e]["capacity"] = capacity

    return G


def get_rand_local_graph(size, avg_degree = 3, local=10, unit_capacity= None, show_graph=False):
    p = avg_degree / size
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(size))
    for i in np.arange(size):
        start, end = i-local, i+local
        if start < 0:
            end -= start 
            start = 0 
            
        if end > size-1:
            start -= end - (size-1)
            end = size-1
            
        if start < 0:
            start = 0

        range = np.arange(start, end + 1)
         

        to_edges  = r.choices(range, k=avg_degree)
        for to in to_edges:
            if to == 0 or to == i:
                continue
            c = unit_capacity 
            if unit_capacity == None:
                c = r.randint(1, 10)
        
            if not G.has_edge(to, i):
                G.add_edge(i, to, capacity = c)

    return G

def get_rand_regular_graph(size, degree = 3, randomize_capacity= False):
    G = nx.random_regular_graph(d=degree, n=size).to_directed()

    edges = G.edges()

    for e in G.edges():

        c = 10
        if randomize_capacity:
            c = r.randint(1, 10) if e[1] != 0 else 0
            
        G.edges[e]["capacity"] = c

    return G


def get_rand_graph(size, avg_degree = 3, randomize_capacity= False):
    p = avg_degree / size
    G = nx.gnp_random_graph(p=p, n=size).to_directed()
    
    for e in G.edges():

        c = 10 if e[1] != 0 else 0
        if randomize_capacity:
            c = r.randint(1, 10) if e[1] != 0 else 0
        
        G.edges[e]["capacity"] = c

    return G

def cycle_flow_test(capacity):
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(5))

    G.add_edge(0,1, capacity=capacity)
    G.add_edge(0,3, capacity=capacity)
    G.add_edge(1,2, capacity=capacity)
    G.add_edge(2,3, capacity=capacity)
    G.add_edge(3,1, capacity=capacity)
    G.add_edge(3,4, capacity=capacity)

    print_entire_graph(G)

    return G

def print_entire_graph(G, to_file=False):
    nx.draw(G, with_labels=True)
    plt.draw()
    plt.savefig("graph.png")
    if not to_file:
        plt.show()
    

def load_DIMACS(file_name, unit_capacity = 20):
    G = nx.DiGraph()
    f = open(file_name)
    source = 0
    target = 0

    for line in f:
        parts = line.split(" ")
        if parts[0] == 'n':
            if parts[2].strip() == 's':
                source = int(parts[1])
            if parts[2].strip() == 't':
                target = int(parts[1])
        elif parts[0] == "a" or parts[0] == "e":
            from_node = int(parts[1])
            to_node = int(parts[2])
            capacity = 0
            if unit_capacity != None:
                capacity = unit_capacity
            elif len(parts) > 3:
                capacity = int(parts[3])
            if capacity > 0:
                if from_node not in G:
                    G.add_node(from_node) 
                if to_node not in G:
                    G.add_node(to_node) 
                G.add_edge(from_node, to_node, capacity = capacity)

    if source == 0:
        source = np.min(G.nodes)
    if target == 0:
        target = np.max(G.nodes)

    bad_edges = []
    for e in G.edges():
        if e[1] == source:
            bad_edges.append(e)
        if e[0] == target:
            bad_edges.append(e)

    for e in bad_edges:
        G.remove_edge(e[0], e[1])
    

    return G, source, target


def load_SNAP(file_name, unit_capacity = 20):
    G = nx.DiGraph()
    f = open(file_name)
    nodes = set()
    edges = set()
    source = 0e10
    target = 0
    adj_list = {}

    for line in f:
        parts = line.split()
        if parts[0] == '#':
            continue
        else:
            from_node = int(parts[0])
            to_node = int(parts[1])
            capacity = 0
            if unit_capacity != None:
                capacity = unit_capacity
            elif len(parts) > 3:
                capacity = int(parts[3])
            if capacity > 0:
                if from_node not in nodes:
                    nodes.add(from_node) 
                    adj_list[from_node] = set()
                    
                if to_node not in nodes:
                    nodes.add(to_node)
                    adj_list[to_node] = set()

                adj_list[from_node].add(to_node)
                    

    if source == 0:
        source = np.min([n for n in nodes])
    if target == 0:
        target = np.max([n for n in nodes])

    visited = set()
    visited.add(source)
    bfs_queue = Queue()
    bfs_queue.put(source)
    while not bfs_queue.empty():
        from_node = bfs_queue.get()
        for to_node in adj_list[from_node]:
            if not to_node in visited:
                visited.add(to_node)
                if to_node != target:
                    bfs_queue.put(to_node)
            if to_node != source and (to_node, from_node) not in edges:
                edges.add((from_node, to_node))


    # bad_edges = []
    # for e in edges:
    #     if e[1] == source:
    #         bad_edges.append(e)
    #     if e[0] == target:
    #         bad_edges.append(e)

    # for e in bad_edges:
    #     edges.remove((e[0], e[1]))

    G.add_nodes_from(nodes)
    G.add_edges_from([(e[0], e[1], {"capacity": capacity}) for e in edges])
    

    return G, source, target

