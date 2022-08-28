from cProfile import label
from cmath import sqrt
from math import fabs
from pickle import STOP
from platform import node
from tabnanny import verbose
from xml.etree.ElementPath import find
import networkx as nx
from networkx.algorithms.flow import build_residual_network
import random as r
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy import rand 
from scipy.optimize import linprog
from pulp import LpProblem, lpSum, LpVariable, LpMinimize, value
from itertools import chain
from utils import get_nodes_from_edges
import copy
import math
from utils import *
from utils import _bidirectional_shortest_path
from test_graphs import *
from test_solvers import test_pypoman
from datetime import datetime


#sys.stdout = open('stdout.txt', 'w')
total_flows = 0
verbose_print = False

def test_biconnected():
    size = 15
    G, size = get_2_cycles_graph(size)
    #connected = np.hstack((np.arange(0, 5), np.arange(6, size * 2)))
    connected = np.arange(0, size)
    for c in biconnected_dfs(G, connected):
        print(c)


def search_max_flow(R, nodes, constrained_edges, flow, unit_capacity, level=0, percent = 100, percent_done = 0, check_for_cycle=False, c_idx=1):
    global total_flows

    if check_for_cycle:
        flow_cycle = remove_flow_cycle(flow, True, constrained_edges)
        if flow_cycle:
            search_break_cycle(R, constrained_edges, nodes, flow_cycle, flow, unit_capacity, level, percent, percent_done)
            return 
        print_all_flow(flow, percent + percent_done, level)

    found = False
    cycle_problem = False
    max_tries = 3
    bad_edges = []
    prev_cycle = None

    while not found and len(bad_edges) < max_tries:
        
        cycle = find_augmenting_cycle(R, nodes, constrained_edges, flow, bad_edges)
        
        if cycle == None:
            if len(bad_edges) == 0:
                # if verbose_print:
                #     print("level:", str(level) + ".1")
                #     print_edges(nodes, [], flow)
                
                
                
                return 2
            else:
                cycle = prev_cycle
                break
        
        capacity = get_cycle_capacity(R, cycle, flow) if unit_capacity is None else unit_capacity
        
        flow2 = copy.deepcopy(flow)
        augment_flow(flow2, cycle, capacity)

        flow_cycle = remove_flow_cycle(flow2, True, constrained_edges)
        if flow_cycle != None:
            bad_edges.append((cycle[-1], cycle[0]))
            prev_cycle = cycle
            cycle_problem = True
        else:
            found = True

    if found:
        print_all_flow(flow2, percent + percent_done, level)

    added_edge = (cycle[-1], cycle[0])
    # for i in np.arange(len(cycle)):
    #     if flow[cycle[i]][cycle[(i+1)%len(cycle)]] != 0:
    #         added_edge = (cycle[i], cycle[(i+1)%len(cycle)])
    #         break
    # if added_edge == None:
    #     added_edge = (cycle[0], cycle[1])


    set_constrained_edges(constrained_edges, added_edge)
    
    components = list(strongly_connected_components(R, flow, nodes, constrained_edges))
            

    if verbose_print:
        
        print("level:", str(level) + ".1")
        print_edges(nodes, components, flow, cycle)


    if len(components) > 1:
        print("* components:", [len(c) for c in components])

    idx = 0
    for c in components:
        
        p_done = percent_done + percent / 2 / len(components) * idx

        search_max_flow(R, c, constrained_edges, flow, unit_capacity, level + 1, 
                                    percent / 2 / len(components), p_done, c_idx=idx+c_idx)
        
        idx += 1

    

    if not found and cycle_problem:
        search_break_cycle(R, constrained_edges, nodes, flow_cycle, flow2, unit_capacity, level, percent/2, percent_done + percent/2)
        set_constrained_edges(constrained_edges, added_edge, False)
        
        return

    components2 = list(strongly_connected_components(R, flow2, nodes, constrained_edges))

    if verbose_print:
        
        print("level:", str(level) + ".2")
        print_edges(nodes, components, flow2, cycle)

    idx = 0
    
    if len(components2) > 1:
        print("* components:", [len(c) for c in components2])

    for c2 in components2:
        p = percent / 2 * (1 / len(components2))    
        
        p_done = percent_done + percent / 2 + percent / 2 * (idx / len(components2))
        res1 = search_max_flow(R, c2, constrained_edges, flow2, unit_capacity, level + 1, p, p_done, c_idx=idx)


        idx += 1

        # TODO REMOVE
        if res1 == 2 and c2 == {15,17,18}:
            #print_graph(R, nodes, constrained_edges, flow2)
            components2 = list(strongly_connected_components(R, flow2, nodes, constrained_edges))

    
    
    #print(f"percent done {percent_done + percent} level: {level}")
    set_constrained_edges(constrained_edges, added_edge, False)

def search_break_cycle(R, constrained_edges, nodes, flow_cycle, flow, unit_capacity, level, percent, percent_done):
    
    for i in np.arange(len(flow_cycle)):
        included_edges = [(flow_cycle[i+1], flow_cycle[i]) for i in np.arange(i)]
        excluded_edges = [(flow_cycle[(i+1)% len(flow_cycle)], flow_cycle[i])]
        success, new_constraints = set_flow_on_edges(R, nodes, flow, included_edges, excluded_edges, constrained_edges, unit_capacity)
        if success:
            search_max_flow(R, nodes, constrained_edges, flow, unit_capacity, level+1, percent, percent_done, True, 0)
        if new_constraints != None:
            for c in new_constraints:
                set_constrained_edges(constrained_edges, c, False)

        
def set_constrained_edges(constrained_edges, e, is_add=True):
    if is_add:
        constrained_edges.add(e)
        constrained_edges.add((e[1], e[0]))
    else:
        constrained_edges.remove(e)
        constrained_edges.remove((e[1], e[0]))

def set_flow_on_edges(G, nodes, flow, included_edges, excluded_edges, constrained_edges, unit_capacity):
    new_constraints = set()
    for e in included_edges:
        if flow[e[0]][e[1]] > 0 and e not in constrained_edges:
            new_constraints.add(e)
            set_constrained_edges(constrained_edges, e)
            
    for e in excluded_edges:
        if flow[e[0]][e[1]] <= 0 and e not in constrained_edges:
            new_constraints.add(e)
            set_constrained_edges(constrained_edges, e)


    for e in included_edges:
        while flow[e[0]][e[1]] <= 0:
            if e in constrained_edges:
                return False, new_constraints
            cycle = find_augmenting_cycle(G, nodes, constrained_edges, flow, [], e)
            if cycle == None:
                return False, new_constraints
            capacity = get_cycle_capacity(G, cycle, flow) if unit_capacity is None else unit_capacity
            augment_flow(flow, cycle, capacity)
            if e not in constrained_edges:
                set_constrained_edges(constrained_edges, e)
                new_constraints.add(e)
    for e in excluded_edges:
        while flow[e[0]][e[1]] > 0:
            if e in constrained_edges:
                return False, new_constraints
            cycle = find_augmenting_cycle(G, nodes, constrained_edges, flow, [], (e[1], e[0]))
            if cycle == None:
                return False, new_constraints
            capacity = get_cycle_capacity(G, cycle, flow) if unit_capacity is None else unit_capacity
            augment_flow(flow, cycle, capacity)
            if e not in constrained_edges:
                set_constrained_edges(constrained_edges, e)
                new_constraints.add(e)

    return True, new_constraints

def print_edges(all_nodes, components, flow, cycle = None):
    
    if not verbose_print:
        return

    if cycle != None:
        f = flow[cycle[0]][cycle[1]]
        if f > 0:
            print({"f": cycle[0], "t": cycle[1], "c": f})   
        else:
            f = flow[cycle[1]][cycle[0]]
            if f > 0:
                print({"f": cycle[1], "t": cycle[0], "c": f})   

    for from_node in all_nodes:
        for to_node in flow[from_node]:
            if to_node in all_nodes and all(from_node not in c or to_node not in c for c in components) and flow[from_node][to_node] > 0:
                print({"f": from_node, "t": to_node, "c": flow[from_node][to_node]})   
    

def get_capacity(e, flow):
    return e["capacity"] - flow[e[0]][e[1]]

_start = None

def print_all_flow(flow, percent_done, level, count = None):
    global total_flows
    global _start
    total_flows += 1
    
    
    if count == None:
         count = total_flows
    else:
        total_flows = count

    if total_flows == 1:
        _start = time.time_ns() // 1000000

    time_passed = time.time_ns() // 1000000 - _start

    print(f"************** flow# {count} time per flow: {time_passed/count}***************")
    sys.stdout.flush()
    return
    #print(f"percent done {percent_done} level: {level}")
    for u in flow:
        for v in flow[u]:
            
            flow_value = flow[u][v]
            if flow_value > 0:
                print({"f": u, "t": v, "c":  float(flow_value)})


def print_flow(flow_dir, c):
    for u in c:
        for v in flow_dir[u]:
            if v in c:
                flow = flow_dir[u][v]
                if flow > 0:
                    print({"f": u, "t": v, "flow": flow})

def augment_flow(flow, cycle, capacity):
    l = len(cycle)
    for i in np.arange(l):
        flow[cycle[i]][cycle[(i+1)%l]] += capacity
        if cycle[(i+1)%l] not in flow:
            flow[cycle[(i+1)%l]] = {}
        if cycle[i] not in flow[cycle[(i+1)%l]]:
            flow[cycle[(i+1)%l]][cycle[i]] = - flow[cycle[i]][cycle[(i+1)%l]]    
        flow[cycle[(i+1)%l]][cycle[i]] -= capacity

def get_cycle_capacity(R, cycle, flow):
    
    l = len(cycle)
    c = get_capacity(R[cycle[0]][cycle[1]], flow)
    for i in np.arange(1, l):
        c = min(c, get_capacity(R[cycle[i]][cycle[(i+1)%l]]), flow)

    return c

def find_augmenting_cycle(R, nodes, constranied_edges, flow, bad_edges, edge = None, flow_cycle=None):
    
    if edge == None:
        edge = find_residual_edge(flow, nodes, constranied_edges, bad_edges, flow_cycle)
    
    if edge == None:
        return None

    path = _bidirectional_shortest_path(R, edge[1], edge[0], flow, nodes, constranied_edges)
    
    if path == None:
        #print("problem no path found")
        return None

    return path[1]

def find_residual_edge(flow, nodes, constrained_edges, bad_edges, flow_cycle=None):
    if flow_cycle == None:
        for u in nodes:
            for v in flow[u]:
                if v in nodes and (u,v) not in constrained_edges and flow[u][v]  < 0 and (u,v) not in bad_edges:
                    return (u,v)
    else:
        for i in np.arange(len(flow_cycle)):
            edge = (flow_cycle[i], flow_cycle[(i+1)%len(flow_cycle)])
            if edge[0] in nodes and edge[1] in nodes and edge not in constrained_edges and edge not in bad_edges:
                return edge
                
    return None


def print_graph(G, nodes, ignore_edges, flow):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for i in nodes:
        for to in G[i]:
            if to in nodes and (i, to) not in ignore_edges and (G[i][to]["capacity"] - flow[i][to] > 0):
                graph.add_edge(i, to)
            if to in nodes and (to, i) not in ignore_edges and flow[i][to] > 0:
                graph.add_edge(to, i)

    nx.draw(graph, with_labels=True)
    plt.draw()
    plt.show()
    


def remove_flow_cycle(flow, check_constraints = False, constrained_edges = None, detect_only=False, augmented_cycles=[]):
    
    visited = set()
    in_stack = set()
    stack = []
    found_cycle = False

    for f in flow:
        if f not in visited:
            queue = [f]
            

            while queue:
                
                current = queue[-1]

                
                visited.add(current)
                in_stack.add(current)
                stack.append(current)
                for to in flow[current]:
                    if current in flow and to in flow[current] and flow[current][to] > 0:




                        if to in in_stack:
                            if detect_only:
                                return True
                            #handle cycle
                            prev = current
                            next = to
                            f = flow[prev][next]
                            idx = 2
                            while True:
                                f = min(f, flow[prev][next])
                                if prev == to:
                                    break
                                next = prev
                                prev = stack[-idx]
                                idx += 1
                            prev = current
                            next = to
                            cycle = []
                            cycle_is_constrained = False

                            while True:
                                if check_constraints:
                                    cycle.append(prev)
                                    if (prev,next) in constrained_edges:
                                        cycle_is_constrained = True
                                else:
                                    flow[prev][next] -= f
                                    flow[next][prev] += f
                                
                                if prev == to:
                                    current = to
                                    found_cycle = True
                                    while queue[-1] != current:
                                        queue.pop()
                                    if check_constraints:
                                        if cycle_is_constrained:
                                            return cycle
                                        else:
                                            augment_flow(flow, cycle, f)
                                            if augmented_cycles != None:
                                                augmented_cycles.append(cycle)
                                    break

                                
                                in_stack.remove(prev)
                                visited.remove(prev)    
                                stack.pop()
                                next = prev
                                prev = stack[-1]
                    

                        elif to not in visited:    
                            queue.append(to)
                        
                    if found_cycle:
                        break

                if found_cycle:
                    found_cycle = False
                    stack.pop()
                    continue
                while len(stack) > 0 and stack[-1] == queue[-1]:
                    in_stack.remove(queue[-1])
                    queue.pop()
                    stack.pop()

def flow_from_vertex(G, vertex):
    flow = {}
    idx = 0
    for e in G.edges():
        if vertex[idx] > 0:
            if e[0] not in flow:
                flow[e[0]] = {}
            if e[1] not in flow:
                flow[e[1]] = {}
            if e[1] not in flow[e[0]]:
                flow[e[0]][e[1]] = vertex[idx]
            
        idx += 1

    return flow


def print_vertices(G, vertices, filter):
    idx2 = 1
    for flow in vertices:
        
        idx = 0

        if filter:
            vertex_flow = flow_from_vertex(G, flow)
            if remove_flow_cycle(vertex_flow, detect_only=True):
                continue
        print(f"************** flow# {idx2}***************")
        for e in G.edges():


            if flow[idx] > 0:
                print({"f": e[0], "t": e[1], "c": flow[idx]})   
            idx += 1

        idx2 += 1

def get_flow_edges(flow):
    edges = []
    for u in flow:
        for j in flow[u]:
            if flow[u][j] > 0:
                edges.append((u, j))

    return edges
total_count = 0
def lawler(R, flow, constrained_edges, unit_capacity, count = 1):
    
    level_count = 0
    total_count = 0
    edges = get_flow_edges(flow)
    
    for i in np.arange(len(edges)):
        flow2 = copy.deepcopy(flow)
        included_edges = [edges[j] for j in np.arange(i)]
        excluded_edges = [edges[i]]
        
        success, new_constraints = set_flow_on_edges(R, None, flow2, included_edges, excluded_edges, constrained_edges, unit_capacity)
        if success:
            total_count += 1
            if not remove_flow_cycle(flow2, detect_only=True):
                level_count += 1
                print_all_flow(flow2, 0, 0, count + level_count)
            else:
                print("flow cycle exists")
                sys.stdout.flush()
                
            level_count1, total_count1 = lawler(R, flow2, constrained_edges, unit_capacity, count + level_count)
            level_count += level_count1
            total_count += total_count1
        if new_constraints != None:
            for c in new_constraints:
                set_constrained_edges(constrained_edges, c, False)

    return level_count, total_count
