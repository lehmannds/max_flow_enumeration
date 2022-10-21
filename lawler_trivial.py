import numpy as np
from version1 import augment_flow, set_flow_on_edges, remove_flow_cycle, print_all_flow, set_constrained_edges, _max_number_for_search, _max_time_ms
import sys
import copy
import time
import networkx as nx


print_flow_details = False
total_flows = 0
_start = time.time_ns()

def print_all_flow(flow, constrained_edges, count = None):
    global total_flows
    global _start
    total_flows += 1
    
    
    if count == None:
         count = total_flows
    else:
        total_flows = count

    if total_flows <= 2:
        _start = time.time_ns() // 1000000

    time_passed = time.time_ns() // 1000000 - _start

    print(f"************** flow# {count} time per flow: {time_passed/count}***************")
    sys.stdout.flush()

    if not print_flow_details:
        return
    
    for u,v  in [e for e in constrained_edges if constrained_edges[e]]:
        print({"f": u, "t": v, "c":  20 })

    for u in flow:
        if u != new_source:
            for v in flow[u]:
                if v != new_target:
                    flow_value = flow[u][v]
                    if flow_value > 0:
                        print({"f": u, "t": v, "c":  int(flow_value)})



def get_flow_edges(flow):
    edges = []

    for u in flow:
        if u != new_source:
            for j in flow[u]:
                if j != new_target and flow[u][j] > 0:
                    edges.append((u, j))

    return edges

def set_constraints_on_graph(G, edge, is_include):
    
    G.remove_edge(edge[0], edge[1])
    if is_include:
        for new_edge in [(new_source, edge[1]), (edge[0], new_target)]:
        
            if new_edge not in G.edges:
                G.add_edge(new_edge[0], new_edge[1], capacity=20)
            else:
                G.edges[new_edge]["capacity"] += 20
            

def remove_constraints_from_graph(G, edge, is_include):
    
    G.add_edge(*edge, capacity=20)
    if is_include:
        G.edges[new_source, edge[1]]["capacity"] -= 20
        G.edges[edge[0], new_target]["capacity"] -= 20
        
    


new_source = -1
new_target = -2


def lawler(G, flow, source, target, flow_value, check_flow_cycle=False, **kwargs):
    total_flows = 0
    # print_all_flow(flow, [])
    
    _start = time.time_ns() // 1000000
    constrained_edges = {}
    required_flow = flow_value
    

    G.add_node(new_source) # source
    G.add_node(new_target) #target
    G.add_edge(target, new_target, capacity=flow_value)
    G.add_edge(new_source, source, capacity=flow_value)
    
    total_count = 1
    level_count = 1
    # constrained_edges = set()
    stack = [{"added_constraints": [], "done": False, "flow":flow}]
    while stack:
        current = stack[-1]

        if current["done"]:
            for c in current["added_constraints"]:
                remove_constraints_from_graph(G, c, current["added_constraints"][c])
                del constrained_edges[c]
                if current["added_constraints"][c]:
                    required_flow -= 20
            
            stack.pop()
            continue

        current["done"] = True
        
        for c in current["added_constraints"]:
            if c not in constrained_edges:
                set_constraints_on_graph(G, c, current["added_constraints"][c])
                constrained_edges[c] = current["added_constraints"][c]
                if current["added_constraints"][c]:
                    required_flow += 20
        
        edges = get_flow_edges(current["flow"])
        
        for i in reversed(range(len(edges))): # np.arange(len(edges)):
            new_constraints = {}
            failed = False

            included_edges = [edges[j] for j in np.arange(i)]
            excluded_edge = edges[i]

            for edge in included_edges:
                if edge not in constrained_edges:
                    set_constraints_on_graph(G, edge, True)
                    constrained_edges[edge] = True
                    new_constraints[edge] = True
                    required_flow += 20
            
            if excluded_edge not in constrained_edges:
                set_constraints_on_graph(G, excluded_edge, False)
                constrained_edges[excluded_edge] = False
                new_constraints[excluded_edge] = False
            elif constrained_edges[excluded_edge]:
                failed = True

            if not failed:
                mf = nx.algorithms.maximum_flow(G, new_source, new_target)
                if mf[0] != required_flow:
                    failed = True
                else:
                    fff = 1
                    # print_all_flow(mf[1], constrained_edges)

            if not failed:

                if check_flow_cycle:
                    flow_cycle = remove_flow_cycle(mf[1], detect_only=True, add_edges=[c for c in constrained_edges if constrained_edges[c]])
                    if not flow_cycle:
                        level_count += 1

                total_count += 1
                
                if total_count % 1000 == 0:
                    time_passed = time.time_ns() // 1000000 - _start
                    print(f"total_count {total_count} time per flow: {time_passed/total_count}")
                    sys.stdout.flush()
                
                ms_passed = time.time_ns()  // 1000000 - _start 
                if level_count >= _max_number_for_search or ms_passed > _max_time_ms:
                    print(f"total number of flows: {total_count}, no cycles: {level_count}")
                    return ms_passed, total_count, level_count
                
                stack.append(
                    {
                        "added_constraints": new_constraints,
                        "done": False,
                        "flow": mf[1]
                    })
                
            if new_constraints != None:
                for c in new_constraints:
                    remove_constraints_from_graph(G, c, constrained_edges[c])
                    if constrained_edges[c]:
                        required_flow -= 20
                    del constrained_edges[c]
                    
    print(f"number_of_flow_cycles {total_count - level_count}")
    print(f"no cycles {level_count}")
    
    return time.time_ns() // 1000000 - _start, total_count, level_count
