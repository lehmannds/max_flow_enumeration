import numpy as np
from version1 import augment_flow, set_flow_on_edges, remove_flow_cycle, print_all_flow, set_constrained_edges, _max_number_for_search, _max_time_ms
import sys
import copy
import time

def get_flow_edges(flow, cycle_to_break):
    edges = []

    if cycle_to_break:
        c = len(cycle_to_break)
        for i in range(c):
            edges.append((cycle_to_break[(i + 1) % c], cycle_to_break[i]))

    else:
        for u in flow:
            for j in flow[u]:
                if flow[u][j] > 0:
                    edges.append((u, j))

    return edges

def lawler(R, flow, unit_capacity, timeout=20, check_flow_cycle=True, **kwargs):
    _start = time.time_ns() // 1000000
    _max_time_ms = timeout * 1000
    
    total_count = 1
    level_count = 1
    constrained_edges = set()
    stack = [{"flow_cycles": [], "added_constraints": [], "done": False, "cycle_to_break": None}]
    while stack:
        current = stack[-1]

        if current["done"]:
            for c in current["added_constraints"]:
                set_constrained_edges(constrained_edges, c, False)
            for cycle in current["flow_cycles"]:
                augment_flow(flow, cycle, -unit_capacity)
            stack.pop()
            continue

        current["done"] = True

        for c in current["added_constraints"]:
            set_constrained_edges(constrained_edges, c, True)
        for cycle in current["flow_cycles"]:
            augment_flow(flow, cycle, unit_capacity)

        edges = get_flow_edges(flow, current["cycle_to_break"])
        
        for i in reversed(range(len(edges))): # np.arange(len(edges)):
            
            included_edges = [edges[j] for j in np.arange(i)]
            excluded_edges = [edges[i]]
            
            success, new_constraints, all_cycles = set_flow_on_edges(R, None, flow, included_edges, excluded_edges, constrained_edges, unit_capacity)
            if success:
                total_count += 1
                if total_count % 1000 == 0:
                    time_passed = time.time_ns() // 1000000 - _start
                    print(f"total_count {total_count} time per flow: {time_passed/total_count}")
                    sys.stdout.flush()
                
                flow_cycle = check_flow_cycle and remove_flow_cycle(flow, detect_only=True)
                if not flow_cycle:
                    level_count += 1
                    # print_all_flow(flow, 0, 0, level_count)
                else:
                    pass
                    # print("flow cycle exists")
                    # sys.stdout.flush()

                ms_passed = time.time_ns()  // 1000000 - _start 
                if level_count >= _max_number_for_search or ms_passed > _max_time_ms:
                    print(f"total number of flows: {total_count}, no cycles: {level_count}")
                    return ms_passed, total_count, level_count
                stack.append(
                    {
                        "flow_cycles": all_cycles,
                        "cycle_to_break": flow_cycle,
                        "added_constraints": new_constraints,
                        "done": False
                    })
                
            if new_constraints != None:
                for c in new_constraints:
                    set_constrained_edges(constrained_edges, c, False)
            for cycle in all_cycles:
                augment_flow(flow, cycle, -unit_capacity)

    print(f"number_of_flow_cycles {total_count - level_count}")
    return time.time_ns() // 1000000 - _start, total_count, level_count
