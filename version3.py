from platform import node
import queue
from version1 import print_all_flow, find_augmenting_cycle, get_cycle_capacity, augment_flow, remove_flow_cycle,set_constrained_edges,strongly_connected_components
import copy


_start = None

def search_max_flow(R, start_nodes, constrained_edges, flow, unit_capacity, current_flow_cycle = None):
    global total_flows
    stack = [{"flow_cycles": [], "added_constraints": [], "nodes": None, "done": False, "flow_cycle": None}]
    max_tries = 3
    current_nodes = start_nodes
    while stack:

        current = stack[-1]
        if current["nodes"] != None:
            current_nodes = current["nodes"]
        if current["done"]:
            for edge in current["added_constraints"]:
                set_constrained_edges(constrained_edges, edge, False)
            for cycle1 in current["flow_cycles"]:
                augment_flow(flow, cycle1, -unit_capacity)
            stack.pop()
            continue
        found = False
        bad_edges = []
        prev_cycle = None
        
            
        cycle = find_augmenting_cycle(R, current_nodes, constrained_edges, flow, bad_edges, flow_cycle=current["flow_cycle"])
        
        if cycle == None:
            current["done"] = True
            continue

        added_edge = (cycle[0], cycle[-1])
        stack[-1]["added_constraints"].append(added_edge)
        set_constrained_edges(constrained_edges, added_edge)
        if len(stack) % 25 == 0:
            print(f"** stack: {len(stack)} nodes:{len(current_nodes)}")
        #region flow 1 branch
        
        
        components = list(strongly_connected_components(R, flow, current_nodes, constrained_edges))
        
        idx = 0
        if len(components) > 1:
            current["done"] = True
            current["nodes"] = current_nodes
        
            for c in components:
                
                el = {  "flow_cycles": [], 
                        "added_constraints": [], 
                        "nodes": c,
                        "done": False,
                        "flow_cycle": current["flow_cycle"]
                    }
                stack.append(el)
        elif len(components) == 1:
            changed = len(components[0]) < len(current_nodes)
            if changed:
                if len(stack) > 1 and stack[-2]["nodes"] == None:
                    stack[-2]["nodes"] = current_nodes
                current["nodes"] = components[0]
                    
        else:
            current["done"] = True
            
        #endregion
        

        # region flow 2 branch

        capacity = get_cycle_capacity(R, cycle, flow) if unit_capacity is None else unit_capacity
        
        augment_flow(flow, cycle, capacity)
        cycles = [cycle]
        flow_cycle = remove_flow_cycle(flow, True, constrained_edges, augmented_cycles=cycles)
        
        if flow_cycle == None:
            print_all_flow(flow, 0, 0)


        components2 = list(strongly_connected_components(R, flow, current_nodes, constrained_edges))

        idx = 0       
        for c2 in components2:
            changed = len(c2) < len(current_nodes)
            el = {  "flow_cycles": cycles if idx == 0 else [], 
                        "added_constraints": [], 
                        "nodes": c2 if changed else None, 
                        "done": False,
                        "flow_cycle": flow_cycle
                    }
            if changed and stack[-1]["nodes"] == None:
                stack[-1]["nodes"] = current_nodes
            stack.append(el)
            idx += 1

        if len(components2) == 0:
            for cycle1 in cycles:
                augment_flow(flow, cycle1, -unit_capacity)
        

        
