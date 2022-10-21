from platform import node
import queue
from utils import strongly_connected_components
from version1 import print_all_flow, find_augmenting_cycle, get_cycle_capacity, augment_flow, remove_flow_cycle,set_constrained_edges, _max_number_for_search, _max_time_ms
import copy
import time



def search_max_flow(R, start_nodes, constrained_edges, flow, unit_capacity, always_do_components = False, connected=False, **kwargs):
    _start = time.time_ns() // 1000000
    total_count = 1
    stack = [{"connected": connected, "flow_cycles": [], "added_constraints": [], 
                "nodes": None, "done": False, "flow_cycle": None, "do_components": always_do_components}]
    number_of_flow_cycles = 0    
    max_tries = 3
    current_nodes = start_nodes or R.nodes
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
        
        do_components = current["do_components"]

        if not do_components:
            cycle = find_augmenting_cycle(R, current_nodes, constrained_edges, flow, bad_edges, flow_cycle=current["flow_cycle"])
            
            if cycle == None:
                if current["connected"]:
                    current["done"] = True
                    continue
                else:
                    do_components = True


        if not do_components:
            added_edge = (cycle[0], cycle[-1])
            stack[-1]["added_constraints"].append(added_edge)
            stack[-1]["connected"] = False
            set_constrained_edges(constrained_edges, added_edge)
            if len(stack[-1]["added_constraints"]) % 25 == 0:
                print(f"** stack: {len(stack[-1]['added_constraints'])} nodes:{len(current_nodes)}")
        #region flow 1 branch
        
        components = [current_nodes]
        if do_components:
            components = strongly_connected_components(R, flow, current_nodes, constrained_edges)
            

        
        idx = 0
        if do_components:
            current["done"] = True
            current["nodes"] = current_nodes
            components_count = 0
            for c in components:
                components_count += 1
                el = {  "flow_cycles": [], 
                        "added_constraints": [], 
                        "nodes": c,
                        "done": False,
                        "flow_cycle": current["flow_cycle"],
                        "connected": do_components,
                        "do_components": always_do_components and not do_components
                    }
                stack.append(el)

            if components_count > 1:
                pass
                # print(f"components_count {components_count}")
        # else:
        #     changed = len(components[0]) < len(current_nodes)
        #     if changed:
        #         if len(stack) > 1 and stack[-2]["nodes"] == None:
        #             stack[-2]["nodes"] = current_nodes
        #         current["nodes"] = components[0]
        #     elif do_components:
        #         current["done"] = True    

                    
        
            
        #endregion

        if do_components:
            continue


        # region flow 2 branch

        capacity = get_cycle_capacity(R, cycle, flow) if unit_capacity is None else unit_capacity
        
        augment_flow(flow, cycle, capacity)
        cycles = [cycle]
        flow_cycle = remove_flow_cycle(flow, True, constrained_edges, augmented_cycles=cycles)
        # flow_cycle = None

        total_count += 1
        if flow_cycle == None:
            if total_count % 1000 == 0:
                print_all_flow(flow, 0, 0, total_count)
                # print("flow count:", total_count)
        else:
            number_of_flow_cycles += 1
            # print("flow cycle exists.")

        ms_passed = time.time_ns() // 1000000 - _start 
        if total_count - number_of_flow_cycles >= _max_number_for_search  or ms_passed > _max_time_ms:
            print(f"{total_count} flows, number_of_flow_cycles={number_of_flow_cycles}")
            return ms_passed, total_count, total_count - number_of_flow_cycles


        #components2 = list(strongly_connected_components(R, flow, current_nodes, constrained_edges))
        components2 = [current_nodes]

        idx = 0       
        for c2 in components2:
            changed = len(c2) < len(current_nodes)
            el = {  "flow_cycles": cycles if idx == 0 else [], 
                        "added_constraints": [], 
                        "nodes": c2 if changed else None, 
                        "done": False,
                        "connected": False,
                        "do_components": always_do_components,
                        "flow_cycle": flow_cycle
                    }
            if changed and stack[-1]["nodes"] == None:
                stack[-1]["nodes"] = current_nodes
            stack.append(el)
            idx += 1

        if len(components2) == 0:
            for cycle1 in cycles:
                augment_flow(flow, cycle1, -unit_capacity)
        
    print(f"number_of_flow_cycles {number_of_flow_cycles}")
    return time.time_ns() // 1000000 - _start, total_count, total_count - number_of_flow_cycles
        
