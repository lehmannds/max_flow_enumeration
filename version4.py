from itertools import count
import uuid
from platform import node
import queue
from utils import strongly_connected_components
from version1 import print_all_flow, find_augmenting_cycle, get_cycle_capacity, augment_flow, remove_flow_cycle,set_constrained_edges, _max_number_for_search, _max_time_ms
import copy
import time

_time = None
from datetime import datetime
import sys

def print_time(doing = None):
    global _time
    current = datetime.now()
    if _time is not None and doing:
        print(int((current - _time).total_seconds() * 1000), "ms", "doing:", doing)

    _time = current



def search_max_flow(R, start_nodes, constrained_edges, flow, unit_capacity, always_do_components = False, connected=False, timeout=20, check_flow_cycle=True, **kwargs):
    _start = time.time_ns() // 1000000
    _max_time_ms = timeout * 1000
    counts = [["Total", 0]]

    total_count = 1
    stack = [{"connected": connected, "flow_cycles": [], "added_constraints": [], 
                "nodes": None, "done": False, "flow_cycle": None, "do_components": True}]
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
            guid = current.get("guid", None)
            add_to_count = current.get("count", False)
            if not guid:
                if add_to_count:
                    counts[-1][1] += 1
            else:
                idx_component = current.get("idx_component")
                counts[-idx_component - 1][1] *=  counts[-1][1]
                counts.pop()
                
                is_first = idx_component == 1
                if is_first:
                    counts[-2][1] += counts[-1][1]
                    counts.pop()
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
                    current["count"] = True
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
            components = list(components)
            
            sys.stdout.flush()

            if len(components) == 0:
                current["count"] = True
            # if len(components) > 1:
            #     print(f"components: {[len(c) for c in components]}")

        
        idx = 0
        if do_components:
            current["done"] = True
            current["nodes"] = current_nodes
            components_count = 0
            parent_guid = str(uuid.uuid4())
            if len(components) > 0:
                counts.append([parent_guid, 1])
            idx_component = 1
            for c in components:
                guid = str(uuid.uuid4())
                counts.append([guid, 1])
                components_count += 1
                el = {  "flow_cycles": [], 
                        "added_constraints": [], 
                        "nodes": c,
                        "idx_component": idx_component,
                        "done": False,
                        "parent_guid": parent_guid,
                        "guid": guid,
                        "flow_cycle": current["flow_cycle"],
                        "connected": do_components,
                        "do_components": always_do_components and not do_components
                    }
                idx_component += 1
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

        capacity = unit_capacity or get_cycle_capacity(R, cycle, flow)
        
        augment_flow(flow, cycle, capacity)
        cycles = [cycle]
        if check_flow_cycle:
            flow_cycle = remove_flow_cycle(flow, True, constrained_edges, augmented_cycles=cycles, detect_only=True)
        else:
            flow_cycle = None

        total_count += 1
        if total_count % 2000 == 0:
            print_all_flow(flow, 0, 0, total_count)
            sys.stdout.flush()
        if not flow_cycle:
            pass
                # print("flow count:", total_count)
        else:
            number_of_flow_cycles += 1
            # print("flow cycle exists.")

        ms_passed = time.time_ns() // 1000000 - _start 
        if total_count - number_of_flow_cycles >= _max_number_for_search  or ms_passed > _max_time_ms:
            total_with_join = get_total_count(counts)
            print(counts)
            print(f"{total_count} flows, number_of_flow_cycles={number_of_flow_cycles} total_with_join: {total_with_join}")
            return ms_passed, total_with_join,   - number_of_flow_cycles


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

    print(counts) 
    print(f"number_of_flow_cycles {number_of_flow_cycles}")
    
    return time.time_ns() // 1000000 - _start, get_total_count(counts), total_count - number_of_flow_cycles
        
def get_total_count(counts):
    return sum([c[1] for c in counts])