from version1 import print_all_flow, find_augmenting_cycle, get_cycle_capacity, augment_flow, remove_flow_cycle,set_constrained_edges,strongly_connected_components
import copy


_start = None

def search_max_flow(R, nodes, constrained_edges, flow, unit_capacity, current_flow_cycle = None):
    global total_flows

    # region find augmenting cycle

    found = False
    cycle_problem = False
    max_tries = 1
    bad_edges = []
    prev_cycle = None
    flow_cycle = None

    while not found and len(bad_edges) < max_tries:
        
        cycle = find_augmenting_cycle(R, nodes, constrained_edges, flow, bad_edges, flow_cycle=current_flow_cycle)
        
        if cycle == None:
            if len(bad_edges) == 0:
                return 0
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
        print_all_flow(flow2, 0, 0)

    added_edge = (cycle[-1], cycle[-2])
    
    set_constrained_edges(constrained_edges, added_edge)
    
    # endregion
    
    #region flow 1 branch
    components = list(strongly_connected_components(R, flow, nodes, constrained_edges))
    
    if len(components) > 1:
        print("* components:", [len(c) for c in components])

    idx = 0
    for c in components:
        search_max_flow(R, c, constrained_edges, flow, unit_capacity, current_flow_cycle)
        idx += 1

    #endregion

    # region flow 2 branch

    components2 = list(strongly_connected_components(R, flow2, nodes, constrained_edges))

    idx = 0
    
    if len(components2) > 1:
        print("* components:", [len(c) for c in components2])

    for c2 in components2:
        res1 = search_max_flow(R, c2, constrained_edges, flow2, unit_capacity, current_flow_cycle=flow_cycle)
        idx += 1

    # endregion
    
    #print(f"percent done {percent_done + percent} level: {level}")
    set_constrained_edges(constrained_edges, added_edge, False)
