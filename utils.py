from itertools import chain
import sys

def get_nodes_from_edges(edges):
    nodes = set()
    for e in edges:
        nodes.add(e[0])
        nodes.add(e[1])

    return nodes



def _bidirectional_pred_succ(G, source, target, flow, included_nodes=None, ignore_edges=None):
    """Bidirectional shortest path helper.
    Returns (pred,succ,w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    
    add_ignore = None
    if not ignore_edges:
        ignore_edges = set()
    if (source, target) not in ignore_edges:
        add_ignore = (source, target) 
        ignore_edges.add(add_ignore)
    
    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if included_nodes:

        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w in included_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if G._adj[w][v]["capacity"] - flow[w][v] > 0:
                            if (w, v) not in ignore_edges:
                                yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        
                        if G._adj[v][w]["capacity"] - flow[v][w] > 0:
                            if (v, w) not in ignore_edges:
                                yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:

            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    # predecesssor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc(v):
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        # found path
                        if add_ignore != None:
                            ignore_edges.remove(add_ignore)
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred(v):
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        # found path
                        if add_ignore != None:
                            ignore_edges.remove(add_ignore)
                        return pred, succ, w

    if add_ignore != None:
        ignore_edges.remove(add_ignore)

    # if len(included_nodes) == 3:
    #     print_graph(G, included_nodes, ignore_edges, flow)
    
        
    return None
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")



def _bidirectional_shortest_path(
    G, source, target, flow, nodes=None, edges=None
):
    # call helper to do the real work
    results = _bidirectional_pred_succ(G, source, target, flow, nodes, edges)
    if results == None:
        return None

    pred, succ, w = results

    # build path from pred+w+succ
    path = []
    # from w to target
    while w is not None:
        path.append(w)
        w = succ[w]
    # from source to w
    w = pred[path[0]]
    while w is not None:
        path.insert(0, w)
        w = pred[w]

    return len(path), path


def strongly_connected_components(G, flow, nodes, constrained_edges):
    """Generate nodes in strongly connected components of graph.
        copied from the regular implementation to make changes to use capacity
    """
    preorder = {}
    lowlink = {}
    print(f"started strongly_connected_components {len(nodes)}")
    sys.stdout.flush()    
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    for source in nodes:
        if source not in scc_found:
            queue = [source]
            
            while queue:
                if len(preorder) % 50000 == 0 and len(preorder) > 0:
                    print(f"preorder length {len(preorder)}")
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                    
                
                done = True
                for w in G._adj[v]:
                    
                    if w in nodes and w not in preorder and (G._adj[v][w]["capacity"] - flow[v][w] > 0) and (v,w) not in constrained_edges and (w,v) not in constrained_edges:
                        
                        queue.append(w)
                        done = False
                        break
                            
                if done:
                    
                    lowlink[v] = preorder[v]
                    for w in G._adj[v]:
                        if w in nodes and w not in scc_found and (G._adj[v][w]["capacity"] - flow[v][w] > 0) and (v,w) not in constrained_edges and (w,v) not in constrained_edges:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])

                                
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        if len(scc) > 2:
                            if len(scc) > 500:
                                print(f"biconnected_dfs for {len(scc)} nodes")
                            for c in biconnected_dfs(G, scc, constrained_edges):
                                yield c
                    else:
                        scc_queue.append(v)




def biconnected_dfs(G, connected_nodes, constrained_edges):
    # depth-first search algorithm to generate articulation points
    # and biconnected components
    visited = set()
    for start in connected_nodes:
        if start in visited:
            continue
        if start not in connected_nodes:
            continue
        discovery = {start: 0}  # time of first discovery of node during search
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, chain(iter(G._adj[start]), iter(G.in_edges._adjdict[start])))]
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if child not in connected_nodes:
                    continue
                if (parent, child) in constrained_edges or (child, parent) in constrained_edges:
                    continue
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = min(low[parent], discovery[child])
                        edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, chain(iter(G._adj[child]), iter(G.in_edges._adjdict[child]))))




                    edge_stack.append((parent, child))
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        
                        ind = edge_stack.index((grandparent, parent))
                        if len(edge_stack[ind:]) > 1:
                            yield get_nodes_from_edges(edge_stack[ind:])
                        edge_stack = edge_stack[:ind]
                        
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:  # length 1 so grandparent is root
                    root_children += 1
                    
                    ind = edge_stack.index((grandparent, parent))
                    if len(edge_stack[ind:]) > 1:
                        yield get_nodes_from_edges(edge_stack[ind:])
        
