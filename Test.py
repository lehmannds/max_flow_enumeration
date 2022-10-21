from cProfile import label
from cmath import sqrt
from math import fabs
from operator import truediv
import pickle

from platform import node
import pstats
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
from version1 import remove_flow_cycle,print_all_flow, search_max_flow, print_vertices
import version1
import version3
import version4
import cProfile
from lawler1 import lawler as lawler1
import lawler2


_time = None
write_gpickle = True
run_lawler = False

def print_time(doing = None):
    global _time
    current = datetime.now()
    if _time is not None and doing:
        print(int((current - _time).total_seconds() * 1000), "ms", "doing:", doing)

    _time = current

def get_simple_graph(size = 9, write_graph=True, get_max_flow=False, name=None, use_existing=False, deg=3, file_name=None):
    unit_capacity = 20
    fn = name or (f"{file_name}.g" if file_name else f"graph{size}")
    fn = f"simple_rand_graphs/{fn}"
    set_source = False
    read_from_disk = False
    if use_existing and os.path.exists(fn):
        G = nx.readwrite.read_gpickle(fn)
        read_from_disk = True
    elif size <= 40 and not file_name:
        G = nx.readwrite.read_gpickle(fn)
    else:
        if name and "multiple_components" in name:
            G = get_multiple_components_test_graph(number_of_branches=13, show_graph=False)
        elif file_name:
            G, source, target = load_SNAP(f"C:/huji/thesis/test_files/{file_name}")
            set_source = True
        else:
            G = get_rand_local_graph(size, unit_capacity= unit_capacity, avg_degree=deg, local=deg* 3)
    # G = get_full_one_sided_graph(size, unit_capacity, False)
    # G = get_rand_local_graph(size, unit_capacity= unit_capacity)
    graph_name = f"get_full_one_sided_graph({size})"
    # print_entire_graph(G, True)
    if not set_source:
        source, target = 0, len(G.nodes) - 1
    
    
    if write_graph and not read_from_disk:
        nx.readwrite.write_gpickle(G, fn)
    
    if get_max_flow:
        mflow_file_name = fn + "max_flow"
        if os.path.exists(mflow_file_name):
            with open(mflow_file_name, "rb") as f:
                mf = pickle.load(f)
        else:
            mf = nx.algorithms.maximum_flow(G, source, target)
            # flow = copy.deepcopy(mf[1])
            if os.path.exists(fn):
                with open(mflow_file_name, "wb") as f:
                    pickle.dump(mf, f)
        print(f"flow: {mf[0]}, edge_count:{len(G.edges())}")
            
    return G, mf, source, target

def get_all_max_flows(size = 5, last_graph=False, simple=True, version = 3, run_normal = True, file=None):
        
    results = []
    locality = min(int(size / 5),3)
    unit_capacity = 20
    degree = min(3, int(math.sqrt(size) ))
    print_time()
    graph_name = ""
    if not simple:
        #G, source, target = load_DIMACS("C:/huji/thesis/test_files/simple_with_2_way_edges.txt", unit_capacity)
        #G, source, target = load_DIMACS("C:/Users/Daniel/Downloads/huck.col.txt", unit_capacity)
        #G, source, target = load_DIMACS("C:/Users/Daniel/Downloads/anna.col.txt")
        #G, source, target = load_DIMACS("C:/Users/Daniel\Downloads/dimacs_deconv/graph3x3.max/export/graph3x3.max")
        #G, source, target = load_DIMACS("C:/Users/Daniel/Downloads/dimacs_super_res/super_res-E1.max/export/super_res-E1.max")
        #G =  get_rand_local_graph(size, 3, local=locality, unit_capacity=unit_capacity) #get_full_graph(size, unit_capacity, show_graph=False)#nx.readwrite.read_gpickle("graph")#get_multiple_components_test_graph()#get_rand_local_graph(size, 4, local=6, unit_capacity=unit_capacity)#get_multiple_components_test_graph()#get_simple_test_graph()##cycle_flow_test(20)##
        #G = nx.readwrite.read_gpickle("lawler")
        # G, source, target = load_SNAP("C:/huji/thesis/test_files/p2p-Gnutella08.txt")
        # G, source, target = load_SNAP("C:/huji/thesis/test_files/p2p-Gnutella31.txt")
        # G, source, target = load_SNAP("C:/huji/thesis/test_files/Slashdot0811.txt")
        # G, source, target = load_SNAP("C:/huji/thesis/test_files/twitter_combined.txt")
        # G, source, target = load_SNAP("C:/huji/thesis/test_files/roadNet-PA.txt")
        graph_name = os.path.basename(file)
        G, source, target = load_SNAP(file)
    else:
        G = get_full_one_sided_graph(size, unit_capacity, False)
        graph_name = f"get_full_one_sided_graph({size})"
        # print_entire_graph(G, False)
        source, target = 0, len(G.nodes) - 1
    
    print_time("loading graph")
    #G = get_rand_p_graph(6, 0.00005, unit_capacity)
    if last_graph:
        G = nx.readwrite.read_gpickle("graph")
    
    
    #G = get_rand_p_graph(7, 0.65, unit_capacity)
    
    

    versions = [1,1,version3,version4]

    if write_gpickle and len(G.edges) < 1000:
        nx.readwrite.write_gpickle(G, "graph")
    
    
    mf = nx.algorithms.maximum_flow(G, source, target)
    print_time("finding max flow")
    
    flow = copy.deepcopy(mf[1])
    print_time("deep copy")
    print(f"flow: {mf[0]}, edge_count:{len(G.edges())}")
    
    run_pypoman = True and not last_graph
    
    run_version = version > 2

    if run_pypoman:
        sys.stdout = open('stdout_pypoman.txt', 'w')
        vertices = test_pypoman(G, mf[0], source, target)
        print_vertices(G, vertices, False)
        print_time(f"run_pypoman")
    
    
    R = build_residual_network(G, "capacity")
    print_time(f"building residual")
    for n1 in mf[1]:
        for n2 in mf[1][n1]:
            f = mf[1][n1][n2]
            if n2 in mf[1] and n1 in mf[1][n2]:
                f -= mf[1][n2][n1]
                flow[n1][n2] = f
            flow[n2][n1] = -f
    print_time(f"build flow object")
    remove_flow_cycle(flow)    
    print_time(f"remove flow cycle")
    if run_lawler:
        sys.stdout = open(f'{graph_name}stdout_lawler.txt', 'w')
        print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))
        print_all_flow(flow, 0, 0, 1)
        time, total_count, no_flow_count = lawler2.lawler(R, flow, unit_capacity)
        results.append(['lawler', time, total_count, no_flow_count])
        print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))
        # print (f"** total count: {total_count}")

        print_time(f"run lawler")

    do_components = False
    connected = False
    if do_components:
        components = strongly_connected_components(R, flow, {n for n in G.nodes}, set())
        connected = True
    else:
        components = [{n for n in G.nodes}]

    
    
    if run_version:
        sys.stdout = open(f'{graph_name}stdout{version}.txt', 'w')
        
        print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))
        print_all_flow(flow, 0, 0, 1)
        # print(f"componets:", [len(c) for c in components])
        for c in components:
            print(f"start component len = {len(c)} .")
            time, total_count, no_flow_count = versions[version-1].search_max_flow(R, c, set(), flow, unit_capacity, connected=connected)
            results.append([f'version {version}', time, total_count, no_flow_count])
            


        print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))
    print_time(f"version: {version}")    
    return results
    if not run_normal:
        return

    sys.stdout = open('stdout1.txt', 'w')
    
    print("*", {"flow": mf[0]})

    print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))

    if mf[0] > 0:
        print_all_flow(flow, 0, 0, 1)
    else:
        print("no flows")
        return 0

    idx = 0 
    percent_done = 0

    # if len(components) > 1:
    #     print("* components:", [len(c) for c in components])

    print_time(f"till start iterate component")
    for c in components:
        #print(c)
        p = 0.1 # 1 / len(components)
        search_max_flow(R, c, set(), flow, unit_capacity, 0, p, percent_done, c_idx=idx)
        percent_done += p    
        idx += 1
        print_time(f"component")
    print("******************************")
    
    print (datetime.now().isoformat(sep=' ', timespec='milliseconds'))
    return

def compare_files(idx1, idx2,run_all, size):
    file_names = [f"get_full_one_sided_graph({size})stdout{idx1}.txt", f"get_full_one_sided_graph({size})stdout{idx2}.txt"]
    
    files = [open(name) for name in file_names]
    flows = [{} for f in files]
    
    ok = True

    idx2 = 0
    for f in files:
        current_flow = ""
        idx = 1
        for line in f:
            if line[0] == "*":
                if current_flow:
                    h = hash(current_flow)
                    if h in flows[idx2]:
                        print(f"flow {idx} in file {file_names[idx2]} is also {flows[idx2][h]}")        
                        ok = False
                        if not run_all:
                            return False
                    flows[idx2][h] = idx
                    idx += 1
                current_flow = ""
            if line[0] == "{":
                current_flow += line

        if current_flow:
            flows[idx2][hash(current_flow)] = idx
        idx2 += 1

    for i in np.arange(len(flows)):
        for j in flows[i]:
            if j not in flows[1-i]:
                print(f"flow {flows[i][j]} in file {file_names[i]} not in in file {file_names[1-i]}")
                if not run_all:
                    return False
                ok = False

    return ok


# sizes = [2000]#np.arange(6, 50, 2)
# times = 1
# avgs = []
# maxs = []

# for size in sizes:
#     number_of_flows = [get_all_max_flows(size) for i in np.arange(times)]
#     print(number_of_flows)
#     avgs.append(np.average(number_of_flows))
#     maxs.append(np.max(number_of_flows))

run_compare = False
def test_run(v3 = False, version=4, simple=False, size=9, last_graph=False, count=1, file=None):
    not_failed = True
    idx = 0

    while not_failed and idx < count:
        results = get_all_max_flows(last_graph=last_graph, size=size, simple=simple, run_normal=False, version=4, file=file)
        sys.stdout = sys.__stdout__
        if run_compare:
            not_failed = compare_files("_lawler", 4, True, size)
        idx += 1

    return results

def testSCC():

    print(f"start get graph")
    size = 300000
    G, source, target = load_SNAP("C:/huji/thesis/test_files/roadNet-PA.txt")
    source, target = 0, len(G.nodes) - 1
    print(f"finished. \n find max flow")
    mf = nx.algorithms.maximum_flow(G, source, target)
    flow = mf[1]

    for n1 in mf[1]:
        for n2 in mf[1][n1]:
            f = mf[1][n1][n2]
            if n2 in mf[1] and n1 in mf[1][n2]:
                f -= mf[1][n2][n1]
                flow[n1][n2] = f
            flow[n2][n1] = -f
    



    print(f"finished. \n build residual")
    R = build_residual_network(G, "capacity")
    flow = mf[1]
    print(f"finished. \n remove_flow_cycle")
    remove_flow_cycle(flow)    

    print(f"finished. \n SCC computation")
    s = set()
    for n in G.nodes:
        s.add(n)
    print(f"create")
    components = list(strongly_connected_components(R, flow, s, set()))

version1.print_flow_details = False
    
def stats():
    v3 = False 
    dir = "v4"
    size = 25
    dat_file = f"./{dir}/output.dat"
    str = f"True, version=4, simple=True, size={size}, last_graph=False, count=1, file=None"
    cProfile.run(f"test_run({str})", dat_file)
    # cProfile.run(f"testSCC()", dat_file)
    with open(f"./{dir}/time.txt", "w") as f:
        p = pstats.Stats(dat_file, stream=f)
        p.sort_stats("time").print_stats()

    with open(f"./{dir}/calls.txt", "w") as f:
        p = pstats.Stats(dat_file, stream=f)
        p.sort_stats("calls").print_stats()

    # plt.plot(sizes, avgs, label="average")
    # plt.plot(sizes, maxs, label="max")
    # plt.show()

# stats()

files = ["C:/huji/thesis/test_files/p2p-Gnutella31.txt",
        "C:/huji/thesis/test_files/p2p-Gnutella31.txt",
        "C:/huji/thesis/test_files/Slashdot0811.txt",
        "C:/huji/thesis/test_files/twitter_combined.txt",
        "C:/huji/thesis/test_files/roadNet-PA.txt"]

def parse_output(output):
    lines = output.split("\n")
    count = 0
    start = 0
    end = 0
    for idx, line in enumerate(lines):
        if line.startswith("begin"):
            start = idx 
        if line.startswith("end"):
            end = idx

    if start == 0:
        count = 0
    elif end != 0:
        count = end - start - 2
    else:
        count = len(lines) - start - 2

    return count, None




def run_many_tests():
    all_results = []
    # sizes = [10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
    # sizes = [7] * 20
    sizes = [10]
    size = 0
    file = ""
    
    simple = True
    for size in sizes:
    # for file in files:
        results = test_run(True, version=4, simple=simple, size=size  , last_graph=False, count=1, file=file)
        for r in results:
            all_results.append([size, *r])

        print("*" * 100)
        print(all_results)


def test_pypoman_on_many_graphs():
    import subprocess
    from multiprocessing import Process

    sys.stdout = open('stdout_pypoman.txt', 'w')
    # sizes = [5, 8, 11, 14, 17, 20]
    sizes = [40, 80]
    for size in sizes:
        # print_time(f"st, art  pypoman")
        print(f"start size {size}" + ("*" * 50))
        print_time()
        mf = [0,0]
        while mf[0] == 0:
            G, mf, source, target = get_simple_graph(size, True, True)
        
        timeout = 600
        lawler2._max_time_ms = timeout * 1000   
        R = build_residual_network(G, "capacity")

        flow = copy.deepcopy(mf[1])

        for n1 in mf[1]:
            for n2 in mf[1][n1]:
                f = mf[1][n1][n2]
                if n2 in mf[1] and n1 in mf[1][n2]:
                    f -= mf[1][n2][n1]
                    flow[n1][n2] = f
                flow[n2][n1] = -f

        remove_flow_cycle(flow) 

        time, total_count, no_flow_count = lawler2.lawler(R, flow, 20)
        print (f"lawler total count: {total_count}")

        print_time(f"run lawler")
        sys.stdout.flush()


        # print("start pypoman")
        # print_time()
        # p1 = Process(target=test_pypoman, args=(G, mf[0], source, target), name='test_pypoman')
        # p1.start()
        # # vertices =
        # p1.join(timeout=timeout)
        # p1.kill()
        # if p1.exitcode == 0:
        #     print_time(f"run_pypoman finished size: {size}")
        # else:
        #     print_time(f"run_pypoman timed out size: {size}")
        # sys.stdout.flush()
        
        
        print_time()
        
        finished = False
        try:
            print("start lrs")
            lrs_out_fn = "lrs_out.txt"
            result = subprocess.run(["C:/projects/test_python_project/lrs1", "inputs/out.ine"], 
                        cwd="C:/projects/test_python_project/",
                        timeout=timeout,
                        stdout = open(lrs_out_fn, 'w'))
                        
            
            with  open(lrs_out_fn, 'r') as f:
                output = f.read()

            count, flows = parse_output(output)
            finished = True
        except subprocess.TimeoutExpired as e:
            with  open(lrs_out_fn, 'r') as f:
                output = f.read()
            # output = e.output.decode(encoding="utf-8", errors="ignore")
            count, flows = parse_output(output)

        # print(result)
        print_time(f"lrs - flow count: {count}, size: {size}, finished: {finished}")
        print_time(f"lrs: {size}")
        sys.stdout.flush()


if __name__ == '__main__':
    test_pypoman_on_many_graphs()