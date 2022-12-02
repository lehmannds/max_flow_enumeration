from copy import copy
import unittest
import sys
import os
sys.path.append(f"{os.getcwd()}")
from test_solvers import test_pypoman
from lawler_trivial import lawler
from lawler2 import lawler as lawler2
from lawler1 import lawler as lawler1
from version4 import search_max_flow
import version1
from Test import print_time, get_simple_graph, print_all_flow
import numpy as np
from version1 import print_vertices, remove_flow_cycle
from networkx.algorithms.flow import build_residual_network
import copy
import time as m_time

class Test(unittest.TestCase):

    @staticmethod 
    def copy_flow(mf):
        flow = copy.deepcopy(mf[1])
        for n1 in mf[1]:
            for n2 in mf[1][n1]:
                f = mf[1][n1][n2]
                if n2 in mf[1] and n1 in mf[1][n2]:
                    f -= mf[1][n2][n1]
                    flow[n1][n2] = f
                flow[n2][n1] = -f

        # if R:
        #     for n1 in flow:
        #         for n2 in flow[n1]:
        #             if flow[n1][n2] > 0:
        #                 R.edges[n1, n2]["capacity"] -= flow[n1][n2]
        #                 R.edges[n2, n1]["capacity"] += flow[n1][n2]


        return flow

    @staticmethod
    def compare_files(file1, file2):
        file_names = [file1, file2]
        
        files = [open(name) for name in file_names]
        flows = [{} for f in files]
        
        ok = True

        idx2 = 0
        for f in files:
            current_flow = []
            idx = 1
            for line in f:
                if line[0] == "*":
                    if len(current_flow):
                        h = hash("".join(sorted(current_flow)))
                        if h in flows[idx2]:
                            print(f"flow {idx} in file {file_names[idx2]} is also {flows[idx2][h]}")        
                            ok = False
                            
                        flows[idx2][h] = idx
                        idx += 1
                    current_flow = []
                if line[0] == "{":
                    current_flow.append(line)

            if len(current_flow):
                h = hash("".join(sorted(current_flow)))
                flows[idx2][h] = idx
            idx2 += 1

        for i in np.arange(len(flows)):
            for j in flows[i]:
                if j not in flows[1-i]:
                    print(f"flow {flows[i][j]} in file {file_names[i]} not in in file {file_names[1-i]}")
                    
                    ok = False

        return ok


    def test_lawler(self):
        
        # version1._max_time_ms = 20 * 1000
        total_results = open("compare_results.txt", "w")
        results = []
        funcs = [
            # {
            #     "file_name": 'lawler_trival.txt',
            #     "func": lawler,
            #     "pass_residual": False
            # },
            # {
            #     "file_name": 'lawler1.txt',
            #     "func": lawler1,
            #     "pass_residual": True

            # },
            # {
            #     "file_name": 'lawler2.txt',
            #     "func": lawler2,
            #     "pass_residual": True

            # },
            {
                "file_name": 'version4.txt',
                "func": search_max_flow,
                "pass_residual": True
            }   
        ]

        timeout = 60
                        
        print_time()
        files = ["p2p-Gnutella08.txt", "p2p-Gnutella31.txt"]
        # files = ["roadNet-PA.txt", "roadNet-TX.txt", "roadNet-CA.txt"]
        # files = ["roadNet-CA.txt"]
        
        
        for file in files:
            sys.stdout = total_results
            first = True
            print(f'starting file {file}' + "*" * 50)
            sys.stdout.flush()
            r = [file]
            for f in funcs:
                sys.stdout = total_results
                print(f'starting {f["file_name"]}')
                print (m_time.strftime("%H:%M:%S", m_time.localtime()))
                mf = [0,0]
                while mf[0] == 0:
                    
                    sys.stdout.flush()
                    # roadNet-PA.txt   p2p-Gnutella08.txt
                    G, mf, source, target = get_simple_graph(None, True, True, None, True, deg=10, file_name=file) # not first, deg=10)
                    # G, mf, source, target = get_simple_graph(size, False, True, f"multiple_components{size}", True, deg=10) # not first, deg=10)
                    # G, mf, source, target = get_simple_graph(size, False, True, f"simple_graph{size}", True, deg=10) # not first, deg=10)
                    # G, mf, source, target = get_simple_graph(size, False, True, f"graph{size}", True, deg=10) # not first, deg=10)

                
                if first:
                    r.append(len(G.nodes))
                    r.append(len(G.edges))
                first = False         
                sys.stdout.flush()     
                sys.stdout = open(f["file_name"], 'w')
                # print_time(f"st, art  pypoman")
                # print(f"start size {size}" + ("*" * 50))
                print(f"start file {file}" + ("*" * 50))
                    
                # print_all_flow(mf[1], 0, 0, count=1, full_print=True)
                print_time()

                lawler_imp = f["func"]
                R = None
                
                if f["pass_residual"]:
                    R = build_residual_network(G, "capacity")
                    
                flow = Test.copy_flow(mf)
                remove_flow_cycle(flow, False)

                count = target + 1
                start_nodes = set()
                for i in G.nodes:
                    start_nodes.add(i)
                time, total_count, no_flow_count = lawler_imp(R or G, flow = flow or mf[1], 
                        source = source, target = target, flow_value = mf[0], unit_capacity = 20, 
                        check_flow_cycle=False, start_nodes=start_nodes, constrained_edges=set(),
                        always_do_components = False, timeout=timeout)
                r.append(time)
                r.append(total_count)
                r.append(no_flow_count)
                print (f"lawler total count: {total_count}")
                print (f"no_flow_count: {no_flow_count}")
                
                print_time("computing flows")
            sys.stdout = total_results        
            results.append(r)
            print(results)
    
    def test_compare(self):
        not_failed = True
        count = 10
        idx = 0
        size = 9
        results_output = open("compare_results.txt", "w")

        while not_failed and idx < count:
            # print(f"idx {idx}")
            G, mf, source, target = get_simple_graph(size, True, True, "compare_graph")
            sys.stdout = open('lawler_trival.txt', 'w')
            time, total_count, no_flow_count = lawler(G, mf[1], source, target, mf[0])
            vertices = test_pypoman(G, mf[0], source, target)
            print_vertices(G, vertices, True)

            sys.stdout = results_output
            print(f"v count {len(vertices)}")
            not_failed = Test.compare_files("lawler_trival.txt", "stdout_pypoman.txt")
            idx += 1
            sys.stdout.flush()
        
    def test_version_4_counts(self):
        
        size = 10
        G, mf, source, target = get_simple_graph(size, False, True, f"multiple_components{size}", True, deg=10) # not first, deg=10)
        timeout = 60000
        
        
        R = build_residual_network(G, "capacity")
            
        flow = Test.copy_flow(mf)
        remove_flow_cycle(flow, False)

        count = target + 1
        start_nodes = set()
        for i in G.nodes:
            start_nodes.add(i)
        time, total_count, no_flow_count = search_max_flow(R or G, flow = flow or mf[1], 
                source = source, target = target, flow_value = mf[0], unit_capacity = 20, 
                check_flow_cycle=False, start_nodes=start_nodes, constrained_edges=set(),
                always_do_components = False, timeout=timeout)
        