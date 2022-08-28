from pickle import STOP
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
import pypoman

from test_graphs import *


def pulptest():

    sizes = (1000 + 2500 * np.arange(0, 5)).astype('int')
    times = np.zeros(len(sizes))
    timesLP = np.zeros(len(sizes))
    max_flows = np.zeros(len(sizes))
    idx = 0

    for size in sizes:
        G = get_simple_test_graph() #get_rand_graph(size, 10)
        print({"started size": size})
        m = len(G.edges())
        n = size
        edges = G.edges()
    
        prob = LpProblem("X",LpMinimize)
        edge_vars = LpVariable.dicts("edges",edges,lowBound=0,cat='Continuous')
        s_edges = { key:value for (key,value) in edge_vars.items() if key[0] == 0}
        prob += lpSum([-s_edges[i] for i in s_edges])

        all_constraints = []

        for key in edges:
            
            prob += edge_vars[key] <= edges[key]["capacity"]

        inDict = {}
        for key in edges:
           if key[1] not in inDict:
               inDict[key[1]] = []
           inDict[key[1]].append(key[0])

        for n_key in G.nodes():
            
            if n_key != 0 and n_key != (size -1):
                vars1 = [edge_vars[(n_key, i)] for i in G._adj[n_key]]
                if n_key in inDict:
                    vars2 = [-edge_vars[(in_key, n_key)] for in_key in inDict[n_key]]
                else:
                    vars2 = []
                
                
                prob += lpSum([*vars1, *vars2]) == 0
        
        #prob.constraints = all_constraints

        set_print_to_console()
        start = time.time()
        s = prob.solve()
        LPopt = value(prob.objective)
        if  LPopt != None:
            LPopt = - LPopt
        end = time.time()
        timesLP[idx] = end - start
        set_print_to_console(False)

        start = time.time()
        mf = nx.algorithms.maximum_flow_value(G, 0, size - 1)
        end = time.time()
        times[idx] = end - start
        max_flows[idx] = mf

        print({"size": size, "mf": mf, "time": times[idx], "timeLP": timesLP[idx], "idx": idx, "opt": LPopt})
        idx += 1

    print({"mf": max_flows})

    plt.plot(sizes, times, label="Max-Flow")
    plt.plot(sizes, timesLP, label="LP")
    plt.legend()
    plt.show()    

def test_pypoman(G, max, source, target):
    #ineq_lhs, ineq_rhs = Get_Test_A_B()
    start = time.localtime()
    print (time.strftime("%H:%M:%S", start))
    ineq_lhs, ineq_rhs = Get_A_b(G, max, source, target)
    vertices = pypoman.compute_polytope_vertices(ineq_lhs, ineq_rhs)
    # for v in vertices:
    #     print({"v": v})

    # print(list(G.edges))
    print (time.strftime("%H:%M:%S", time.localtime()))
    
    return vertices


def test_solvers():

    sizes = (3 + 6 * np.arange(0, 10)).astype('int')
    times = np.zeros(len(sizes))
    timesLP = np.zeros(len(sizes))
    idx = 0
    for size in sizes:
        G = nx.complete_graph(size).to_directed()

        print({"started size": size})
        m = len(G.edges())
        n = size

        ineq_lhs = np.zeros((m, m))
        ineq_rhs = np.zeros(m)

        eq_lhs = np.zeros((n - 2, m))
        eq_rhs = np.zeros(n - 2)

        capacities = np.zeros(len(G.edges()))

        idx2 = 0
        for e in G.edges():
            rand = r.randint(0, 10) if e[1] != 0 else 0
            capacities[idx2] = rand
            G.edges[e]["capacity"] = rand
            idx2 += 1

        start = time.time()

        mf = nx.algorithms.maximum_flow_value(G, 0, size - 1)

        end = time.time()
        times[idx] = end - start
        
        ineq_lhs[0:m, :] =  np.diag(np.ones(m))
        ineq_rhs[0:m] =  capacities

        
        for i in np.arange(1, size-1):
            row = i - 1
            eq_lhs[row, i * (n-1): (i + 1) * (n-1)] = np.ones(n-1)
            for j in np.arange(n):
                if j != i:
                    eq_lhs[row, j * (n-1) + i - (1 if j < i else 0)] = -1
            
        
        c= np.zeros(m)
        c[:n-1] = -1
        start = time.time()
        opt = linprog(c=c, A_ub=ineq_lhs, b_ub=ineq_rhs, A_eq=eq_lhs, b_eq=eq_rhs,
                   #method="revised simplex")
                   method="interior-point")

        
        end = time.time()
        timesLP[idx] = end - start


        print({"size": size, "mf": mf, "time": times[idx], "timeLP": timesLP[idx], "idx": idx, "opt": -opt.fun})
        idx += 1

    plt.plot(sizes, times, label="Max-Flow")
    plt.plot(sizes, timesLP, label="LP")
    plt.legend()
    plt.show()


def Get_Test_A_B():
    A = np.array([
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],
    [1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0],
    [0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],
    [0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1]])
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 3])
    return A,b

def Get_A_b(G, max, source, target):
    m = len(G.edges())
    n = len(G.nodes())

    ineq_count = 2 * m + 2 * n

    ineq_lhs = np.zeros((ineq_count, m))
    ineq_rhs = np.zeros(ineq_count)

    # start edge capacity constaints
    # less than capacity contraints
    ineq_lhs[0:m, :] =  np.diag(np.ones(m))
    # more than zero constraints
    ineq_lhs[m:2*m, :] =  -np.diag(np.ones(m))
    
    idx = 0
    for e in G.edges():
        ineq_rhs[idx] = G.edges[e]["capacity"]
        #ineq_rhs[idx + m] = 0
        idx += 1
    
    # end edge capacity constaints


    idx = 2 * m

    # start excess flow constaints

    for n in G.nodes():
        idx2 = 0
        if n == target:
            # it is enough to set the constraint on the source
            idx += 2
            continue
        if n == source:
            # for all other nodes the excess flow should be zero
            ineq_rhs[idx] =  max
            ineq_rhs[idx + 1] =  -max
        for e in G.edges():
            if e[0] == n:
                # sum up all entering flow
                ineq_lhs[idx, idx2] =  1
                ineq_lhs[idx+1, idx2] =  -1
            if e[1] == n:
                # subtract exiting flow
                ineq_lhs[idx, idx2] =  -1
                ineq_lhs[idx+1, idx2] =  1

                

            idx2 += 1

        idx+=2

    return ineq_lhs, ineq_rhs



    