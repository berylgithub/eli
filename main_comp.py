# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 08:59:25 2021

@author: Saint8312
"""

import numpy as np
import elimination_ordering as eo
import time, pickle
from os import listdir
from os.path import isfile, join
import pandas as pd


def visu():
    p=5 #grid row
    q=25 #grid col
    print("<><><><> Grid with p=",p,"and q=",q,"<><><><>")
    grid = eo.grid_generator(p,q,0) #generate grid matrix
    
    #elimination ordering:
    EO = eo.elimination_ordering_class(grid, visualization=True, p=p, q=q) #must be on global scope
    e = EO.elimination_ordering(grid)
    return None

def generate_iperm():
    ndpath = "matrices/grid/ndmetis_input/"
    gridpath = "matrices/grid/grids/"
    #ps = [2,5,10,15,20]
    #qs = [p**2 for p in ps]
    qs = list(range(1,21))
    ps = [5]*len(qs)
    for i in range(len(ps)):
        p = ps[i]; q = qs[i]
        grid = eo.grid_generator(p,q,0)
        print(grid.shape)
        #fname="p_q_"+str(k) #"p_q_" or "p_qsqr_" or others
        fname = str(p)+"_"+str(q)
        '''with open(gridpath+fname+".grid", 'wb') as fp:
            pickle.dump(grid, fp)'''
        eo.adj_mat_to_metis_file(grid, ndpath+fname+".grid.metisgraph")
    #save grid sizes (p,q):
    data = {"p":ps, "q":qs}
    date = "02032021"
    with open(gridpath+str(len(ps))+"_"+date+"_grids_sizes.info", 'wb') as fp:
        pickle.dump(data, fp)
    
    
def computation():
    '''Main for computations'''
    '''All statistics for grids (+jointree)'''
    '''do elimination ordering and metis on each of the grids'''
    rootpath = "matrices/grid/"
    #gridpath = "matrices/grid/grids/"
    ipermpath = rootpath+"ndmetis_iperm/"
    files = np.array([f for f in listdir(ipermpath) if isfile(join(ipermpath, f))])
    print(files)
    
    #data containers:
    nv = []
    ne = []
    #mat_types = [] #bipartite or not, 1/0
    times = []
    fills_eli = []
    fills_metis = []
    fills_eli_ratio = []
    fills_metis_ratio = []
    eli_metis_ratios = []
    es = [] #list of e from eli
    #jointree data:
    max_C_eli = []
    max_K_eli = []
    max_C_metis = []
    max_K_metis = []
    #grids specific data:
    ps = []; qs = []
    fnames = []
    
    for file in files:
        grid = None
        str_pq = file.split('.')[0]; fnames.append(str_pq+".grid")
        p,q = str_pq.split('_'); p,q = (int(p), int(q)); ps.append(p); qs.append(q)
        #rather than loading grids from disk, it'll be more space efficient if we just re-generate the grids in memory
        grid = eo.grid_generator(p,q,0)
        print("\ngrid_shape",grid.shape)
        vertices = grid.shape[0]
        edges = np.sum(grid[np.triu_indices(grid.shape[0], 1)]) #sum of upper triangular
        nv.append(vertices); ne.append(edges)
        #elimination ordering:
        start = time.time()
        EO = eo.elimination_ordering_class(grid, visualization=False, p=p, q=q) #must be on global scope
        e = EO.elimination_ordering(grid)
        es.append(e)
        end = time.time()
        elapsed = end-start
        times.append(elapsed)
        print("time elapsed for elimination ordering: ",elapsed,"s")
        #grid is deleted above, so reload here:
        grid = eo.grid_generator(p,q,0)
        #eliminate + jointree generation:
        fill_eli, _, C_vs, sep_idxs = eo.eliminate(grid, e,join_tree=True); print("eli||generating jointree for ",str_pq,"completed!")
        _, _, max_C, max_K = eo.absorption(e, C_vs, sep_idxs); print("eli||absorption for",str_pq,"completed!")
        fills_eli.append(fill_eli)
        max_C_eli.append(max_C); max_K_eli.append(max_K); print("eli||max_C, max_K:",max_C, max_K)
        print("eli||fills",fill_eli)
        fills_eli_ratio.append(float(fill_eli/edges))
        print()
        #metis:
        grid = eo.grid_generator(p,q,0)
        metis_order = eo.iperm_to_orderlist(ipermpath+file)
        fill_metis, _, C_vs, sep_idxs = eo.eliminate(grid, metis_order, join_tree=True)
        fills_metis.append(fill_metis)
        print("metis||generating jointree for ",str_pq,"completed!")    
        _, _, max_C, max_K = eo.absorption(e, C_vs, sep_idxs)
        max_C_metis.append(max_C); max_K_metis.append(max_K); print("metis||max_C, max_K:",max_C, max_K)
        print("metis||absorption for",str_pq,"completed!")
        print("metis", fill_metis)
        fills_metis_ratio.append(float(fill_metis/edges))
        print()
        if (fill_eli != 0) and (fill_metis != 0): 
            eli_metis_ratios.append(float(fill_eli/fill_metis))
        else:
            eli_metis_ratios.append(0)
    
        #store results in file, within loop, so it will always store something even when stopped earlier
        data = {
            "grids": fnames,
            "nv": nv,
            "ne": ne,
            "eli_fills": fills_eli,
            "eli_ratio": fills_eli_ratio,
            "eli_times": times,
            "metis_fills": fills_metis,
            "metis_ratio": fills_metis_ratio,
            "eli/metis": eli_metis_ratios,
            "e_order":es,
            #jointree data:
            "max_C_eli": max_C_eli,
            "max_K_eli": max_K_eli,
            "max_C_metis": max_C_metis,
            "max_K_metis":max_K_metis
        }
        with open(rootpath+'grid_pqincr_ndiff_02032021.jt.p', 'wb') as fp:
            pickle.dump(data, fp)    
        with open(rootpath+'grid_pqincr_ndiff_02032021.jt.p', 'rb') as fp:
            data = pickle.load(fp)
            print(data)
    
if __name__ == '__main__':
    #generate_iperm()
    computation()
    #visu()