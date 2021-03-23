# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:03:21 2021

@author: Saint8312
"""

"""
elimination ordering in networkx data structure with order of O(nodes+edges) space
"""

import numpy as np
import networkx as nx
from itertools import combinations
from scipy.io import mmread, mminfo
from itertools import chain

#for grid drawing only:
import matplotlib.pyplot as plt
from matplotlib import colors


class elimination_ordering_class:
    def __init__(self, graph, visualization=False, p=0, q=0):
        '''function for data initialization, returns:
        - e vector placeholder
        - weight vector w
        - empty merge forest
        - first zero and last zero idxes
        - deleted bool array
        '''
        self.n_init = graph.number_of_nodes() #initial graph size, usable for static vectors
        self.n = self.n_init #dynamic graph size
        self.e = np.array([-1]*self.n) #for now the placeholder is an array of -1
        self.w = np.array([1]*self.n) #weight vector for merge forest
        self.merge_forest = nx.Graph() #merge forest for merging procedures
        self.deleted = np.array([False]*self.n)
        self.first_zero = 0; self.last_zero = -1
        
        '''calculate the valencies early, so that within the stages there will be no valencies re-calculation:'''
        self.valencies = np.array([len(graph[i]) for i in graph.nodes]) #valency[i] will be subtracted for each operation in a node
        self.sum_valencies = np.sum(self.valencies) #for calculating the mean_valency, the sum should be subtracted for each operation in a node
        
        #for grid information only:
        self.p = p #row
        self.q = q #col
        
        #for visualization
        self.round = 0
        self.visu = False
        if visualization:
            self.visu = True
            self.place_loc = np.zeros(self.n) #if the placement occurs in separate then set the element as "-1"
            self.rounds_e = np.zeros(self.n) #indicate the rounds of which the vertex is eliminated from
            
            #specialized for normalize stage visualization:
            self.R_switch = False #true if sum(R_counters) >= 0 
            self.R_counters = np.zeros(6) #each index is for each rule, Ri = R_counters_{i-1}
            self.R_strings = [] #to be printed at the end of normalize stage
                
            #specialized for separate stage visualization:
            self.separate_placed_rounds = []
        
        #for separator display visualization:
        self.Nks = [] #list of separator indexes per round for all rounds
        
    '''Normalize Stage'''
    '''preliminaries:
    - node == vertex ("vertices" for plural)
    - nodes' index start at 0
    - all graphs and trees are represented in matrix farm
    - valency is the sum of the edges' weights of a node, i.e. the sum of the row in a corresponding index, in numpy:
        np.sum(graph[node_idx])
    - currently, the matrix is assumed to be symmetric (undirected)
    - a fill is only calculated during the elimination, not during the algorithm process to get the elimination ordering,
    and a fill is A[i][j] = 1 iff A[i][k] = A[k][i] = A[j][k] = A[k][j] = 1 then A[i][k] = A[k][i] = A[j][k] = A[k][j] = 0
    - the diagonal element is always ignored (must be zero'd first if it's nonzero)
    - non zeros' values doesn't matter (unweighted), it's always binary (0,1)
    '''

    #normalize stage:
    def normalize(self, graph):
        if self.visu and self.round < 1:
            #print("\n++++ Normalization Stage ++++") 
            self.R_strings.append("++++ Normalization Stage ++++") #print this only if normalize is not empty
        modified = np.array([1]*self.n_init) #modified = 1, otherwise 0'''
        mod_count = self.n #modified counter
        #normalize stage
        #for now, cyclic ordering assumption: start from the 1st index to last idx, hence for-loop
        #need merge check for every node passed, by w[i] > 1
        #print("i, n, m, valency, e, summodified, firstzero, lastzero")
        while np.any(modified[list(graph.nodes)]):
        #while mod_count > 0:
            #print()
            #for i in range(n_init):
            '''for visu purpose, use the self.round'''
            if self.visu and self.round < 1:
                self.R_strings.append("++ new normalization cycle ++")
                self.R_strings.append("++ i, n, valency, m ++")
            for i in list(graph.nodes):
                '''
                #check if it is already deleted, if yes, skip:
                if self.deleted[i]: #deleted in prev round
                    modified[i] = 0 #set modified to 0
                    print("already deleted:",i)
                    continue
                '''
                #check if all vertices are already unmodified:
                if np.any(modified) == False:
                #if mod_count == 0:
                    break
                #recalculate all of the values:
                valency = self.valencies[i] #get vertex's valency
                m = None
                neighbours = list(graph[i]) #get the neighbours of i
                len_neighbours = len(neighbours)
                #check all of the conditions based on the valency
                if valency == self.n-1:
                    ##always check for merge - i.e w[i] > 1
                    if self.w[i] > 1:
                        ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                        len_e = len(self.e)
                        len_ord_list = len(ordered_list)
                        self.e[len_e + self.last_zero - len_ord_list + 1 : len_e + self.last_zero + 1] = ordered_list #lastzero placement
                        if self.visu:
                            self.rounds_e[ordered_list] = self.round
                        self.last_zero -= len_ord_list #decrement last zero by the size of the ordered list 
                    else:
                        #add to the last zero and decrement the indexer:
                        self.e[self.last_zero] = i
                        if self.visu:
                            self.rounds_e[i] = self.round
                        self.last_zero -= 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 1, place "+str(i)+" last")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[0] += 1
                    self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                    self.valencies[i] = 0 #set valency[i] = 0
                    self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                    self.n -= 1 #decrease n
                    graph.remove_node(i) #delete node from graph
                    self.deleted[i] = True
                    modified[neighbours] = 1
                elif (valency > np.ceil(self.n/2)) and (valency == np.max(self.valencies)):
                    if self.w[i] > 1:
                        ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                        len_e = len(self.e)
                        len_ord_list = len(ordered_list)
                        self.e[len_e + self.last_zero - len_ord_list + 1 : len_e + self.last_zero + 1] = ordered_list
                        if self.visu:
                            self.rounds_e[ordered_list] = self.round
                        self.last_zero -= len_ord_list
                    else:
                        self.e[self.last_zero] = i
                        if self.visu:
                            self.rounds_e[i] = self.round
                        self.last_zero -= 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 2, place "+str(i)+" last")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[1] += 1
                    self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                    self.valencies[i] = 0 #set valency[i] = 0
                    self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                    self.n -= 1 #decrease n
                    graph.remove_node(i) #delete node from graph
                    self.deleted[i] = True
                    modified[neighbours] = 1
                elif valency <= 1:
                    #e.insert(0, i) #place vertex first
                    if self.w[i] > 1:
                        ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                        len_e = len(self.e)
                        len_ord_list = len(ordered_list)
                        self.e[self.first_zero : self.first_zero + len_ord_list] = ordered_list #insert by firstzero pos
                        if self.visu:
                            self.rounds_e[ordered_list] = self.round
                        self.first_zero += len_ord_list #increment the first zero by the size of the ordered list
                    else:
                        #add to the first zero pos and increment the indexer:
                        self.e[self.first_zero] = i
                        if self.visu:
                            self.rounds_e[i] = self.round
                        self.first_zero += 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 3, place "+str(i)+" first")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[2] += 1
                    self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                    self.valencies[i] = 0 #set valency[i] = 0
                    self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                    self.n -= 1 #decrease n
                    graph.remove_node(i) #delete node from graph
                    self.deleted[i] = True
                    modified[neighbours] = 1
                elif valency == 2:
                    #e.insert(0, i)
                    if self.w[i] > 1:
                        ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                        len_e = len(self.e)
                        len_ord_list = len(ordered_list)
                        self.e[self.first_zero : self.first_zero + len_ord_list] = ordered_list #insert by firstzero pos
                        if self.visu:
                            self.rounds_e[ordered_list] = self.round
                        self.first_zero += len_ord_list
                    else:
                        #add to the first zero pos and increment the indexer:
                        self.e[self.first_zero] = i
                        if self.visu:
                            self.rounds_e[i] = self.round
                        self.first_zero += 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 4, place "+str(i)+" first")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[3] += 1
                    #check if the neigbours are already connected:
                    if graph.has_edge(neighbours[0], neighbours[1]):
                        self.valencies[neighbours] -= 1 #subtract the valencies
                    else:
                        graph.add_edge(neighbours[0], neighbours[1]) #fill an edge between j
                    self.sum_valencies -= self.valencies[i] #subtract the sum_valencies by i only, because of the fill between j
                    self.valencies[i] = 0 #set valency[i] = 0
                    self.n -= 1 #decrease n
                    graph.remove_node(i) #delete node from graph
                    self.deleted[i] = True
                    modified[neighbours] = 1
                else:
                    #m = np.min([self.sum_valencies/self.n, np.floor(self.n**(1/4) + 3)])
                    mean_v = self.sum_valencies/self.n
                    n_fourth = np.floor(self.n**(1/4) + 3)
                    m = n_fourth
                    if mean_v < n_fourth:
                        m = mean_v
                    if valency <= m:
                        #R5:
                        if clique_check_1(graph, neighbours):
                            if self.w[i] > 1:
                                ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                                #print("tree[i]",np.where(merge_forest[i] == 1))
                                len_e = len(self.e)
                                len_ord_list = len(ordered_list)
                                self.e[self.first_zero : self.first_zero + len_ord_list] = ordered_list #insert by firstzero pos
                                if self.visu:
                                    self.rounds_e[ordered_list] = self.round
                                self.first_zero += len_ord_list
                                #print("place multiple nodes",ordered_list)
                            else:
                                #add to the first zero pos and increment the indexer:
                                #print("place one node")
                                self.e[self.first_zero] = i
                                if self.visu:
                                    self.rounds_e[i] = self.round
                                self.first_zero += 1
                            if self.visu and self.round < 1:
                                self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 5, place "+str(i)+" first")
                                self.R_switch = True
                            if self.visu:
                                self.R_counters[4] += 1
                                self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                            self.valencies[i] = 0 #set valency[i] = 0
                            self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                            self.n -= 1 #decrease n
                            graph.remove_node(i) #delete node from graph
                            self.deleted[i] = True
                            modified[neighbours] = 1
                        #R6:
                        else:
                            bool_subset, j_node = check_subset_1(graph, neighbours)
                            if bool_subset:
                                self.merge_forest.add_edge(j_node, i) #merge i into j, add directed edge j->i
                                self.w[j_node] += 1 #increment weight
                                if self.visu and self.round < 1:
                                    self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 6, merged "+str(i)+" to "+str(j_node))
                                    self.R_switch = True
                                if self.visu:
                                    self.R_counters[5] += 1
                                self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                                self.valencies[i] = 0 #set valency[i] = 0
                                self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                                self.n -= 1 #decrease n
                                graph.remove_node(i) #delete node from graph
                                self.deleted[i] = True
                                modified[neighbours] = 1
                '''
                elif (valency <= m): 
                    bool_subset, j_node = check_subset(graph, neighbours) #gamma(i) \subset j^uptack, j \in gamma(i)
                    if bool_subset:
                        self.merge_forest.add_edge(j_node, i) #merge i into j, add directed edge j->i
                        self.w[j_node] += 1 #increment weight
                        if self.visu and self.round < 1:
                            self.R_strings.append(str(i)+" "+ str(self.n)+" "+ str(valency)+" "+ str(m)+ "||rule 6, merged "+str(i)+" to "+str(j_node))
                            self.R_switch = True
                        if self.visu:
                            self.R_counters[5] += 1
                        self.sum_valencies -= (self.valencies[i] + len_neighbours) #subtract the sum_valencies by the deleted nodes
                        self.valencies[i] = 0 #set valency[i] = 0
                        self.valencies[neighbours] -= 1 #update valencies[j] -= 1
                        self.n -= 1 #decrease n
                        graph.remove_node(i) #delete node from graph
                        self.deleted[i] = True
                        modified[neighbours] = 1
                '''   
                modified[i] = 0 #set vertex as unmodified
        
            #print per cycle here:
            if self.visu and self.R_switch and self.round < 1:
                for s in self.R_strings:
                    print(s)
                self.R_strings = [] #empty the strings container
                self.R_switch = False #turn switch off
        #return e, w, first_zero, last_zero, merge_forest
        #return first_zero, last_zero
        
    '''Separate Stage'''
    def separate(self, graph):
        #global deleted, e, w, first_zero, last_zero, merge_forest
        if self.visu and self.round < 1:
            print("\n---- Separation stage ----")
            
        if self.visu:
            separate_placed_round = 0
            
        #n_init = graph.number_of_nodes() #actual graph size after eliminations (if happened before this)

        '''RCM part'''
        #1, d=0, pick vertex e with max valency:
        d_prime = 0
        #n_nodes = get_total_nodes(graph, graph.shape[0]) #current total nodes
        '''valency calculation must be done in the very first step'''
        #valencies = np.array([len(graph[i]) for i in graph.nodes]) #array of valency, not ordered monotonicly (not contiguous) increasing, there maybe cutoff somewhere
        e_sep = np.argmax(self.valencies) #get the node with max valency immediately since valencies is contiguous
        if self.visu and self.round < 1:
            print("step 1, e, valency[e]:", e_sep, self.valencies[e_sep])
        
        #2, need to find a set of M with max distansce from e, which requires BFS or djikstra:
        #print("#2: ")
        distances = nx.single_source_shortest_path_length(graph, e_sep) #dict of {node: distance}, it is unsorted
        #conn_components = np.where(distances != np.inf)[0] #indexes of connected components within the subgraph where e resides
        s = len(distances) #total connected components
        d = np.max(list(distances.values())) #max distance from e
        M = [vert for vert in distances if distances[vert] == d]  #set of vertices with max distance from e

        if self.visu and self.round < 1:
            print("step 2, d, M, s:",d,M,s)
            
        #3, if d'>d, d'=d, pick a vertex from M with max valency, back to 2 if the first e is close to the second e
        loopcount = 0 #for repetition statistics
        while d>d_prime:
            if self.visu and self.round < 1:
                print("step 3: ")
                print("d > d', goto 2")
            #print("d, d_prime, e_sep",d, d_prime, e_sep)
            d_prime = d
            max_vertex,_ = get_max_valency(M, self.valencies) #probably need to be replaced, because valencies indexes doesnt correspond to full valency of nodes
            #print("M, valencies",M, valencies)
            e_sep = max_vertex

            #do 2 again:
            distances = nx.single_source_shortest_path_length(graph, e_sep) #dict of {node: distance}, it is unsorted
            #conn_components = np.where(distances != np.inf)[0] #indexes of connected components within the subgraph where e resides
            s = len(distances) #total connected components
            d = np.max(list(distances.values())) #max distance from e
            M = [vert for vert in distances if distances[vert] == d]  #set of vertices with max distance from e
            loopcount+=1
            
            if self.visu and self.round < 1:
                print("step 2, d, M, s:",d,M,s)
        #print("RCM loopcount", loopcount)
        
        '''end of RCM'''
        d = int(d)
        
        #4, get the N_k, n_k from e, 0<=k<=d, d=max distance, k \in Z
        #print("#3.5: n_k from e, 0<=k<=d, d=max distance")
        '''fill all N and n:'''
        N = []
        n = np.zeros(d+1)
        for i in range(0, d+1):
            N.append([k for k,v in distances.items() if v==i])
            n[i] = len(N[i])
        if self.visu and self.round < 1:
            print("step 4, n_k:",n)
        
        #5, Compute the partial sums v_k= sum_{i<=k} n_i (numpy.cumsum):
        v = np.cumsum(n)
        if self.visu and self.round < 1:
            print("step 5, v:",v)
        
        #6, Compute the k where min(v_d-v_k,v_k-n_k)/n_k is max, with the smallest n_k.
        max_idx = None
        max_val = -np.inf
        possible_ks = []
        c_ks = np.zeros(d+1)
        for k in range(d+1):
            c_k = np.min([v[d] - v[k], v[k] - n[k]])/n[k]
            c_ks[k] = c_k #for another n_k possibilites
            if c_k > max_val:
                max_val = c_k; max_idx = k;
            #print(c_k,"[",v[d] - v[k], v[k] - n[k],"]", n[k])
        #look for smallest n_k:
        for i in range(c_ks.shape[0]):
            if c_ks[i] == max_val:
                possible_ks.append(i)
        k = None
        if len(possible_ks) <= 1:
            k = max_idx
        else:
            k = possible_ks[np.argmin(n[possible_ks])]
        if self.visu and self.round < 1:
            print("step 6, k=",k)
        

        #tried = np.array([0]*self.n); tried[e_sep] = 1 #is this still usable?
        
        #step 8: compute the b_i for i\in N_k, sort N_k in increasing order of b_i:
        b = np.zeros(self.n_init)
        #i_idxs = N[k]
        for node_i in N[k]:
            out_w_nodes = np.intersect1d(graph[node_i], N[k+1])
            b[node_i] = np.sum(self.w[out_w_nodes]) #w = weights from normalization, need to know which value belongs to which
            #print("gamma_i, N[k+1]",gamma_i, N[k+1])
        sorted_b_Nk_idx = np.argsort(b[N[k]])
        sorted_Nk = [N[k][i] for i in sorted_b_Nk_idx]
        if self.visu and self.round < 1:
            print("step 8:")
            print("all b_i :",b)
            print("sorted N[k] by b_i:",sorted_Nk)
        #place the i with positive b_i last:
        for i in reversed(sorted_Nk):
            if b[i] > 0: #place i last when b[i] > 0
                if self.w[i] > 1:
                    ordered_list = get_ordered_list_merged_vertex(self.merge_forest, i)
                    len_e = len(self.e)
                    len_ord_list = len(ordered_list)
                    self.e[len_e + self.last_zero - len_ord_list + 1 : len_e + self.last_zero + 1] = ordered_list
                    self.last_zero -= len_ord_list
                    if self.visu:
                        self.place_loc[ordered_list] = -1
                        self.rounds_e[ordered_list] = self.round
                        separate_placed_round += len_ord_list
                    if self.visu and self.round < 1:
                        print("weight[i] > 1, place merge-tree alongside",i,",placed list: ",ordered_list)
                else:
                    self.e[self.last_zero] = i
                    if self.visu:
                        self.place_loc[i] = -1
                        self.rounds_e[i] = self.round
                        separate_placed_round += 1
                    self.last_zero -= 1
                    if self.visu and self.round < 1:
                        print("placed",i,"last")
                #update nodes data:
                self.sum_valencies -= (self.valencies[i] + len(list(graph[i]))) #subtract the sum_valencies by the deleted nodes
                self.valencies[i] = 0 #set valency[i] = 0
                self.valencies[list(graph[i])] -= 1 #update valencies[j] -= 1
                self.n -= 1 #decrease n
                graph.remove_node(i) #delete node from graph
                self.deleted[i] = True
        
        '''display grid here'''
        '''in step 4 before the inner while loop a display
        of the vertices as a p x q gray image - encode distances <k,k,k+1,>k+1
        as light gray, black, dark gray, white. You can stop the algorithm
        after the first round; then larger instances can be created.'''
        
        if self.visu and self.round < 1:
            print("****grid display:")
            print("grid for k =",k)
            Nbk = []; Nak = []

            Nbk = list(chain.from_iterable(N[:k]))
            #print("n",n)
            try: 
                Nak = list(chain.from_iterable(N[k+2:]))
            except:
                print("n[k+2] unreachable")

            #transform vertex index to coordinate:
            #flatten N<k + N_k + N_k+1 + N>k+1:

            A = np.zeros((self.p, self.q)) #matrix color placeholder
            #fill color on coordinate:
            for i in range(self.p*self.q):
                vertex_id = i
                x_idx = vertex_id%self.q
                y_idx = int(vertex_id/self.q)
                #fill color on x_idx, y_idx:
                if vertex_id in Nbk:
                    #print(x_idx, y_idx, "light gray")
                    A[y_idx, x_idx] = 1
                elif vertex_id in N[k]:
                    #print(x_idx, y_idx, "black")
                    A[y_idx, x_idx] = 2
                elif vertex_id in N[k+1]:
                    #print(x_idx, y_idx, "darkgray")
                    A[y_idx, x_idx] = 3
                elif vertex_id in Nak:
                    #print(x_idx, y_idx, "white")
                    A[y_idx, x_idx] = 4
                #else:
                    #print(x_idx, y_idx, "blue")
            #print(A)
            data = A+0.5 #for colouring purpose, the range is between discrete numbers
            #print(data)
            # create discrete colormap
            colours = ['blue', '#d3d3d3','black','#A9A9A9', 'white'] 
            #colours = ['blue', '#d3d3d3',(0.1, 0.2, 0.5),'#A9A9A9', 'white']
            #print("RGBA", colors.to_rgba('blue', alpha=None))
            cmap = colors.ListedColormap(colours)
            bounds = np.arange(0, len(colours)+1, 1)
            norm = colors.BoundaryNorm(bounds, cmap.N)
            fig, ax = plt.subplots()
            ax.imshow(data, cmap=cmap, norm=norm, origin="upper")
            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
            ax.set_xticks(np.arange(-.5, self.q, 1));
            ax.set_yticks(np.arange(-.5, self.p, 1));
            plt.show()
            print("****end of grid display")
        
        '''end of grid display'''
        
        '''grid separator data for display at the end of algorithm'''
        self.Nks.append(N[k])
        '''end of data for display'''
        
        #tried[N[k]] = 1 #mark all i \in N_k as tried
        if self.visu:
            self.separate_placed_rounds.append(separate_placed_round)

    '''Combining both normalize and separate stage'''
    def elimination_ordering(self, graph, log=False):
        #alternate normalize and separate while the graph is not empty
        #i=0
        while np.sum(graph) > 0:
            
            '''for visu:'''
            if self.visu and self.round < 1:
                print("===================================")
                print(">>>> Round Iteration",self.round,":")
            '''end of visu'''
            
            if np.sum(graph) == 0:
                break
            if log:
                print("Normalize:")
            self.normalize(graph)
            if log:
                print("e, w, first_zero, last_zero, deleted", self.e, self.w, self.first_zero, self.last_zero, np.where(self.deleted == True))
            if np.sum(graph) == 0:
                break
            '''for visu; print normalize results'''
            if self.visu and self.round < 1:
                print("current e_vector: ",self.e)
                print()
            '''end of visu'''
            if log:
                print("\n Separate:")
            self.separate(graph)
            '''for visu; print separate results'''
            if self.visu and self.round < 1:
                print("current e_vector: ",self.e)
                print()
            '''end of visu'''
            if log:
                print("e, w, first_zero, last_zero, deleted \n", self.e, self.w, self.first_zero, self.last_zero, np.where(self.deleted == True))
            #print(graph, merge_forest)
            if log:
                print("==================NEW ROUND======================= \n")
            #i += 1
            
            self.round += 1
        
        if self.visu:
            #print the total number of times each rule happens from normalize stage for all rounds:
            self.R_counters = self.R_counters.astype(np.int64, copy=False)          
            print("Total number of times each rule in Normalize stage is effective for all rounds:")
            for i in range(self.R_counters.shape[0]):
                print("Rule",i+1,": ",self.R_counters[i],"times")
            #print the total placed vertices in separate stage per round
            if len(self.separate_placed_rounds) > 0:
                print("Total number of vertices placed in Separate stage per round:")
                for i,r in enumerate(self.separate_placed_rounds):
                    print("Round",i+1,":",r,"vertices")
                
        if self.visu:
            return (self.e, self.R_counters, self.separate_placed_rounds)
        else:
            return self.e 
         
'''separate stage helper'''
'''dijkstra https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm''' 

#function to find max valency from nodes
def get_max_valency(subset_nodes, valencies):
    '''
    O(m*n) work instead of O(n) of the previous version
    '''
    max_valency = -float("inf")
    max_vertex = None
    '''
    #get index of subset nodes in the full nodes:
    m_idx = []
    for m in subset_nodes:
        m_idx.append(nodes.index(m))
    '''
    #sorting:
    for m in subset_nodes:
        if valencies[m] > max_valency:
            max_valency = valencies[m]
            max_vertex = m
    return max_vertex, max_valency
'''end of helper'''

'''normalize-helper functions:'''
def get_ordered_list_merged_vertex(forest, placed_vertex):
    '''get ordered list from the placed vertex.
    the idea is to traverse using BFS from the root (placed_vertex) all the way to the very first merged vertex,
    returns the reversed BFS result
    '''
    ordered_list = list(nx.bfs_edges(forest, placed_vertex))
    ordered_list = [placed_vertex] + [v for u,v in ordered_list]
    return list(reversed(ordered_list))

def clique_check(graph, nodelist):
    H = graph.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n*(n-1)/2


def check_subset(graph, neighbours):
    bool_subset = False
    j_get = None
    for j_node in neighbours:
        #probably need to be stopped earlier? instead of taking the last neighbour index
        gamma_j = list(graph[j_node])
        j_T = np.append(gamma_j, j_node) #j^up_tack = j union gamma(j):= j added to its neighbours
        if set(neighbours).issubset(set(j_T)): #gamma(i) \subset j^up_tack
            bool_subset = True
            j_get = j_node
            break #stop when found
    return bool_subset, j_get

def clique_check_1(graph, gamma_i):
    clique = True
    for k in gamma_i:
        for j in gamma_i:
            if k!=j:
                if not graph.has_edge(k, j):
                    clique = False
                    break
        if clique == False:
            break
    return clique

def check_subset_1(graph, gamma_i):
    bool_subset = False
    j_get = None
    H = gamma_i.copy()
    for k in gamma_i:
        for j in gamma_i:
            if k!=j:
                #print(k, j, H)
                if not H:
                    break #stop when H is empty
                if not graph.has_edge(k, j) and j in H:
                    H.remove(j)
    if H:
        j_get = H[0]
        bool_subset = True
    #print(H)
    return bool_subset, j_get

'''============== Utilities =============='''
def eliminate(graph, elimination_order, join_tree=False):
    '''elimination function: eliminates the vertices based on the order resulted from elimination ordering algorithms
    - takes in the vertices order from the any elimination ordering algorithms (e.g. METIS' nested dissection)
    - fill will be added when the "center" of the vertex is eliminated, e.g., 1-2-3, eliminate 2, fill(1-3), fill_count+=1
    - for now, assume the fill will be comb(n,2), so if there are 3 vertices which depend on an eliminated vertex, there will be 6 fills
    - if join_tree = True, then the procedure of generating C_v and separator indexes for join tree will be executed, otherwise it will be skipped
    '''
    count_fill = 0
    if join_tree:
        C_vs = []
        sep_idxs = []
    for v in elimination_order:
        #find neighbours and fill the fill-in indexes:
        K_v = np.array(list(graph[v]))
        if join_tree:
            C_v = np.array([v] + [w for w in K_v])
            sep_idx = 0 #separator index of (J|K), e.g. 5|29, meaning sepidx = 0; 52|9, meaning sepidx = 1
            C_vs.append(C_v); sep_idxs.append(sep_idx)
        fill_idxs = list(combinations(K_v, 2))
        if len(fill_idxs) > 0:
            for fill in fill_idxs:
                if not graph.has_edge(fill[0], fill[1]):
                    graph.add_edge(fill[0], fill[1]) #add fill
                    count_fill += 1
        graph.remove_node(v) #eliminate v
    return_data = None
    if join_tree:
        return_data = (count_fill, graph, C_vs, sep_idxs) 
    else:
        return_data = (count_fill, graph)
    return return_data

def absorption(e, C_vs, sep_idxs):
    '''absorption method, takes in elimination ordering, C_v and separator indexes'''
    length = len(e)
    absorbed = np.array([False]*length)
    for i in range(length):
        #print("C_vs[i]", C_vs[i])
        for j in range(i, length): #absorb into the earliest:
            if (set(C_vs[j]) < set(C_vs[i])) and (absorbed[j] == False): #check if C_v[j] is in C_v[i]
                sep_idxs[i] += sep_idxs[j]+1 #move separator of i by sep_idx[j]+1
                absorbed[j] = True #delete C_v[j] by marking it as "abvsorbed"

    C_vs = np.array(C_vs)
    C_vs = np.delete(C_vs, np.where(absorbed == True)[0])
    sep_idxs = np.delete(sep_idxs, np.where(absorbed == True)[0])
    #print(C_vs, sep_idxs)
    #calculate C_v and K_v sizes:
    length = C_vs.shape[0]
    C_sizes = np.array([C_v.shape[0] for C_v in C_vs]) 
    K_sizes = np.array([C_sizes[i]-sep_idxs[i]-1 for i in range(length)]) #K_size = C_size - sep_idx - 1
    max_C = np.max(C_sizes)
    max_K = np.max(K_sizes)
    return C_sizes, K_sizes, max_C, max_K
    
def adj_tuples_to_metis_file(graph, filename):
    '''write nx adjacency (tuples of edge) to file'''
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    first_line = np.array([n_nodes, n_edges]) #[nodes, edges]
    adj_list = []
    for i in range(n_nodes):
        neighbours = np.array(graph[i]) + 1 #metis indexing starts from 1
        adj_list.append(neighbours)
    adj_list = np.array(adj_list)
    
    with open(filename,"w") as f:
        f.write(str(first_line[0])+" "+str(first_line[1])+"\n")
        for i in range(adj_list.shape[0]):
            for j in range(adj_list[i].shape[0]):
                f.write(str(adj_list[i][j])+" ")
            f.write("\n")
    print("writing",filename,"done!")
    
def iperm_to_orderlist(filename):
    '''read iperm from ndmetis and convert it to list'''
    f = open(filename, "r")
    order = []
    for x in f:
        order.append(int(x))
    order = np.array(order)
    #according to metis documentatoin:
    actual_order = np.zeros(order.shape[0])
    for i in range(order.shape[0]):
        actual_order[i] = np.where(order == i)[0]
    actual_order = actual_order.astype(np.int64, copy=False)
    return actual_order


def load_matrix_market(filename):
    '''skipped for now '''
    '''test using matrices from matrix market'''
    #filename = "matrices/bcsstm01.mtx.gz"
    #metadata = mminfo(filename)
    Matrix = mmread(filename)
    A = Matrix.toarray()
    '''preprocess the matrix'''
    A = A.astype(np.int64, copy=False)
    return A


def grid_generator(p, q):
    '''p*q grid generator, p = row, q = col
    '''
    grid = nx.grid_graph((q,p)) #generate lattice grid q*p
    mapping = {}
    for x in range(q):
        for y in range(q):
            mapping[(x,y)] = x*q + y
    grid = nx.relabel_nodes(grid, mapping)
    return grid

def generate_separator_display(p,q,Nks):
    '''
    p = gridrow
    q = col
    Nks = separators
    
    display grids and the separator colours by increasing (R,G,B) per iteration
    '''
    length = len(Nks)
    A = np.zeros((p,q))+length #matrix color placeholder (white)
    norm_val = np.arange(0, length+1, 1) #discrete [0, length + 1] \in Z
    #fill color on coordinate:
    for i in range(p*q):
        vertex_id = i
        x_idx = vertex_id%q
        y_idx = int(vertex_id/q)
        #fill color on x_idx, y_idx:
        for j in range(length):
            if vertex_id in Nks[j]:
                A[y_idx, x_idx] = norm_val[j]
    

    offset = 0.5
    A += offset
    max_whiteness = 0.75
    colours = np.linspace(0, max_whiteness, length) #from black to gray-ish
    colours = [(col, col, col, 1) for col in colours] + [(1,1,1,1)] #discrete colormap, with alpha=1
    cmap = colors.ListedColormap(colours)
    bounds = np.arange(0, len(colours)+1, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(A, cmap=cmap, norm=norm, origin="upper")
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, q, 1));
    ax.set_yticks(np.arange(-.5, p, 1));
    plt.show()



if __name__ == "__main__":
    import time
    import cProfile

    '''
    p=64;q=64
    grid = grid_generator(p,q)
    eonx = elimination_ordering_class(grid, visualization=True, p=p, q=q)
    start= time.time()
    #eonx.separate(grid)
    eonx.elimination_ordering(grid)
    print(time.time()-start)
    print(eonx.e)
    grid = grid_generator(p,q)
    r = eliminate(grid, eonx.e, join_tree=True)
    print(r[0])
    r = absorption(eonx.e, r[2], r[3])
    print(r[2], r[3])
    '''
    
    
    p=128;q=128
    grid = grid_generator(p,q)
    eonx = elimination_ordering_class(grid, visualization=False, p=p, q=q)
    start = time.time()
    #eonx.elimination_ordering(grid)
    cProfile.run('eonx.elimination_ordering(grid)', sort='cumtime')
    print(time.time()-start)
    grid = grid_generator(p,q)
    r = eliminate(grid, eonx.e)
    print(r[0])
    
    '''
    graph = nx.Graph([(0,1),(0,2),(0,3),(1,3),(2,3),(1,2)])
    gamma_i = [0,1,2,3]
    #for time complexity checking:
    def _repeat():
        for i in range(1000000):
            clique_check(graph, gamma_i)
            #clique_check_1(graph, gamma_i)
    print(clique_check(graph, gamma_i))
    print(clique_check_1(graph, gamma_i))
    cProfile.run('_repeat()', sort='cumtime')
    '''