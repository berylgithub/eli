#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy.special import comb
from itertools import combinations
from scipy.io import mmread, mminfo
from itertools import chain

#for grid drawing only:
import matplotlib.pyplot as plt
from matplotlib import colors


# ### Start of Elimination Ordering.

# #### Normalize Stage

class elimination_ordering_class:
    def __init__(self, graph, visualization=False, p=0, q=0):
        '''function for data initialization, returns:
        - e vector placeholder
        - weight vector w
        - empty merge forest
        - first zero and last zero idxes
        - deleted bool array
        '''
        n = graph.shape[0]
        self.e = np.array([-1]*n) #for now the placeholder is an array of -1
        self.w = np.array([1]*n) #weight vector for merge forest
        self.merge_forest = np.zeros((n,n)) #merge forest for assessment criteria
        self.deleted = np.array([False]*n)
        self.first_zero = 0; self.last_zero = -1
        
        #for visualization
        self.round = 0
        self.visu = False
        if visualization:
            self.visu = True
            self.place_loc = np.zeros(n) #if the placement occurs in separate then set the element as "-1"
            self.rounds_e = np.zeros(n) #indicate the rounds of which the vertex is eliminated from
            
        #for grid information only:
        self.p = p #row
        self.q = q #col
        
        #specialized for normalize stage visualization:
        self.R_switch = False #true if sum(R_counters) >= 0 
        self.R_counters = np.zeros(6) #each index is for each rule, Ri = R_counters_{i-1}
        self.R_strings = [] #to be printed at the end of normalize stage
            
        #specialized for separate stage visualization:
        self.separate_placed_rounds = []
        
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
        #global deleted, e, w, first_zero, last_zero, merge_forest
        n = n_init = graph.shape[0] #number of nodes
        '''e = np.array([-1]*n) #for now the placeholder is an array of -1
        w = np.array([1]*n) #weight vector for merge forest
        merge_forest = np.zeros((n,n)) #merge forest for assessment criteria'''
        modified = np.array([1]*n) #modified = 1, otherwise 0'''


        #normalize stage
        #for now, cyclic ordering assumption: start from the 1st index to last idx, hence for-loop
        #need merge check for every node passed, by w[i] > 1
        #print("i, n, m, valency, e, summodified, firstzero, lastzero")
        while np.sum(modified) > 0:
            #print()
            #for i in range(n_init):
            '''for visu purpose, use the self.round'''
            if self.visu and self.round < 1:
                self.R_strings.append("++ new normalization cycle ++")
                self.R_strings.append("++ i, n, valency, m ++")
            for i in range(n_init):
                #check if it is already deleted, if yes, skip:
                if self.deleted[i]: #deleted in prev round
                    modified[i] = 0 #set modified to 0
                    #print("already deleted:",i)
                    continue
                if np.sum(modified) == 0:
                    break
                #recalculate all of the values:
                n = get_total_nodes(graph, n_init) #recalculate n by excluding zero vectored rows (disconnected vertices)
                valencies = np.array([np.sum(graph[j]) for j in range(n_init)]) #needs to recalculate the valency for each update due to the graph-change
                mean_valency = np.sum(valencies)/n #get mean valency
                max_valency = np.max(valencies) #get max valency
                valency = np.sum(graph[i]) #get vertex's valency
                m = np.min([mean_valency, np.floor(n**(1/4) + 3)])
                #m = np.floor(n**(1/4) + 3) #probably this is the correct interpretiation
                #print("mean_valency, np.floor(n**(1/4) + 3)",mean_valency, np.floor(n**(1/4) + 3))
                neighbours = np.where(graph[i] == 1)[0] #get the neighbours of i
                #print(i,n,m,valency,e,np.sum(modified),first_zero, last_zero)
                #print("neighbours",neighbours)
                #check all of the conditions based on the valency
                if valency == n-1:
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
                    graph[i] = graph[:,i] = 0  #remove from graph by deleting edges
                    self.deleted[i] = True
                    #graph = np.delete(graph, i, 0) #delete from graph -- this should be the proper deletion method, though not sure if it's faster
                    #graph = np.delete(graph, i, 1)
                    modified[neighbours] = 1 #set neighbours as modified
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 1, place "+str(i)+" last")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[0] += 1
                elif (valency > np.ceil(n/2)) and (valency == max_valency):
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
                    graph[i] = graph[:,i] = 0
                    self.deleted[i] = True
                    modified[neighbours] = 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 2, place "+str(i)+" last")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[1] += 1
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
                    graph[i] = graph[:,i] = 0
                    self.deleted[i] = True
                    modified[neighbours] = 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 3, place "+str(i)+" first")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[2] += 1
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
                    graph[neighbours[0]][neighbours[1]] = graph[neighbours[1]][neighbours[0]] = 1 #make edge between them -- fill the value of the cell with 1
                    graph[i] = graph[:,i] = 0
                    self.deleted[i] = True
                    modified[neighbours] = 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 4, place "+str(i)+" first")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[3] += 1
                elif (valency <= m) and (clique_check(graph, neighbours)):
                    #e.insert(0, i)
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
                    graph[i] = graph[:,i] = 0
                    self.deleted[i] = True
                    modified[neighbours] = 1
                    if self.visu and self.round < 1:
                        self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 5, place "+str(i)+" first")
                        self.R_switch = True
                    if self.visu:
                        self.R_counters[4] += 1
                elif (valency <= m): 
                    bool_subset, j_node = check_subset(graph, neighbours) #gamma(i) \subset j^uptack, j \in gamma(i)
                    if bool_subset:
                        self.merge_forest[j_node][i] = 1 #merge i into j, add directed edge j->i
                        self.w[j_node] += 1 #increment weight
                        graph[i] = graph[:,i] = 0
                        self.deleted[i] = True
                        modified[neighbours] = 1
                        if self.visu and self.round < 1:
                            self.R_strings.append(str(i)+" "+ str(n)+" "+ str(valency)+" "+ str(m)+ "||rule 6, merged"+str(i)+"to"+str(j_node))
                            self.R_switch = True
                        if self.visu:
                            self.R_counters[5] += 1
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
            
        n_init = graph.shape[0] #actual graph size

        '''RCM part'''
        #1, d=0, pick vertex e with max valency:
        d_prime = 0
        #n_nodes = get_total_nodes(graph, graph.shape[0]) #current total nodes
        valencies = np.array([np.sum(graph[i]) for i in range(n_init)])
        e_sep = np.argmax(valencies) #get the node with max valency
        if self.visu and self.round < 1:
            print("step 1, e, valency[e]:", e_sep, valencies[e_sep])
        
        #2, need to find a set of M with max distansce from e, which requires BFS or djikstra:
        #print("#2: ")
        distances, _ = dijkstra_shortest_path(graph, e_sep)
        #print("distances",distances)
        conn_components = np.where(distances != np.inf)[0] #indexes of connected components within the subgraph where e resides
        conn_distances = distances[conn_components] #distances of connected components (distances excluding inf)
        s = conn_components.shape[0] #total connected components
        d = np.max(conn_distances) #max distance from e
        M = np.where(distances == d)[0] #set of vertices with max distance from e
        #print("n_init, valencies, e_sep, s, d, M, conn_distances")
        #print(n_init, valencies, e_sep, s, d, M, conn_distances)
        
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
            max_vertex,_ = get_max_valency(M, valencies)
            #print("M, valencies",M, valencies)
            e_sep = max_vertex

            #do 2 again:
            distances, _ = dijkstra_shortest_path(graph, e_sep)
            conn_components = np.where(distances != np.inf)[0] #indexes of connected components within the subgraph where e resides
            conn_distances = distances[conn_components] #distances of connected components (distances excluding inf)
            s = conn_components.shape[0] #total connected components
            d = np.max(conn_distances) #max distance from e
            M = np.where(distances == d)[0] #set of vertices with max distance from e
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
            N.append(np.where(distances == i)[0])
            n[i] = len(N[i])
        if self.visu and self.round < 1:
            print("step 4, n_k:",n)
        '''
        #Version 2:
        #5, fill u_k, 0=<k<d; then look for k_0 where u_k_0 is closest to s/2:
        u = np.zeros(d)
        u[0] = s - n[0]
        for i in range(1,d):
            u[i] = u[i-1] - n[i]
        min_diff = np.inf
        min_idx = None
        threshold = s/2

        for i in range(d):
            diff = np.abs(u[i] - threshold)
            if diff < min_diff:
                min_diff = diff
                min_idx = i
        '''
        
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
        
        '''
        #Version 2 algorithm:
        #7, find k in {k0-1,k0,k0+1}\cap{1:d-1} with smallest n_k OR largest (n_{k+1} - n_k):
        cap = np.intersect1d([min_idx - 1, min_idx, min_idx + 1], np.array(range(1,d)))
        '''
        '''
        #smalles n_k:
        n_candidates = n[cap]
        idx = np.argmin(n_candidates)
        k = cap[idx]
        '''
        '''
        #largest n_{k+1} - n_k:
        n_candidates = [np.abs(n[c+1]-n[c]) for c in cap]
        idx = np.argmax(n_candidates)
        k = cap[idx]
        '''
        
        #4, initialization of several variables:
        #print("#4: ")
         
        '''temporarily disable blocks below:'''
        #k=0;
        #N=[np.array([e_sep])]; n=[1]; u = s-1;
        '''end of cc'''
        tried = np.array([0]*n_init); tried[e_sep] = 1
        #seploop = 0

        '''
        #disable while loop
        while True:
        '''

        #first line:
        #gamma_{k+1}(e):=get neighbours/set of points from e with the distance of k+1
        '''temporarily disable N and n assignments'''
        #N_next = np.where(distances == k+1)[0] #get the set of neighbours with distance = k+1
        #N.append(N_next)
        #n.append(len(N[k+1])) #or sum of weights?
        '''end of N and n assignments'''
        #u -= n[k+1]
        #print("k,N,n,u",k,N,n,u)

        #print("n_arr[k] <= n_arr[k+1] < n_arr[k+2]",n_arr[k] <= n_arr[k+1] < n_arr[k+2])

        #print("k+2, len(n_arr), d",k+2, len(n_arr), d)
        '''
        if k+2 < len(n_arr): #temporary fix, by skipping the block if k+2 >= len(n)
            if (n_arr[k] <= n_arr[k+1] < n_arr[k+2]):
                k += 1
                continue
        '''
        '''
        #disable while loop:
        if (k < d-1) and (n[k] <= n[k+1] < n[k+2]) and (u > 0.4*s): #another fix, by adding more skip-conditions
            k += 1
            if self.round < 1:
                print("(k < d-1) and (n[k] <= n[k+1] < n[k+2]) and (u > 0.4*s) condition reached, increment k")
            continue
        '''
        '''
        #second line, determining "in degrees":
        #c = {} #need to know which c corresponds to which node, so probably use a dictionary
        c = np.zeros(n_init)
        #j_idxs = N[k+1] #to keep track the used node indexes
        for node_j in N[k+1]: #indexing not by nodes' indices, but by c's internal index
            gamma_j = np.where(graph[node_j] == 1)[0]
            c[node_j] = (np.intersect1d(gamma_j, N[k])).shape[0]
            #print("gamma_j, N[k]",gamma_j, N[k])
        #print("c[j_idxs]",c[j_idxs])
        '''
        #step 8: compute the b_i for i\in N_k, sort N_k in increasing order of b_i:
        
        b = np.zeros(n_init)
        #i_idxs = N[k]
        for node_i in N[k]:
            gamma_i = np.where(graph[node_i] == 1)[0]
            out_w_nodes = np.intersect1d(gamma_i, N[k+1])
            b[node_i] = np.sum(self.w[out_w_nodes]) #w = weights from normalization, need to know which value belongs to which
            #print("gamma_i, N[k+1]",gamma_i, N[k+1])
        sorted_b_Nk_idx = np.argsort(b[N[k]])
        sorted_Nk = N[k][sorted_b_Nk_idx]
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
                graph[i] = graph[:,i] = 0
                self.deleted[i] = True
                
                
        
        '''
        #disable while loop
        if (u > 0.4*s) and (n[k+1] < n[k]): #threshold = 0.4s
            if self.round < 1:
                print("(u > 0.4*s) and (n[k+1] < n[k]) reached, breaking loop")
            break
        '''
        
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
        
        
#        #fourth line:
#        if self.visu and self.round < 1:
#            print("while n_k > 0: ")
#        while n[k] > 0:
#            #print("n[k]>0",n[k] > 0)
#            '''
#            if (u > 0.4*s) and (n[k+1] < n[k]): #threshold = 0.4s
#                #print("(u > 0.4*s) and (n[k+1] < n[k])",(u > 0.4*s) and (n[k+1] < n[k]))
#                break
#            '''
#            #place i with largest b_i last: (the rule should follow the placement rule in normalization)
#            #new condition to check, when b_i = 0, then break:
#            if np.sum(b) == 0:
#                if self.visu and self.round < 1:
#                    print("all b_i are zero, break loop")
#                #print("k,d,b,c",k,d,b,c)
#                break
#            
#            placed = np.argmax(b)
#            #print("placed",placed)
#            ##start of temporary fix
#            #if b[placed] > 0: #meaning, gamma(i) \intersect N_{k+1} is not {}
#            if self.w[placed] > 1:
#                ordered_list = get_ordered_list_merged_vertex(self.merge_forest, placed)
#                len_e = len(self.e)
#                len_ord_list = len(ordered_list)
#                self.e[len_e + self.last_zero - len_ord_list + 1 : len_e + self.last_zero + 1] = ordered_list
#                self.last_zero -= len_ord_list
#                if self.visu:
#                    self.place_loc[ordered_list] = -1
#                    self.rounds_e[ordered_list] = self.round
#                    separate_placed_round += len_ord_list
#            else:
#                self.e[self.last_zero] = placed
#                if self.visu:
#                    self.place_loc[placed] = -1
#                    self.rounds_e[placed] = self.round
#                    separate_placed_round += 1
#                self.last_zero -= 1
#            graph[placed] = graph[:,placed] = 0
#            self.deleted[placed] = True
#            if self.visu and self.round < 1:
#                print("largest b_i =",b[placed],", placed",placed,"last")
#            b[placed] = 0 #remove from b
#            
#            #print("e,fz,lz after placement:",e,first_zero,last_zero)
#            #decrement s, n_k, c_j:
#            #print("s,n[k],c",s,n[k],c)
#            s -= 1; n[k] -= 1; c[N[k+1]] -= 1
#            ##end of temporary fix#
#            #print("s,n[k],c",s,n[k],c)
#            #if c_j == 0: ......
#            #drop c_j from N; incr u; decr n[k+1]:
#            for node_j in N[k+1]:
#                if c[node_j] == 0:
#                    N[k+1] = N[k+1][N[k+1] != node_j] #drop cj from N
#                   #u += 1; n[k+1] -= 1
#                    #print("N, u, n, c[node_j]",N, u, n, c[node_j])
#                    if self.visu and self.round < 1:
#                        print("drop c_j = 0, j =",node_j)
        
        
        '''
        #disable while loop
        if n[k] == 0: 
            if self.round < 1:
                print("n_k = 0, break")
            break
        '''
        tried[N[k]] = 1 #mark all i \in N_k as tried
        
        '''
        #disable while loop
        k += 1 #, increment k
        '''
        if self.visu:
            self.separate_placed_rounds.append(separate_placed_round)
        
        '''
        if self.visu and self.round < 1:
            print("increment k, k =",k)
        seploop+=1
        '''
        
        

                #break #for loop breaking purpose during tests -- removed on actual scenario
            #break #for loop breaking purpose during tests -- removed on actual scenario

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
         
    
    
'''normalize-helper functions:'''
#for transforiming tree matrices to ordered list
def topological_sort_tree(tree_in):
    #look for the "first" node, which is the node with no incoming edges:
    tree = np.copy(tree_in) #copy the tree so the input wont be affected
    size = tree.shape[0]
    S = []
    for i in range(size):
        #need to exclude disconnected nodes by checking row-wise too:
        if np.sum(tree[:,i]) == 0:
            if np.sum(tree[i]) > 0:
                S.append(i)
    #print("S",S)

    enque = lambda q, x: q.append(x)
    deque = lambda q: q.pop(0)
    #kahn's algorithm for topological sort (https://en.wikipedia.org/wiki/Topological_sorting):
    #input: tree, first_nodes
    L = []
    while len(S) > 0:
        n = deque(S)
        L.append(n)
        ms = np.where(tree[n] == 1)[0]    #look for set of destination nodesf rom n (neighbours)
        #for each node m with an edge e from n to m:
        for m in ms:
            tree[n][m] = 0 #remove edge e from the graph
            if np.sum(tree[:,m]) == 0: #if m has no other incoming edges then
                enque(S, m) #insert m into S
    #there should be a final check whether the graph still has some edges, but it isnt necessary for tree cases since trees wont have DAG
    return L

#Breadth-First-Search traversal algorithm:
def BFS(graph, source):
    n_init = graph.shape[0]
    q = []
    enque = lambda q, x: q.append(x)
    deque = lambda q: q.pop(0)
    visited = np.array([0]*n_init)
    #distances = np.array([0]*n_init)
    visited[source] = 1
    enque(q, source)
    q_counter = 1 #to keep track how many neighbours enqueued
    path = []
    while q:
        v = deque(q)
        q_counter = q_counter - 1
        neighbours = np.where(graph[v] == 1)[0] #enque all v's neighbours (gamma(v))
        for node_i in neighbours:
            if (visited[node_i] == 0) and (node_i not in q):
                enque(q, node_i)
                visited[node_i] = 1
                q_counter += 1
        #print(v, neighbours, q, q_counter, depth)
        path.append(v)
    return path

#get ordered list from merge forest
def get_ordered_list_merged_vertex(tree, placed_vertex):
    '''algorithm for tree-tracing that covers all scenarios:
    0. transpose the tree (to get the reverse order), due to the nature of the merge procedure, the leaves will be the roots
    1. topological sort to get the root(s)
    2. determine the roots by checking the connections between vertices
    3. if there are more than one roots:
        BFS traverse starting from the placed node to get the ordered lists
    else:
        just use the list from the topological sort as the ordered list
    '''
    #print("edges:", np.where(tree.T == 1))
    topological_list = topological_sort_tree(tree.T)
    #print("topological_list",topological_list)
    '''
    print("topological_list",topological_list)
    print(tree)
    print(tree.T)
    '''
    #check the number of roots and get the corresponding roots:
    #length = len(topological_list)
    roots = [topological_list[0]]
    for i_elem in topological_list:
        non_root_found = False
        for j_elem in topological_list:
            if i_elem != j_elem:
                if tree.T[i_elem][j_elem] == 0:
                    #print(i_elem, j_elem)
                    roots.append(j_elem)
                else:
                    non_root_found = True
                    break
        if non_root_found:
            break
    #print("roots",roots)
    #if more than one roots, do BFS starting from the placed node, else just use the topological_list:
    ordered_list = None
    #print("ordlist bfs reversed:",list(reversed(BFS(tree, placed_vertex))))
    if len(roots) > 1:
        #ordered_list = BFS(tree.T, placed_vertex)
        ordered_list = list(reversed(BFS(tree, placed_vertex)))
        #print("orderedlist bfs",ordered_list)
    else:
        ordered_list = topological_list
        #print("orderedlist",topological_list)
    #print("ordered_list",ordered_list)
    return ordered_list

#clique checker:
def clique_check(graph, vertices_idx):
    #get subgraph, by slicing indexes:
    subgraph = graph[vertices_idx][:,vertices_idx]
    n = subgraph.shape[0]
    #check for clique from subgraph:
    upper_tri = subgraph[np.triu_indices(n, 1)]
    return np.sum(upper_tri) == comb(n, 2)

#subset checker:
def check_subset(graph, neighbours):
    bool_subset = False
    j_get = None
    for j_node in neighbours:
        #probably need to be stopped earlier? instead of taking the last neighbour index
        gamma_j = np.where(graph[j_node] == 1)[0]
        j_T = np.append(gamma_j, j_node) #j^up_tack = j union gamma(j):= j added to its neighbours
        if set(neighbours).issubset(set(j_T)): #gamma(i) \subset j^up_tack
            bool_subset = True
            j_get = j_node
            break #stop when found
    return bool_subset, j_get

#more accurate way of checking the total nodes within a graph, since the edge is represented 
#by the value of A[i][j] cell, e.g if i <-> j is connected, it means A[i][j] = A[j][i] 1, otherwise 0, 
#so the size of the matrix may not correspond to the total number of nodes
def get_total_nodes(graph, row_size):
    counter = 0
    for i in range(row_size):
        if np.sum(graph[i]) > 0:
            counter += 1
    return counter

'''end of normalize stage helper'''
         
'''separate stage helper'''
'''dijkstra https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm''' 
#function to help djikstra algorithm:
def get_min_distance_vertex(Q, distances):
    min_dist = float("inf")
    min_v = None
    for v in range(Q.shape[0]):
        if (distances[v] < min_dist) and (Q[v] == 1):
            min_dist = distances[v]
            min_v = v
    return min_dist, min_v

#start of dijkstra algorithm:
def dijkstra_shortest_path(graph, source):
    #n_init = get_total_nodes(graph, graph.shape[0])
    n_init = graph.shape[0]
    Q = np.array([1]*n_init)
    #print(Q)
    #source = root = 0
    distances = np.array([float("inf")]*n_init) #set distance vector
    distances[source] = 0
    prev = np.array([None]*n_init)

    while np.sum(Q) > 0:
        _, u = get_min_distance_vertex(Q, distances) #get the vertex with minimum distance
        Q[u] = False #remove u from Q
        neighbours = np.where(graph[u] == 1)[0]
#        print("len(Q), neighbours",len(Q), neighbours)
        for v in neighbours:
            if Q[v] == 1:
                alt = distances[u] + graph[u][v] #distance is equal to the weight of the edge between u and v
                if alt < distances[v]:
                    distances[v] = alt
                    prev[v] = u
                #print(alt)
    return distances, prev

#function to find max valency from nodes
def get_max_valency(subset_nodes, valencies):
    max_valency = -float("inf")
    max_vertex = None
    for m in subset_nodes:
        if valencies[m] > max_valency:
            max_valency = valencies[m]
            max_vertex = m
    return max_vertex, max_valency
'''end of helper'''



'''Utilities'''
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
        K_v = np.where(graph[v] == 1)[0]
        #C_v = np.union1d(v, K_v)
        if join_tree:
            C_v = np.array([v] + [w for w in K_v])
            sep_idx = 0 #separator index of (J|K), e.g. 5|29, meaning sepidx = 0; 52|9, meaning sepidx = 1
            C_vs.append(C_v); sep_idxs.append(sep_idx)
        fill_idxs = list(combinations(K_v, 2))
        if len(fill_idxs) > 0:
            for fill in fill_idxs:
                if graph[fill[0]][fill[1]] == 0:
                    graph[fill[0]][fill[1]] = graph[fill[1]][fill[0]] = 1
                    count_fill += 1
        graph[v] = graph[:,v] = 0 #eliminate v
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

def adj_mat_to_metis_file(graph, filename):
    '''write adjacency matrix to file'''
    first_line = np.array([graph.shape[0], int(np.sum(graph)/2)]) #[nodes, edges]
    adj_list = []
    for i in range(graph.shape[0]):
        neighbours = np.where(graph[i] == 1)[0]
        neighbours += 1
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

def load_matrix_market(filename, get_mat_meta=False):
    '''test using matrices from matrix market'''
    #filename = "matrices/bcsstm01.mtx.gz"
    metadata = mminfo(filename)
    Matrix = mmread(filename)
    A = Matrix.toarray()
    #print(A)
    #print(np.nonzero(A))
    '''preprocess the matrix'''
    A = A.astype(np.int64, copy=False)
    '''    
    #symmetrize the matrix:
    A = A + A.T
    #print("symmetrize:")
    #print(A)
    #set diagonals to zero:
    np.fill_diagonal(A, 0)
    #print("diag")
    #print(A)
    #if a nonzero element is >0 or <0, set it to 1:
    #print("nz")
    A[np.nonzero(A)] = 1
    #print(A)
    '''
    if get_mat_meta:
        return A, metadata
    else:
        return A

def bipartization(G):
    '''square -> bipartite'''
    A = np.copy(G)
    len_A = A.shape[0]
    B = np.zeros([len_A*2, len_A*2])
    len_B = B.shape[0]
    B[:len_A,len_A:len_B] = A #top right
    B.T[:len_A,len_A:len_B] = A #bottome left
    B[np.nonzero(B)] = 1
    return B    

def symmetrization(G):
    '''square -> symmetric'''
    A = np.copy(G)
    #symmetrize the matrix:
    A = A + A.T
    #set diagonals to zero:
    np.fill_diagonal(A, 0)
    #if a nonzero element is >0 or <0, set it to 1:
    #print("nz")
    A[np.nonzero(A)] = 1
    return A

def grid_generator(p, q, k, p_dep=0, q_dep=0):
    '''p*q grid generator, p = row, q = col
    if p_dep = 1: p=q,
    elif p_dep = 2: p=q**2
    else p is an input parameter
    if q_dep = 1: q = 2**k
    else q is an input parameter
    '''
    if q_dep == 1:
        q = 2**k
    if p_dep == 1:
        p = q
    elif p_dep == 2:
        p = q**2
    #print('p,q',p,q)
    grid = np.zeros((p*q, p*q)) #grid matrix
    diag = np.ones(q)
    sub_grid = np.zeros((q,q))
    np.fill_diagonal(grid[q:], diag) #lower diagonal main grid
    np.fill_diagonal(grid[:,q:], diag) #upper diagonal main grid
    np.fill_diagonal(sub_grid[1:], diag) #lower diagonal for subgrid
    np.fill_diagonal(sub_grid[:,1:], diag) #upper diagonal for subgrid
    for i in range(p):
        grid[i*q:(i*q)+q, i*q:(i*q)+q] = sub_grid
    return grid


    


if __name__ == "__main__":
    p=5 #grid row
    q=25 #grid col
    grid = grid_generator(p,q,0) #generate grid matrix
    
    #elimination ordering:
    EO = elimination_ordering_class(grid, visualization=True, p=p, q=q) #must be on global scope
    e,R_counters,separate_placed_rounds = EO.elimination_ordering(grid)
    print(e, EO.place_loc, EO.rounds_e)
    grid = grid_generator(p,q,0) #generate grid matrix
    fills,_ = eliminate(grid, e)
    print("fills = ",fills)
    print(len(set(e)) == len(e))