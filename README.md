# eli

re-ordering of gaussian elimination

### Notes:
I. Normalization:
1. Need to fix the insertion (the first/last zero positions) during node placement/removal from graph:
    A = |x|x|0|0|x|, x = inserted vertices, "insert first":= A[2] = x, "insert last":= A[3] = x
    idx = 0 1 2 3 4 - DONE
2. Need to indicate the "rule" of which the node is eliminated from. - DONE
3. Change the value of n, m, n_2 - DONE
4. idx should be identical to vertex labeling (in this case of python labeling, idx_i = v_i + 1)
5. look at the examples written in the papers from 8-1-2021
6. Still need to fix rule 6
7. Interchange R5 and R6 to see the effect of R6

******
when n = 2 or when there are 2 vertices left, the 4th rule got overwritten by the 1st rule. However, if this is the case, the ordering of the last two placed vertices by the algorithm should not matter (e.g. ....|1|2|... == ....|2|1|...), although this claim needs further confirmation.. -- Confirmed, this case happens when the graph is symmetric, i.e. the order doesn't matter, so the assumption is correct

******
For banded matrix (not sure if this is case specific or only applies to the example below), R6 is never reached (overwritten by R5) due to the clique property of gamma(i).
E.g., one of the banded matrix with entries: <br/>
from scipy.sparse import diags <br/>
A = diags([1,1, 1, 0, 1, 1,1], [-3,-2,-1, 0, 1,2,3], shape=(7, 7), dtype=int).toarray() <br/>
A = <br/>
[0,1,1,1,0,0,0], <br/>
[1,0,1,1,1,0,0], <br/>
[1,1,0,1,1,1,0], <br/>
[1,1,1,0,1,1,1], <br/>
[0,1,1,1,0,1,1], <br/>
[0,0,1,1,1,0,1], <br/>
[0,0,0,1,1,1,0] <br/>
 <br/>
examples for when gamma(i) is subset of j^uptack for some i but gamma(i) is not a clique: <br/>
1---2 <br/>
| \ | <br/>
4---3 <br/>
[0,1,1,1], <br/>
[1,0,1,0], <br/>
[1,1,0,1], <br/>
[1,0,1,0] <br/>

1---2 <br/>
| \ | <br/>
3---4 <br/>
[0,1,1,1], <br/>
[1,0,0,1], <br/>
[1,0,0,1], <br/>
[1,1,1,0]  

Update: after testing banded matrices with any bandwidth size, R6 was never reached. However, when the elmination ordering is used, it produces 0 fills.

=======================================  

II. Separation:
1. Firstzero & lastzero indexer, weight vector, and the graph from Normalization stage are set as input
2. "if n_k == 0; break", this condition is always reached in first iteration, due to N_0 = [e], n_0 = 1 and n_0 is decreased within the first loop. Therefore k is never incremented  
-- temp fix: check the n_k, 0<=k<=d, d=max distance, in the beginning after finding the true e (end node); add a monotonic increase condition before determining the values of indegrees and outweights: if n_k <= n_{k+1} < n_{k+2}: increase k, and continue (skip iteration)
3. The n_k <= n_{k+1} < n_{k+2} condition causes error if k+2 >= |n|
-- temp fix: skip the if block if k+2 >= |n|
4. if the largest b_i = 0, i = 0 is placed, which is incorrect. This happens when gamma(i) \intersect N_{K=1} = {}, for i \in N_k. If this is skipped, this causes an infinite loop due to the n_k never reaches 0.
-- tempfix: break if b_i = 0 for i \in N_k

========================================  

III. Combined Normalize + Separate:
1. After separation stage, when going back to normalization, the disconnected/placed node is processed again, due to modified_vector = reset, placed into R3. Will look for alternative fix.
-- temp fix: maintain a "deleted" vector which contains deleted nodes

After testing using nauru graph, eli differes in 2 fills:  
- eli : 58 fills
- metis : 56 fills ### Need to be checked further whether the ordering is correct or not: "The ordering file of a graph with n vertices consists of n lines with a single number per line. The ith line of the ordering file contains the new order of the ith vertex of the graph. The numbering in the ordering file starts from 0."

========================================== <br/>
30.03.2021:<br/>
Treating the graphs per-component changes the flow organization entirely. Previously, when the whole graph is treated, the flow is: <br/>
    N-S-N-S-......-graph = {} <br/>
    N := Normalize, S:= Separate <br/>
When each component is treated, the flow becomes:
    N-S-N-S-.... <br/>
     \ <br/>
     first_comp = {}<br/>
        \  first_comp := next_comp <br/>
         N-S-N-S-..... <br/>
              \ <br/>
              {} <br/>
               
i.e., when the treated component is empty (which only happens in Normalize stage), the algorithm **must** switch to the Normalize stage to treat the next non-empty component since if it alternates to Separate stage, it will cause error due to the final component being only n_1--n_2 graph.
The new flow is recursive, or possible to be non-recursive by introducing a stack which tracks the components, e.g., first_component, next_components. This flow implies that the notion of "rounds" is not relevant anymore.
========================================== <br/>
12.04.2021: <br/>
Reset modified tag after separation vs modify neighborhood tags during separation.