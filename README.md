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