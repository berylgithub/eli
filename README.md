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
5. **look at the examples written in the papers from 8-1-2021




******
found an algorithm "bug", this happens when n = 2 or when there are 2 vertices left, the 4th rule got overwritten by the 1st rule. However, if this is the case, the ordering of the last two placed vertices by the algorithm should not matter (e.g. ....|1|2|... == ....|2|1|...), although this claim needs further confirmation.. -- this has been confirmed, this case happens when the graph is symmetric, i.e. the order doesn't matter, so the assumption is correct