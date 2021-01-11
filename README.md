# eli

re-ordering of gaussian elimination

### Notes:
I. Normalization:
1. Need to fix the insertion (the first/last zero positions) during node placement/removal from graph:
    A = |x|x|0|0|x|, x = inserted vertices, "insert first":= A[2] = x, "insert last":= A[3] = x
    idx = 0 1 2 3 4
2. Need to indicate the "rule" of which the node is eliminated from.
3. Change the value of n, m, n_2
4. idx should be identical to vertex labeling (in this case of python labeling, idx_i = v_i + 1)
5. **look at the examples written in the papers from 8-1-2021
