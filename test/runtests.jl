using ExtraTrees
using Base.Test

# testing extrema
x = [0.1 0.2; 0.5 0.9; -0.1 1.1]
@test ExtraTrees.extrema(x, [1,2,3], 1) == (-0.1, 0.5)
@test ExtraTrees.extrema(x, [1,2,3], 2) == ( 0.2, 1.1)
@test ExtraTrees.extrema(x, [2,1],   1) == ( 0.1, 0.5)

