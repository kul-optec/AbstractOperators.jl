# @printf("\nTesting deep operations\n")

x = ([2.0, 3.0], [4.0, 5.0, 6.0], [1.0 2.0 3.0; 4.0 5.0 6.0])

lengths_x = (2, 3, 6)
deeplength_x = 11
deepvecnorm_x = 13.45362404707371

@test length.(x) == lengths_x
@test AbstractOperators.deeplength(x) == deeplength_x

y = AbstractOperators.deepsimilar(x)

@test AbstractOperators.deeplength(y) == deeplength_x
@test length.(y) == lengths_x

AbstractOperators.deepcopy!(y, x)

@test y == x
@test AbstractOperators.deepvecnorm(x) ≈ deepvecnorm_x
@test AbstractOperators.deepvecdot(x, y) ≈ deepvecnorm_x^2
@test AbstractOperators.deepmaxabs(x .- y) == 0

t1 = AbstractOperators.deepzeros((Float32, Float64), ((3, ), (4, )) )
