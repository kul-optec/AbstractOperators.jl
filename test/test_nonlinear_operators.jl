	
@printf("\nTesting non linear operators\n")

n = 4
x = randn(n)
b = randn(n)
op = Sigmoid(Float64,(n,),2)
println(op)

@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)

J = Jacobian(op,x)
println(J)

@test vecnorm(J*x-Jfd*x)<1e-6

