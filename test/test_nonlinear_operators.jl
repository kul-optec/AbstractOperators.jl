	
@printf("\nTesting non linear operators\n")

n = 4
x = randn(n)
r = randn(n)
op = Sigmoid(Float64,(n,),2)

y, grad = test_NLop(op,x,r,verb)

Jfd = jacobian_fd(op,x)
@test vecnorm(grad-Jfd'*r)<1e-6

