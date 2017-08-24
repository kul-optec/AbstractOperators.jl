	
@printf("\nTesting non linear operators\n")

n = 4
x = randn(n)
r = randn(n)
op = Sigmoid(Float64,(n,),2)

y, grad = test_NLop(op,x,r,verb)

n,m,l = 4,5,6
x = randn(n,m,l)
r = randn(n,m,l)
op = Sigmoid(Float64,(n,m,l),2)

y, grad = test_NLop(op,x,r,verb)


