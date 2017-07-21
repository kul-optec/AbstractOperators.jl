	
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

@printf("\nTesting  calculus non linear operators\n")
#testing Scale
op = 30*op
println(op)
@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)
J = Jacobian(op,x)
println(J)

@test vecnorm(J*x-Jfd*x)<1e-6

#testing DCAT
n,m = 4,3
x = (randn(n),randn(m))
op = DCAT(MatrixOp(randn(n,n)),Sigmoid(Float64,(m,),2))
println(op)
@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)
J = Jacobian(op,x)

@test vecnorm((J*x)[1]-Jfd[1]*x[1])<1e-6
@test vecnorm((J*x)[2]-Jfd[2]*x[2])<1e-6

#testing HCAT
n,m = 4,3
x = (randn(n),randn(m))
op = HCAT(MatrixOp(randn(m,n)),Sigmoid(Float64,(m,),2))
println(op)
@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)
J = Jacobian(op,x)

@test vecnorm(Jfd*vcat(x...)-J*x)<1e-6

##testing VCAT
n,m = 4,3
x = randn(m)
op = VCAT(MatrixOp(randn(n,m)),Sigmoid(Float64,(m,),2))
println(op)
@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)
J = Jacobian(op,x)

@test vecnorm(Jfd*x-vcat((J*x)...))<1e-6

#testing Compose
l,n,m = 5,4,3
x = randn(m)
y = randn(l)
A = MatrixOp(randn(l,n))
B = Sigmoid(Float64,(n,),2)
C = MatrixOp(randn(n,m))
op = Compose(A,Compose(B,C))
println(op)
@test_throws ErrorException op'

Jfd = jacobian_fd(op,x)

op = Compose(A,Compose(B,C))
op*x            ###forward run is needed otherwise gradient is wrong!!
J = Jacobian(op,x)

@test vecnorm(Jfd'*y-J'*y)<1e-6

##testing Reshape
n = 4
x = randn(n)
b = randn(n)
op = Reshape(Sigmoid(Float64,(n,),2),2,2)
println(op)

Jfd = jacobian_fd(Sigmoid(Float64,(n,),2),x)
J = Jacobian(op,x)

@test vecnorm(Jfd*x-(J*x)[:])<1e-6

##testing Sum
m = 5
x = randn(m)
y = randn(m)
A = MatrixOp(randn(m,m))
B = Sigmoid(Float64,(m,),2)
op = Sum(A,B)
println(op)

Jfd = jacobian_fd(op,x)
J = Jacobian(op,x)

@test vecnorm(Jfd*x-J*x)<1e-6




