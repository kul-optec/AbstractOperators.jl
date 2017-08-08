
@printf("\nTesting  calculus non linear operators\n")
#testing Scale
m = 3
x = randn(m)
op = Sigmoid(Float64,(m,),2)
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





##testing Hadamard
n,m,l = 4,5,6
A = randn(n,m)
B = randn(n,l)
x = (randn(m),randn(l))
P = Hadamard(MatrixOp(A),MatrixOp(B),x)
println(P)
y = P*x
@test norm((A*x[1]).*(B*x[2]) - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:m])<1e-6
@test norm(grad[2]-gradfd[m+1:end])<1e-6

n,m = 4,5
A = randn(n,m)
opS = Sigmoid(Float64,(n,),6)
x = (randn(n),randn(m))
P = Hadamard(opS,MatrixOp(A),x)
println(P)
y = P*x
@test norm((opS*x[1]).*(A*x[2]) - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:n])<1e-6
@test norm(grad[2]-gradfd[n+1:end])<1e-6

n= 4
opS1 = Sigmoid(Float64,(n,),6)
opS2 = Sigmoid(Float64,(n,),10)
x = (randn(n),randn(n))
P = Hadamard(opS1,opS2,x)
println(P)
y = P*x
@test norm((opS1*x[1]).*(opS2*x[2]) - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:n])<1e-6
@test norm(grad[2]-gradfd[n+1:end])<1e-6


n,m,l = 4,5,6
a = randn(m)
b = randn(n)
opA = MatrixMul(a,l)
opB = MatrixMul(b,l)
X = (randn(l,m),randn(l,n))
P = Hadamard(opA,opB,X)
println(P)
y = P*X

@test norm(X[1]*a.*X[2]*b-y)<1e-6

J = Jacobian(P,x)
y = randn(l)

grad = J'*y
