
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

@test norm(grad[1]-gradfd[1:m])<1e-7
@test norm(grad[2]-gradfd[m+1:end])<1e-7

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

@test norm(grad[1]-gradfd[1:n])<1e-7
@test norm(grad[2]-gradfd[n+1:end])<1e-7

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

@test norm(grad[1]-gradfd[1:n])<1e-7
@test norm(grad[2]-gradfd[n+1:end])<1e-7


n,m,l = 4,5,6 
a = randn(m)
b = randn(n)
opA = MatrixMul(a,l)
opB = MatrixMul(b,l)
X = (randn(l,m),randn(l,n))
P = Hadamard(opA,opB,X)
println(P)
y = P*X

@test norm(X[1]*a.*X[2]*b-y)<1e-7

J = Jacobian(P,x)
y = randn(l)

grad = J'*y
