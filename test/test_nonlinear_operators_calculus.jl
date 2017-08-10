
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

#testing HCAT of VCAT
n,m1,m2,m3 = 4,3,2,7
x1 = randn(m1)
x2 = randn(m2)
x3 = randn(m3)
x = (x1,x2,x3)
op1 = VCAT(MatrixOp(randn(n,m1)),Sigmoid(Float64,(m1,),2))
op2 = VCAT(MatrixOp(randn(n,m2)),MatrixOp(randn(m1,m2)))
op3 = VCAT(MatrixOp(randn(n,m3)),MatrixOp(randn(m1,m3)))
op = HCAT(op1,op2,op3)
println(op)
@test_throws ErrorException op'

J = Jacobian(op,x)
println(J)
#TODO finite diff test

#testing VCAT of HCAT
m1,m2,m3,n1,n2 = 3,4,5,6,7
x1 = randn(m1)
x2 = randn(n1)
x3 = randn(m3)
x = (x1,x2,x3)
op1 = HCAT(MatrixOp(randn(n1,m1)),Sigmoid(Float64,(n1,),2),MatrixOp(randn(n1,m3)))
op2 = HCAT(MatrixOp(randn(n2,m1)),MatrixOp(randn(n2,n1)),MatrixOp(randn(n2,m3)))
op = VCAT(op1,op2)
println(op)
@test_throws ErrorException op'

J = Jacobian(op,x)
println(J)
#TODO finite diff test

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

#testing Hadamard
n,m,l = 4,5,6
A = randn(n,m)
B = randn(n,l)
x = (randn(m),randn(l))
P = Hadamard(MatrixOp(A),MatrixOp(B))
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
P = Hadamard(opS,MatrixOp(A))
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
P = Hadamard(opS1,opS2)
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
P = Hadamard(opA,opB)
println(P)
y = P*X

@test norm(X[1]*a.*X[2]*b-y)<1e-6

J = Jacobian(P,X)
y = randn(l)

grad = J'*y

n,m1,m2,m3 = 4,5,6,7
A1 = zeros(n,m1)
A2 = randn(n,m2)
A3 = randn(n,m3)
B1 = randn(n,m1)
B2 = zeros(n,m2)
B3 = zeros(n,m3)
x = (randn(m1),randn(m2),randn(m3))

Y = (A1*x[1]+A2*x[2]+A3*x[3]).*(B1*x[1]+B2*x[2]+B3*x[3])

HA = HCAT(Zeros(Float64,(m1,),(n,)),MatrixOp(A2),MatrixOp(A3)) 
HB = HCAT(MatrixOp(B1),Zeros(Float64,(m2,),(n,)),Zeros(Float64,(m3,),(n,))) 

P = Hadamard(HA,HB)
println(P)
y = P*x
@test norm(Y - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:m1])<1e-6
@test norm(grad[2]-gradfd[m1+1:m1+m2])<1e-6
@test norm(grad[3]-gradfd[m1+m2+1:end])<1e-6

n,m1,m2,m3 = 4,5,6,7
A1 = randn(n,m1)
A2 = randn(n,m2)
A3 = randn(n,m3)
P = Hadamard(MatrixOp(A1),MatrixOp(A2),MatrixOp(A3))
x = (randn(m1),randn(m2),randn(m3))

Y = (A1*x[1]).*(A2*x[2]).*(A3*x[3])
y = P*x
@test norm(Y - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:m1])<1e-6
@test norm(grad[2]-gradfd[m1+1:m1+m2])<1e-6
@test norm(grad[3]-gradfd[m1+m2+1:end])<1e-6

n,m1,m2,m3 = 4,5,6,7
A1 = randn(n,m1)
A2 = randn(n,m2)
A3 = randn(n,m3)
P = Hadamard(HCAT(MatrixOp(A1),MatrixOp(A2)),MatrixOp(A3))
x = (randn(m1),randn(m2),randn(m3))
println(P)

Y = (A1*x[1]+A2*x[2]).*(A3*x[3])
y = P*x
@test norm(Y - y) <= 1e-12

J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)
y = randn(n)

grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]-gradfd[1:m1])<1e-6
@test norm(grad[2]-gradfd[m1+1:m1+m2])<1e-6
@test norm(grad[3]-gradfd[m1+m2+1:end])<1e-6

##testing NonLinearCompose
n,m = 3,4
x = (randn(1,m),randn(n))
A = randn(m,n)

Y = x[1]*(A*x[2])

P = NonLinearCompose( Eye(1,m), MatrixOp(A) )
println(P)
y = P*x

@test norm(Y - y) <= 1e-12
J = Jacobian(P,x)
Jfd = jacobian_fd(P,x)

y = [randn()]
grad = J'*y
gradfd = Jfd'*y

@test norm(grad[1]'-gradfd[1:m])<1e-6
@test norm(grad[2]-gradfd[m+1:end])<1e-6

l,m1,m2,n1,n2 = 2,3,4,5,6
X = (randn(m1,m2),randn(n1,n2))
A = randn(l,m1)
B = randn(m2,n1)

Y = A*X[1]*B*X[2]
P = NonLinearCompose( MatrixOp(A,m2), MatrixOp(B,n2) )
println(P)
y = P*X

@test norm(Y - y) <= 1e-12

J = Jacobian(P,X)
grad = J'*y
grad2 =  ((B*X[2])*(A'*Y)')', B'*(A*X[1])'*Y

@test vecnorm(grad[1]-grad2[1]) <1e-7
@test vecnorm(grad[2]-grad2[2]) <1e-7
















