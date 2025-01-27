@printf("\nTesting  calculus non linear operators\n")
##testing Scale
m = 3
x = randn(m)
r = randn(m)
A = Sigmoid(Float64, (m,), 2)
op = 30 * A

y, grad = test_NLop(op, x, r, verb)

Y = 30 * (A * x)
@test norm(Y - y) < 1e-8

m = 3
x = randn(m)
r = randn(m)
A = Pow(Float64, (m,), 2)
op = -A

y, grad = test_NLop(op, x, r, verb)

Y = -A * x
@test norm(Y - y) < 1e-8

#testing DCAT
n, m = 4, 3
x = ArrayPartition(randn(n), randn(m))
r = ArrayPartition(randn(n), randn(m))
A = randn(n, n)
B = Sigmoid(Float64, (m,), 2)
op = DCAT(MatrixOp(A), B)

y, grad = test_NLop(op, x, r, verb)

Y = ArrayPartition(A * x.x[1], B * x.x[2])
@test norm(Y - y) < 1e-8

#testing HCAT
n, m = 4, 3
x = ArrayPartition(randn(n), randn(m))
r = randn(m)
A = randn(m, n)
B = Sigmoid(Float64, (m,), 2)
op = HCAT(MatrixOp(A), B)

y, grad = test_NLop(op, x, r, verb)

Y = A * x.x[1] + B * x.x[2]
@test norm(Y - y) < 1e-8

m, n = 3, 5
x = ArrayPartition(randn(m), randn(n))
r = randn(m)
A = Sin(Float64, (m,))
M = randn(m, n)
B = MatrixOp(M)
op = HCAT(A, B)

y, grad = test_NLop(op, x, r, verb)

Y = A * x.x[1] + M * x.x[2]
@test norm(Y - y) < 1e-8

p = [2, 1]
opP = AbstractOperators.permute(op, p)
xp = ArrayPartition(x.x[p]...)
J = Jacobian(opP, xp)'
println(size(J, 1))
y, grad = test_NLop(opP, xp, r, verb)

#testing VCAT
n, m = 4, 3
x = randn(m)
r = ArrayPartition(randn(n), randn(m))
A = randn(n, m)
B = Sigmoid(Float64, (m,), 2)
op = VCAT(MatrixOp(A), B)

y, grad = test_NLop(op, x, r, verb)

Y = ArrayPartition(A * x, B * x)
@test norm(Y - y) < 1e-8

#testing HCAT of VCAT
n, m1, m2, m3 = 4, 3, 2, 7
x1 = randn(m1)
x2 = randn(m2)
x3 = randn(m3)
x = ArrayPartition(x1, x2, x3)
r = ArrayPartition(randn(n), randn(m1))
A1 = randn(n, m1)
A2 = randn(n, m2)
A3 = randn(n, m3)
B1 = Sigmoid(Float64, (m1,), 2)
B2 = randn(m1, m2)
B3 = randn(m1, m3)
op1 = VCAT(MatrixOp(A1), B1)
op2 = VCAT(MatrixOp(A2), MatrixOp(B2))
op3 = VCAT(MatrixOp(A3), MatrixOp(B3))
op = HCAT(op1, op2, op3)

y, grad = test_NLop(op, x, r, verb)

Y = ArrayPartition(A1 * x1 + A2 * x2 + A3 * x3, B1 * x1 + B2 * x2 + B3 * x3)
@test norm(Y - y) < 1e-8

#testing VCAT of HCAT
m1, m2, m3, n1, n2 = 3, 4, 5, 6, 7
x1 = randn(m1)
x2 = randn(n1)
x3 = randn(m3)
x = ArrayPartition(x1, x2, x3)
r = ArrayPartition(randn(n1), randn(n2))
A1 = randn(n1, m1)
B1 = Sigmoid(Float64, (n1,), 2)
C1 = randn(n1, m3)
A2 = randn(n2, m1)
B2 = randn(n2, n1)
C2 = randn(n2, m3)
x = ArrayPartition(x1, x2, x3)
op1 = HCAT(MatrixOp(A1), B1, MatrixOp(C1))
op2 = HCAT(MatrixOp(A2), MatrixOp(B2), MatrixOp(C2))
op = VCAT(op1, op2)

y, grad = test_NLop(op, x, r, verb)

Y = ArrayPartition(A1 * x1 + B1 * x2 + C1 * x3, A2 * x1 + B2 * x2 + C2 * x3)
@test norm(Y - y) < 1e-8

#testing Compose

l, n, m = 5, 4, 3
x = randn(m)
r = randn(l)
A = randn(l, n)
C = randn(n, m)
opA = MatrixOp(A)
opB = Sigmoid(Float64, (n,), 2)
opC = MatrixOp(C)
op = Compose(opA, Compose(opB, opC))

y, grad = test_NLop(op, x, r, verb)

Y = A * (opB * (opC * x))
@test norm(Y - y) < 1e-8

## NN
m, n, l = 4, 7, 5
b = randn(l)
opS1 = Sigmoid(Float64, (n,), 2)
x = ArrayPartition(randn(n, l), randn(n))
r = randn(n)

A1 = HCAT(LMatrixOp(b, n), Eye(n))
op = Compose(opS1, A1)
y, grad = test_NLop(op, x, r, verb)

###testing Reshape
n = 4
x = randn(n)
r = randn(n)
opS = Sigmoid(Float64, (n,), 2)
op = Reshape(opS, 2, 2)

y, grad = test_NLop(op, x, r, verb)

Y = reshape(opS * x, 2, 2)
@test norm(Y - y) < 1e-8

###testing BroadCast
n, l = 4, 7
x = randn(n)
r = randn(n, l)
opS = Sigmoid(Float64, (n,), 2)
op = BroadCast(opS, (n, l))

y, grad = test_NLop(op, x, r, verb)

Y = (opS * x) .* ones(n, l)
@test norm(Y - y) < 1e-8

n, l = 1, 7
x = randn(n)
r = randn(n, l)
opS = Sigmoid(Float64, (n,), 2)
op = BroadCast(opS, (n, l))

y, grad = test_NLop(op, x, r, verb)

Y = (opS * x) .* ones(n, l)
@test norm(Y - y) < 1e-8

##testing Sum
m = 5
x = randn(m)
r = randn(m)
A = randn(m, m)
opA = MatrixOp(A)
opB = Sigmoid(Float64, (m,), 2)
op = Sum(opA, opB)

y, grad = test_NLop(op, x, r, verb)

Y = A * x + opB * x
@test norm(Y - y) < 1e-8

## AffineAdd and NonLinearOperator
n = 10
d = randn(n)
T = AffineAdd(Exp(n), d, false)

r = randn(n)
x = randn(size(T, 2))
y, grad = test_NLop(T, x, r, verb)
@test norm(y - (exp.(x) - d)) < 1e-8

## AffineAdd and Compose NonLinearOperator
n = 10
d1 = randn(n)
d2 = randn(n)
T = Compose(AffineAdd(Sin(n), d2), AffineAdd(Eye(n), d1))

r = randn(n)
x = randn(size(T, 2))
y, grad = test_NLop(T, x, r, verb)
@test norm(y - (sin.(x + d1) + d2)) < 1e-8

n = 10
d1 = randn(n)
d2 = randn(n)
d3 = pi
T = Compose(
	AffineAdd(Sin(n), d3), Compose(AffineAdd(Exp(n), d2, false), AffineAdd(Eye(n), d1))
)

r = randn(n)
x = randn(size(T, 2))
y, grad = test_NLop(T, x, r, verb)
@test norm(y - (sin.(exp.(x + d1) - d2) .+ d3)) < 1e-8

#### Axt_mul_Bx
n = 10
A, B = Eye(n), Sin(n)
P = Axt_mul_Bx(A, B)

x = randn(n)
r = randn(1)
y, grad = test_NLop(P, x, r, verb)
@test norm([(A * x)' * (B * x)] - y) < 1e-8

n, m = 3, 4
A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
P = Axt_mul_Bx(A, B)

x = randn(m)
r = randn(1)
y, grad = test_NLop(P, x, r, verb)
@test norm([(A * x)' * (B * x)] - y) < 1e-8

n, m, l = 3, 7, 5
A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
P = Axt_mul_Bx(A, B)
x = randn(m, l)
r = randn(l, l)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x)' * (B * x) - y) < 1e-8

n, m = 3, 7
A, B = Sin(n, m), Cos(n, m)
P = Axt_mul_Bx(A, B)
x = randn(n, m)
r = randn(m, m)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x)' * (B * x) - y) < 1e-8

# testing with HCAT
m, n = 3, 5
x = ArrayPartition(randn(m), randn(n))
r = randn(1)
b = randn(m)
A = AffineAdd(Sin(Float64, (m,)), b)
B = MatrixOp(randn(m, n))
op1 = HCAT(A, B)
C = Cos(Float64, (m,))
D = MatrixOp(randn(m, n))
op2 = HCAT(C, D)
P = Axt_mul_Bx(op1, op2)
y, grad = test_NLop(P, x, r, verb)
@test norm([(op1 * x)' * (op2 * x)] - y) < 1e-8

#test remove_displacement
y2, grad = test_NLop(remove_displacement(P), x, r, verb)
@test norm([(op1 * x - b)' * (op2 * x)] - y2) < 1e-8

# test permute
p = [2, 1]
Pp = AbstractOperators.permute(P, p)
xp = ArrayPartition(x.x[p])
y2, grad = test_NLop(Pp, xp, r, verb)
@test norm(y2 - y) < 1e-8

@test_throws Exception Axt_mul_Bx(Eye(2, 2), Eye(2, 1))
@test_throws Exception Axt_mul_Bx(Eye(2, 2, 2), Eye(2, 2, 2))

## Ax_mul_Bxt
n = 10
A, B = Eye(n), Sin(n)
P = Ax_mul_Bxt(A, B)
x = randn(n)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x)' - y) < 1e-9

n, m = 3, 4
A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
P = Ax_mul_Bxt(A, B)
x = randn(m)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x)' - y) < 1e-8

n, m, l = 3, 7, 5
A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
P = Ax_mul_Bxt(A, B)
x = randn(m, l)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x)' - y) < 1e-8

n, m = 3, 7
A, B = Sin(n, m), Cos(n, m)
P = Ax_mul_Bxt(A, B)
x = randn(n, m)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x)' - y) < 1e-8

# testing with HCAT
m, n = 3, 5
x = ArrayPartition(randn(m), randn(n))
r = randn(m, m)
b = randn(m)
A = AffineAdd(Sin(Float64, (m,)), b)
B = MatrixOp(randn(m, n))
op1 = HCAT(A, B)
C = Cos(Float64, (m,))
D = MatrixOp(randn(m, n))
op2 = HCAT(C, D)
P = Ax_mul_Bxt(op1, op2)
y, grad = test_NLop(P, x, r, verb)
@test norm((op1 * x) * (op2 * x)' - y) < 1e-8

#test remove_displacement
y2, grad = test_NLop(remove_displacement(P), x, r, verb)
@test norm((op1 * x - b) * (op2 * x)' - y2) < 1e-8

# test permute
p = [2, 1]
Pp = AbstractOperators.permute(P, p)
xp = ArrayPartition(x.x[p])
y2, grad = test_NLop(Pp, xp, r, verb)
@test norm(y2 - y) < 1e-8

@test_throws Exception Ax_mul_Bxt(Eye(2, 2), Eye(2, 1))
@test_throws Exception Ax_mul_Bxt(Eye(2, 2, 2), Eye(2, 2, 2))

## Ax_mul_Bx

n = 3
A, B = Eye(n, n), Eye(n, n)
P = Ax_mul_Bx(A, B)
x = randn(n, n)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm(x * x - y) < 1e-9

n = 3
A, B = Sin(n, n), Cos(n, n)
P = Ax_mul_Bx(A, B)
x = randn(n, n)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x) - y) < 1e-9

n = 3
A, B, C = Sin(n, n), Cos(n, n), Atan(n, n)
P = Ax_mul_Bx(A, B)
P2 = Ax_mul_Bx(C, P)
x = randn(n, n)
r = randn(n, n)
y, grad = test_NLop(P2, x, r, verb)
@test norm((C * x) * (A * x) * (B * x) - y) < 1e-9

n, l = 2, 3
A, B = MatrixOp(randn(l, n), l), MatrixOp(randn(l, n), l)
P = Ax_mul_Bx(A, B)
x = randn(n, l)
r = randn(l, l)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) * (B * x) - y) < 1e-8

@test_throws Exception Ax_mul_Bx(Eye(2), Eye(2))
@test_throws Exception Ax_mul_Bx(Eye(2, 2), Eye(2, 1))
@test_throws Exception Ax_mul_Bx(Eye(2, 2, 2), Eye(2, 2, 2))

# testing with HCAT
m, n = 3, 5
x = ArrayPartition(randn(n, n), randn(m, n))
r = randn(n, n)
b = randn(n, n)
A = AffineAdd(Sin(Float64, (n, n)), b)
B = MatrixOp(randn(n, m), n)
op1 = HCAT(A, B)
C = Sin(Float64, (n, n))
D = MatrixOp(randn(n, m), n)
op2 = HCAT(C, D)
P = Ax_mul_Bx(op1, op2)
y, grad = test_NLop(P, x, r, verb)
@test norm((op1 * x) * (op2 * x) - y) < 1e-8

#test remove_displacement
y2, grad = test_NLop(remove_displacement(P), x, r, verb)
@test norm((op1 * x - b) * (op2 * x) - y2) < 1e-8

# test permute
p = [2, 1]
Pp = AbstractOperators.permute(P, p)
xp = ArrayPartition(x.x[p])
y2, grad = test_NLop(Pp, xp, r, verb)
@test norm(y2 - y) < 1e-8

#### some combos of Ax_mul_Bx etc...
n, m, l = 3, 7, 5
A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
P = Ax_mul_Bxt(A, B)
P2 = Axt_mul_Bx(A, P)
x = randn(m, l)
r = randn(l, n)
y, grad = test_NLop(P2, x, r, verb)
@test norm((A * x)' * ((A * x) * (B * x)') - y) < 1e-8

n, m, l, k = 3, 7, 5, 9
A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
C = MatrixOp(randn(k, m), l)
P = Axt_mul_Bx(A, B)
P2 = Ax_mul_Bx(C, P)
x = randn(m, l)
r = randn(k, l)
y, grad = test_NLop(P2, x, r, verb)
@test norm((C * x) * ((A * x)' * (B * x)) - y) < 1e-8

n, m, l, k = 3, 7, 5, 9
A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
C = MatrixOp(randn(k, m), l)
P = Axt_mul_Bx(A, B)
P2 = Ax_mul_Bxt(C, P)
x = randn(m, l)
r = randn(k, l)
y, grad = test_NLop(P2, x, r, verb)
@test norm((C * x) * ((A * x)' * (B * x))' - y) < 1e-8

#### HadamardProd

n = 3
A, B = Eye(n, n), Eye(n, n)
P = HadamardProd(A, B)
x = randn(n, n)
r = randn(n, n)
y, grad = test_NLop(P, x, r, verb)
@test norm(x .* x - y) < 1e-9

n, l = 3, 2
A, B = Sin(n, l), Cos(n, l)
P = HadamardProd(A, B)
x = randn(n, l)
r = randn(n, l)
y, grad = test_NLop(P, x, r, verb)
@test norm((A * x) .* (B * x) - y) < 1e-9

# testing with HCAT
m, n = 3, 5
x = ArrayPartition(randn(m), randn(n))
r = randn(m)
b = randn(m)
A = AffineAdd(Sin(Float64, (m,)), b)
B = MatrixOp(randn(m, n))
op1 = HCAT(A, B)
C = Cos(Float64, (m,))
D = MatrixOp(randn(m, n))
op2 = HCAT(C, D)
P = HadamardProd(op1, op2)
y, grad = test_NLop(P, x, r, verb)
@test norm((op1 * x) .* (op2 * x) - y) < 1e-9

#test remove_displacement
y2, grad = test_NLop(remove_displacement(P), x, r, verb)
@test norm((op1 * x - b) .* (op2 * x) - y2) < 1e-8

# test permute
p = [2, 1]
Pp = AbstractOperators.permute(P, p)
xp = ArrayPartition(x.x[p])
y2, grad = test_NLop(Pp, xp, r, verb)
@test norm(y2 - y) < 1e-8

@test_throws Exception HadamardProd(Eye(2, 2, 2), Eye(1, 2, 2))
