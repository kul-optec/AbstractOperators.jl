@printf("\nTesting linear operators syntax\n")

###### ' ######
n, m = 5, 3
A = randn(n, m)
B = randn(n, m)
C = randn(n, m)
x1 = randn(m)
x2 = randn(n)
opA = MatrixOp(A)
opB = MatrixOp(B)
opC = MatrixOp(C)

opt = opA'
y1 = opt * x2
y2 = A' * x2
@test norm(y1 - y2) < 1e-9

######+,-######
opp = +opA
y1 = opp * x1
y2 = A * x1
@test norm(y1 - y2) < 1e-9

opm = -opA
y1 = opm * x1
y2 = -A * x1
@test norm(y1 - y2) < 1e-9

ops = opA + opB
y1 = ops * x1
y2 = A * x1 + B * x1
@test norm(y1 - y2) < 1e-9

ops = opA - opB
y1 = ops * x1
y2 = A * x1 - B * x1
@test norm(y1 - y2) < 1e-9

ops = opA + opB
opss = ops + opC
y1 = opss * x1
y2 = A * x1 + B * x1 + C * x1
@test norm(y1 - y2) < 1e-9

ops = opB + opC
opss = opA - ops
y1 = opss * x1
y2 = A * x1 - B * x1 - C * x1
@test norm(y1 - y2) < 1e-9

ops = opB + opC
opss = ops - opA
y1 = opss * x1
y2 = -A * x1 + B * x1 + C * x1
@test norm(y1 - y2) < 1e-9

###### * ######

n, m, l = 5, 3, 7
A = randn(n, m)
B = randn(l, n)
x1 = randn(m)
opA = MatrixOp(A)
opB = MatrixOp(B)
Id = Eye(m)
alpha = pi
beta = 4

ops = alpha * opA
y1 = ops * x1
y2 = alpha * A * x1
@test norm(y1 - y2) < 1e-9

ops = alpha * opA
opss = beta * ops
y1 = opss * x1
y2 = beta * alpha * A * x1
@test norm(y1 - y2) < 1e-9

ops1 = alpha * opA
ops2 = beta * opB
opss = ops2 * ops1
y1 = opss * x1
y2 = beta * B * alpha * A * x1
@test norm(y1 - y2) < 1e-9

opc = opB * opA
y1 = opc * x1
y2 = B * A * x1
@test norm(y1 - y2) < 1e-9

opc = Id * opA
y1 = opc * x1
y2 = A * x1
@test norm(y1 - y2) < 1e-9

opc = opA * Id
y1 = opc * x1
y2 = A * x1
@test norm(y1 - y2) < 1e-9

opc = Id * Id
y1 = opc * x1
y2 = x1
@test norm(y1 - y2) < 1e-9

###### getindex ######

ops = opA[1:(n - 1)]
y1 = ops * x1
y2 = (A * x1)[1:(n - 1)]
@test norm(y1 - y2) < 1e-9

opF = DCT(n, m, l)
x3 = randn(n, m, l)
ops = opF[1:(n - 1), :, 2:l]
y1 = ops * x3
y2 = dct(x3)[1:(n - 1), :, 2:l]
@test norm(y1 - y2) < 1e-9

opF = DCT(n, m, l)
x3 = randn(n, m, l)
ops = opF[1:(n - 1), 2:m, 1]
y1 = ops * x3
y2 = dct(x3)[1:(n - 1), 2:m, 1]
@test norm(y1 - y2) < 1e-9

opV = Variation(n, m, l)
ops = opV[1:4]
y1 = ops * x3
y2 = (opV * x3)[1:4]
@test norm(y1 - y2) < 1e-9

ops = (opB * opA)[1:(l - 1)]
y1 = ops * x1
y2 = (B * A * x1)[1:(l - 1)]
@test norm(y1 - y2) < 1e-9

ops = (10.0 * opA)[1:(n - 1)]
y1 = ops * x1
y2 = (10 * A * x1)[1:(n - 1)]
@test norm(y1 - y2) < 1e-9

#slicing HCAT

n, m1, m2, m3 = 5, 6, 7, 8
A = randn(n, m1)
B = randn(n, m2)
C = randn(n, m3)
x1 = randn(m1)
x2 = randn(m2)
x3 = randn(m3)
opA = MatrixOp(A)
opB = MatrixOp(B)
opC = MatrixOp(C)
opH = HCAT(opA, opB, opC)
opH2 = opH[1:2]
y1 = opH2 * ArrayPartition(x1, x2)
y2 = A * x1 + B * x2
@test all(norm.(y1 .- y2) .<= 1e-12)
opH3 = opH[3]
y1 = opH3 * x3
y2 = C * x3
@test all(norm.(y1 .- y2) .<= 1e-12)

opHperm = opH[[3, 1, 2]]
@test norm(opH * ArrayPartition(x1, x2, x3) - opHperm * ArrayPartition(x3, x1, x2)) < 1e-12

@test opHperm[1] == opC
@test opHperm[2] == opA
@test opHperm[3] == opB

opHperm = opH[[3, 1]]
@test norm(opC * x3 + opA * x1 - opHperm * ArrayPartition(x3, x1)) < 1e-12

# slicing Affine add of HCAT
d = randn(n)
opHA = AffineAdd(opH, d)
@test norm(opHA[1] * x1 - (A * x1 + d)) < 1e-9
@test norm(opHA[2] * x2 - (B * x2 + d)) < 1e-9
@test norm(opHA[3] * x3 - (C * x3 + d)) < 1e-9

m4 = 9
x4 = randn(m4)
D = randn(n, n)
E = randn(n, m4)
opD = MatrixOp(D)
opE = MatrixOp(E)
opCH = opD * opH

opHCH = HCAT(opCH, opE)

opH4 = opHCH[4]
@test opH4 == opE
@test_throws ErrorException opHCH[1]
@test_throws ErrorException opHCH[1:2]

#slicing VCAT

n1, n2, n3, m = 5, 6, 7, 8
A = randn(n1, m)
B = randn(n2, m)
C = randn(n3, m)
x1 = randn(m)
opA = MatrixOp(A)
opB = MatrixOp(B)
opC = MatrixOp(C)
opV = VCAT(opA, opB, opC)
opV2 = opV[1:2]
y1 = opV2 * x1
y2 = ArrayPartition(A * x1, B * x1)
@test norm(y1 - y2) <= 1e-12
opV3 = opV[3]
y1 = opV3 * x3
y2 = C * x3
@test norm(y1 - y2) <= 1e-12

###### hcat ######

n, m1, m2 = 5, 6, 7
A = randn(n, m1)
B = randn(n, m2)
opA = MatrixOp(A)
opB = MatrixOp(B)
opH = [opA opB]
x1 = randn(m1)
x2 = randn(m2)
y1 = opH * ArrayPartition(x1, x2)
y2 = [A B] * [x1; x2]
@test norm(y1 - y2) <= 1e-12

opHH = [opH opB]
y1 = opHH * ArrayPartition(x1, x2, x2)
y2 = [A B B] * [x1; x2; x2]
@test norm(y1 - y2) <= 1e-12

###### vcat ######

n, m1, m2 = 5, 6, 7
A = randn(m1, n)
B = randn(m2, n)
opA = MatrixOp(A)
opB = MatrixOp(B)
opH = [opA; opB]
x1 = randn(n)
y1 = opH * x1
y2 = ArrayPartition(A * x1, B * x1)
@test norm(y1 - y2) <= 1e-12

opVV = [opA; opH]
y1 = opVV * x1
y2 = ArrayPartition(A * x1, A * x1, B * x1)
@test norm(y1 - y2) <= 1e-12

###### reshape ######
n, m = 10, 5
A = randn(n, m)
x1 = randn(m)
opA = MatrixOp(A)
opR = reshape(opA, 2, 5)
opR = reshape(opA, (2, 5))
y1 = opR * x1
y2 = reshape(A * x1, 2, 5)
@test norm(y1 - y2) <= 1e-12

# testing ndims & ndoms
L = Variation((3, 4, 5))
@test ndims(L) == (2, 3)
@test ndims(L, 1) == 2
@test ndims(L, 2) == 3
@test ndoms(L) == (1, 1)
H = hcat(L, L)
@test ndims(H) == (2, (3, 3))
@test ndims(H, 1) == 2
@test ndims(H, 2) == (3, 3)
@test ndoms(H) == (1, 2)
@test ndoms(H, 1) == 1
@test ndoms(H, 2) == 2
D = DCAT(L, L)
@test ndims(D) == ((2, 2), (3, 3))
@test ndoms(D) == (2, 2)

###### jacobian ######
n, m = 10, 5
A = MatrixOp(randn(n, m))
B = Sigmoid(Float64, (n,), 100.0)
op = B * A
J = jacobian(op, randn(m))

#### convert ####
L = Eye(10)
LL = convert(AbstractOperator, Float64, (10,), L)
@test LL == L

LL = convert(LinearOperator, Float64, (10,), L)
@test LL == L

@test_throws MethodError convert(NonLinearOperator, Float64, (10,), L)

### displacement ###
L = Eye(10)
@test displacement(L) == 0.0
