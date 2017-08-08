 @printf("\nTesting linear operators\n")

####### Conv ############
n,m = 5, 6
h = randn(m)
op = Conv(Float64,(n,),h)
x1 = randn(n)
y1 = test_op(op, x1, randn(n+m-1), verb)
y2 = conv(x1,h)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = Conv(x1,h)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

######### DCT ############
n = 4
op = DCT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = dct(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = DCT((n,))
op = DCT(n,n)
op = DCT(Complex{Float64}, n,n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == true
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

m = 10
op = DCT(n,m)
x1 = randn(n,m)

@test vecnorm(op'*(op*x1) - x1) <= 1e-12
@test diag_AAc(op) == 1.
@test diag_AcA(op) == 1.

######### IDCT ############
n = 4
op = IDCT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = idct(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = IDCT((n,))
op = IDCT(n,n)
op = IDCT(Complex{Float64}, n,n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == true
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

m = 10
op = IDCT(n,m)
x1 = randn(n,m)

@test vecnorm(op'*(op*x1) - x1) <= 1e-12
@test diag_AAc(op) == 1.
@test diag_AcA(op) == 1.

######## DFT ############
n = 4
op = DFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

op = DFT(Complex{Float64},(n,))
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = DFT((n,))
op = DFT(n,n)
op = DFT(Complex{Float64}, n,n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == false
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

m = 10
op = DFT(n,m)
x1 = randn(n,m)
y1 = op*x1
@test vecnorm(op'*(op*x1) - diag_AcA(op)*x1) <= 1e-12
@test vecnorm(op*(op'*y1) - diag_AAc(op)*y1) <= 1e-12

######### IDFT ############
n = 4
op = IDFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(n,))
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = IDFT((n,))
op = IDFT(n,n)
op = IDFT(Complex{Float64}, n,n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == false
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

m = 10
op = IDFT(n,m)
x1 = randn(n,m)
y1 = op*x1
@test vecnorm(op'*(op*x1) - diag_AcA(op)*x1) <= 1e-12
@test vecnorm(op*(op'*y1) - diag_AAc(op)*y1) <= 1e-12

######### DiagOp ############
n = 4
d = randn(n)
op = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = d.*x1

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
op = DiagOp(d)
op = DiagOp(Float64, d)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == true
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == false
@test is_invertible(DiagOp(ones(10)))       == true
@test is_invertible(DiagOp([ones(5);0]))    == false
@test is_full_row_rank(op)    == true
@test is_full_row_rank(DiagOp([ones(5);0]))    == false
@test is_full_column_rank(op) == true
@test is_full_column_rank(DiagOp([ones(5);0]))    == false

@test diag(op) == d
@test vecnorm(op'*(op*x1) - diag_AcA(op).*x1) <= 1e-12
@test vecnorm(op*(op'*x1) - diag_AAc(op).*x1) <= 1e-12

########## Eye ############
n = 4
op = Eye(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)

@test all(vecnorm.(y1 .- x1) .<= 1e-12)

# other constructors
op = Eye(Float64, (n,))
op = Eye((n,))
op = Eye(n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == true
@test is_diagonal(op)         == true
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == true
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

@test diag(op) == 1
@test diag_AcA(op) == 1
@test diag_AAc(op) == 1

######## Filt ############
n,m = 15,2
b,a = [1.;0.;1.;0.;0.], [1.;1.;1.]
op = Filt(Float64,(n,),b,a)
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = filt(b, a, x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

h = randn(10)
op = Filt(Float64,(n,m),h)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m), verb)
y2 = filt(h, [1.], x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
Filt(n,  b, a)
Filt((n,m),  b, a)
Filt(n,  h)
Filt((n,),  h)
Filt(x1, b, a)
Filt(x1, b)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

######## FiniteDiff ############
n= 10
op = FiniteDiff(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n-1), verb)
y1 = op*collect(linspace(0,1,n))
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

B = -spdiagm(ones(n-1),0,n-1,n)+spdiagm(ones(n-1),1,n-1,n)
@test norm(B*x1-op*x1) <= 1e-8

n,m= 10,5
op = FiniteDiff(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n-1,m), verb)
y1 = op*repmat(collect(linspace(0,1,n)),1,m)
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

n,m= 10,5
op = FiniteDiff(Float64,(n,m),2)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m-1), verb)
y1 = op*repmat(collect(linspace(0,1,n)),1,m)
@test all(vecnorm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n-1,m,l), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),2)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m-1,l), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),3)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m,l-1), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1) .<= 1e-12)

@test_throws ErrorException op = FiniteDiff(Float64,(n,m,l,3))
@test_throws ErrorException op = FiniteDiff(Float64,(n,m,l), 4)

## other constructors
FiniteDiff((n,m))
FiniteDiff(x1)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == false

########## GetIndex ############
n,m = 5,4
k = 3
op = GetIndex(Float64,(n,),(1:k,))
x1 = randn(n)
y1 = test_op(op, x1, randn(k), verb)

@test all(vecnorm.(y1 .- x1[1:k]) .<= 1e-12)

n,m = 5,4
k = 3
op = GetIndex(Float64,(n,m),(1:k,:))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(k,m), verb)

@test all(vecnorm.(y1 .- x1[1:k,:]) .<= 1e-12)

# other constructors
GetIndex((n,m), (1:k,:))
GetIndex(x1, (1:k,:))

@test_throws ErrorException op = GetIndex(Float64,(n,m),(1:k,:,:))
op = GetIndex(Float64,(n,m),(1:n,1:m))
@test typeof(op) <: Eye

op = GetIndex(Float64,(n,),(1:k,))

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == false

@test diag_AAc(op) == 1

######## MatrixOp ############

n,m = 5,4
A = randn(n,m)
op = MatrixOp(Float64,(m,),A)
x1 = randn(m)
y1 = test_op(op, x1, randn(n), verb)
y2 = A*x1

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

c = 3
op = MatrixOp(Float64,(m,c),A)
@test_throws ErrorException op = MatrixOp(Float64,(m,c,3),A)
@test_throws MethodError op = MatrixOp(Float64,(m,c),randn(n,m,2))
x1 = randn(m,c)
y1 = test_op(op, x1, randn(n,c), verb)
y2 = A*x1

# other constructors
op = MatrixOp(A)
op = MatrixOp(Float64, A)
op = MatrixOp(A, c)
op = MatrixOp(Float64, A, c)

op = convert(LinearOperator,A)
op = convert(LinearOperator,A,c)

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(MatrixOp(randn(srand(0),3,4)))    == true
@test is_full_column_rank(MatrixOp(randn(srand(0),3,4))) == false


####### MatrixOp ############

n,m = 5,6
b = randn(m)
op = MatrixMul(Float64,(n,m),b,n)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n), verb)
y2 = x1*b

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

n,m = 5,6
b = randn(m)+im*randn(m)
op = MatrixMul(Complex{Float64},(n,m),b,n)
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = x1*b

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

## other constructors
op = MatrixMul(b,n)

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == false
@test is_full_column_rank(op) == false


######### RowVectorOp ############
n = 5
A = randn(n)
op = RowVectorOp(Float64,(n,),A)
x1 = randn(n)
y1 = test_op(op, x1, [randn()], verb)

n,c = 5,10
A = randn(n)
op = RowVectorOp(Float64,(n,c),A)
x1 = randn(n,c)
y1 = test_op(op, x1, randn(1,c), verb)

# other constructors
op = RowVectorOp(A)
op = RowVectorOp(Float64, A)
op = RowVectorOp(A, c)
op = RowVectorOp(Float64, A, c)

op = convert(LinearOperator,A)
op = convert(LinearOperator,A,c)
op = convert(LinearOperator,A',c)
op = convert(LinearOperator,A')

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

######### MyLinOp ############

n,m = 5,4
A = randn(n,m)
op = MyLinOp(Float64, (m,),(n,), (y,x) -> A_mul_B!(y,A,x), (y,x) -> Ac_mul_B!(y,A,x))
x1 = randn(m)
y1 = test_op(op, x1, randn(n), verb)
y2 = A*x1

# other constructors
op = MyLinOp(Float64, (m,), Float64, (n,), (y,x) -> A_mul_B!(y,A,x), (y,x) -> Ac_mul_B!(y,A,x))


######### MIMOFilt ############
m,n = 10,2
b = [[1.;0.;1.;0.;0.],[1.;0.;1.;0.;0.]]
a = [[1.;1.;1.],[2.;2.;2.]]
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,1), verb)
y2 = filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2])

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

m,n = 10,2
b = [[1.;0.;1.;0.;0.],[1.;0.;1.;0.;0.],[1.;0.;1.;0.;0.],[1.;0.;1.;0.;0.] ]
a = [[1.;1.;1.],[2.;2.;2.],[1.],[1.]]
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,2), verb)
y2 = [filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2]) filt(b[3],a[3],x1[:,1])+filt(b[4],a[4],x1[:,2])]

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

m,n = 10,3
b = [randn(10),randn(5),randn(10),randn(2),randn(10),randn(10)]
a = [[1.],[1.],[1.],[1.],[1.],[1.]]
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,2), verb)
y2 = [filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2])+filt(b[3],a[3],x1[:,3]) filt(b[4],a[4],x1[:,1])+filt(b[5],a[5],x1[:,2])+filt(b[6],a[6],x1[:,3])]

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

## other constructors
MIMOFilt((10,3),  b, a)
MIMOFilt((10,3),  b)
MIMOFilt(x1,  b, a)
MIMOFilt(x1,  b)

#errors
@test_throws ErrorException op = MIMOFilt(Float64, (10,3,2) ,b,a)
a2 = [[1.0f0],[1.0f0],[1.0f0],[1.0f0],[1.0f0],[1.0f0]]
b2 = convert.(Array{Float32,1},b)
@test_throws ErrorException op = MIMOFilt(Float64, (m,n),b2,a2)
@test_throws ErrorException op = MIMOFilt(Float64, (m,n),b,a[1:end-1])
push!(a2,[1.0f0])
push!(b2,randn(Float32,10))
@test_throws ErrorException op = MIMOFilt(Float32, (m,n),b2,a2)
a[1][1] = 0.
@test_throws ErrorException op = MIMOFilt(Float64, (m,n) ,b,a)

b = [randn(10),randn(5),randn(10),randn(2),randn(10),randn(10)]
a = [[1.],[1.],[1.],[1.],[1.],[1.]]
op = MIMOFilt(Float64, (m,n), b, a)

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

######## Variation ############

n,m = 10,5
op = Variation(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(m*n,2), verb)

y1 = op*repmat(collect(linspace(0,1,n)),1,m)
@test all(vecnorm.(y1[:,1] .- 1/(n-1) ) .<= 1e-12)
@test all(vecnorm.(y1[:,2] ) .<= 1e-12)

Dx = spdiagm(ones(n),0,n,n)-spdiagm(ones(n-1),-1,n,n)
Dx[1,1],Dx[1,2] = -1,1 
Dy = spdiagm(ones(m),0,m,m)-spdiagm(ones(m-1),-1,m,m)
Dy[1,1],Dy[1,2] = -1,1 

Dxx = kron(eye(m),Dx)
Dyy = kron(Dy,eye(n))
TV = [Dxx;Dyy]

x1 = randn(n,m)
@test vecnorm(op*x1-reshape(TV*(x1[:]),n*m,2))<1e-12

n,m,l = 10,5,3
op = Variation(Float64,(n,m,l))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(m*n*l,3), verb)

y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1[:,1] .- 1/(n-1) ) .<= 1e-12)
@test all(vecnorm.(y1[:,2] ) .<= 1e-12)
@test all(vecnorm.(y1[:,3] ) .<= 1e-12)

### other constructors
Variation(Float64, n,m)
Variation((n,m))
Variation(n,m)
Variation(x1)

##errors
@test_throws ErrorException op = Variation(Float64,(n,))

###properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == false
@test is_full_column_rank(op) == false

######## Xcorr ############
n,m = 5, 6
h = randn(m)
op = Xcorr(Float64,(n,),h)
x1 = randn(n)
y1 = test_op(op, x1, randn(n+m), verb)
y2 = xcorr(x1, h)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)
# other constructors
op = Xcorr(x1,h)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == true

######## ZeroPad ############
n = (3,)
z = (5,)
op = ZeroPad(Float64,n,z)
x1 = randn(n)
y1 = test_op(op, x1, randn(n.+z), verb)
@test all(vecnorm.(y1 .- [x1;zeros(5)] ) .<= 1e-12)

n = (3,2)
z = (5,3)
op = ZeroPad(Float64,n,z)
x1 = randn(n)
y1 = test_op(op, x1, randn(n.+z), verb)
y2 = zeros(n.+z)
y2[1:n[1],1:n[2]] = x1
@test all(vecnorm.(y1 .- y2 ) .<= 1e-12)

n = (3,2,2)
z = (5,3,1)
op = ZeroPad(Float64,n,z)
x1 = randn(n)
y1 = test_op(op, x1, randn(n.+z), verb)
y2 = zeros(n.+z)
y2[1:n[1],1:n[2],1:n[3]] = x1
@test all(vecnorm.(y1 .- y2 ) .<= 1e-12)

# other constructors
ZeroPad(n, z...)
ZeroPad(Float64, n, z...)
ZeroPad(n, z...)
ZeroPad(x1, z)
ZeroPad(x1, z...)

#errors
@test_throws ErrorException op = ZeroPad(Float64,n,(1,2))
@test_throws ErrorException op = ZeroPad(Float64,n,(1,-2,3))
@test_throws ErrorException op = ZeroPad(Float64,(1,2,3,4),(1,2,3,4))

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == true
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == false
@test is_full_column_rank(op) == true

diag_AcA(op) == 1

########## Zeros ############
n = (3,4)
D = Float64
m = (5,2)
C = Complex{Float64}
op = Zeros(D,n,C,m)
x1 = randn(n)
y1 = test_op(op, x1, randn(m)+im*randn(m), verb)

#properties
@test is_linear(op)           == true
@test is_null(op)             == true
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == false
@test is_full_column_rank(op) == false
