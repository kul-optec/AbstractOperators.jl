 @printf("\nTesting linear operators\n")

######## Conv ############
n,m = 5, 6
h = randn(m)
op = Conv(Float64,(n,),h)
x1 = randn(n)
y1 = test_op(op, x1, randn(n+m-1), verb)
y2 = conv(x1,h)

@test all(norm.(y1 .- y2) .<= 1e-12)

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

########## DCT ############
n = 4
op = DCT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = dct(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

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

@test norm(op'*(op*x1) - x1) <= 1e-12
@test diag_AAc(op) == 1.
@test diag_AcA(op) == 1.

######### IDCT ############
n = 4
op = IDCT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = idct(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

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

@test norm(op'*(op*x1) - x1) <= 1e-12
@test diag_AAc(op) == 1.
@test diag_AcA(op) == 1.

######## DFT ############
# seems like there is an object called DFT in Base julia 0.7 (however in 1.0 was rm)
n,m = 4,7

op = AbstractOperators.DFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Complex{Float64},(n,))
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Complex{Float64},(n,))
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Float64,(n,),1)
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Complex{Float64},(n,),1)
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = fft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, fft(randn(n,m)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Complex{Float64},(n,m))
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, fft(randn(n,m)), verb)
y2 = fft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Float64,(m,n),1)
x1 = randn(m,n)
y1 = test_op(op, x1, fft(randn(m,n)), verb)
y2 = fft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.DFT(Complex{Float64},(n,m),2)
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, fft(randn(n,m)), verb)
y2 = fft(x1,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

# other constructors
op = AbstractOperators.DFT((n,))
op = AbstractOperators.DFT(n,n)
op = AbstractOperators.DFT(Complex{Float64}, n,n)

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
op = AbstractOperators.DFT(n,m)
x1 = randn(n,m)
y1 = op*x1
@test norm(op'*(op*x1) - diag_AcA(op)*x1) <= 1e-12
@test norm(op*(op'*y1) - diag_AAc(op)*y1) <= 1e-12

######### IDFT ############
n,m = 5,6

op = IDFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(n,))
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Float64,(n,),1)
x1 = randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(n,),1)
x1 = randn(n)+im*randn(n)
y1 = test_op(op, x1, fft(randn(n)), verb)
y2 = ifft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = AbstractOperators.IDFT(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, fft(randn(n,m)), verb)
y2 = ifft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(n,m))
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, fft(randn(n,m)), verb)
y2 = ifft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Float64,(m,n),1)
x1 = randn(m,n)
y1 = test_op(op, x1, fft(randn(m,n)), verb)
y2 = ifft(x1,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(m,n),2)
x1 = randn(m,n)+im*randn(m,n)
y1 = test_op(op, x1, fft(randn(m,n)), verb)
y2 = ifft(x1,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 4,19,5
op = IDFT(Complex{Float64},(n,m,l),2)
x1 = fft(randn(n,m,l),2)
y1 = test_op(op, x1, ifft(x1,2), verb)
y2 = ifft(x1,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 4,18,5
op = IDFT(Complex{Float64},(n,m,l),(1,2))
x1 = fft(randn(n,m,l),(1,2))
y1 = test_op(op, x1, ifft(x1,(1,2)), verb)
y2 = ifft(x1,(1,2))

@test all(norm.(y1 .- y2) .<= 1e-12)

op = IDFT(Complex{Float64},(n,m,l),(3,2))
x1 = fft(randn(n,m,l),(3,2))
y1 = test_op(op, x1, ifft(x1,(3,2)), verb)
y2 = ifft(x1,(3,2))

@test all(norm.(y1 .- y2) .<= 1e-12)

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
@test norm(op'*(op*x1) - diag_AcA(op)*x1) <= 1e-12
@test norm(op*(op'*y1) - diag_AAc(op)*y1) <= 1e-12

####### RDFT ############
n = 4
op = RDFT(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, rfft(x1), verb)
y2 = rfft(x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 4,8,5
op = RDFT(Float64,(n,m,l),2)
x1 = randn(n,m,l)
y1 = test_op(op, x1, rfft(x1,2), verb)
y2 = rfft(x1,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

# other constructors
op = RDFT((n,))
op = RDFT(n,n)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == false

####### IRDFT ############
n = 12
op = IRDFT(Complex{Float64},(div(n,2)+1,),n)
x1 = rfft(randn(n))
y1 = test_op(op, x1,irfft(randn(div(n,2)+1),n), verb)
y2 = irfft(x1,n)

@test all(norm.(y1 .- y2) .<= 1e-12)

n = 11
op = IRDFT(Complex{Float64},(div(n,2)+1,),n)
x1 = rfft(randn(n))
y1 = test_op(op, x1,irfft(randn(div(n,2)+1),n), verb)
y2 = irfft(x1,n)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 4,19,5
op = IRDFT(Complex{Float64},(n,div(m,2)+1,l),m,2)
x1 = rfft(randn(n,m,l),2)
y1 = test_op(op, x1, irfft(x1,m,2), verb)
y2 = irfft(x1,m,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 4,18,5
op = IRDFT(Complex{Float64},(n,div(m,2)+1,l),m,2)
x1 = rfft(randn(n,m,l),2)
y1 = test_op(op, x1, irfft(x1,m,2), verb)
y2 = irfft(x1,m,2)

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 5,18,5
op = IRDFT(Complex{Float64},(div(n,2)+1,m,l),n,1)
x1 = rfft(randn(n,m,l),1)
y1 = test_op(op, x1, irfft(x1,n,1), verb)
y2 = irfft(x1,n,1)

@test all(norm.(y1 .- y2) .<= 1e-12)

## other constructors
op = IRDFT((10,),19)

#properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == true
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == false

########## DiagOp ############
n = 4
d = randn(n)
op = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = d.*x1

@test all(norm.(y1 .- y2) .<= 1e-12)

n = 4
d = randn(n)+im*randn(n)
op = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(op, x1, randn(n).+im*randn(n), verb)
y2 = d.*x1

@test all(norm.(y1 .- y2) .<= 1e-12)

n = 4
d = pi
op = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = d.*x1

@test all(norm.(y1 .- y2) .<= 1e-12)

n = 4
d = im
op = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = d.*x1

@test all(norm.(y1 .- y2) .<= 1e-12)

# other constructors
d = randn(4)
op = DiagOp(d)

d = randn(4).+im
op = DiagOp(d)

n = 4
d = pi
op = DiagOp((n,), d)

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
@test norm(op'*(op*x1) .- diag_AcA(op).*x1) <= 1e-12
@test norm(op*(op'*x1) .- diag_AAc(op).*x1) <= 1e-12

n = 4
d = pi
op = DiagOp((n,), d)
x1 = randn(n)

@test diag(op) == d
@test norm(op'*(op*x1) .- diag_AcA(op).*x1) <= 1e-12
@test norm(op*(op'*x1) .- diag_AAc(op).*x1) <= 1e-12

########## Eye ############
n = 4
op = Eye(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)

@test all(norm.(y1 .- x1) .<= 1e-12)

# other constructors
op = Eye(Float64, (n,))
op = Eye((n,))
op = Eye(n)
op = Eye(x1)

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

@test all(norm.(y1 .- y2) .<= 1e-12)

h = randn(10)
op = Filt(Float64,(n,m),h)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m), verb)
y2 = filt(h, [1.], x1)

@test all(norm.(y1 .- y2) .<= 1e-12)

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

###### FiniteDiff ############
n= 10
op = FiniteDiff(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n-1), verb)
y1 = op*collect(range(0,stop=1,length=n))
@test all(norm.(y1 .- 1/9) .<= 1e-12)

I1, J1, V1 = SparseArrays.spdiagm_internal(0 => ones(n-1))
I2, J2, V2 = SparseArrays.spdiagm_internal(1 => ones(n-1))
B = -sparse(I1,J1,V1,n-1,n)+sparse(I2,J2,V2,n-1,n)

@test norm(B*x1-op*x1) <= 1e-8

n,m= 10,5
op = FiniteDiff(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n-1,m), verb)
y1 = op*repeat(collect(range(0,stop=1,length=n)),1,m)
@test all(norm.(y1 .- 1/9) .<= 1e-12)

n,m= 10,5
op = FiniteDiff(Float64,(n,m),2)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m-1), verb)
y1 = op*repeat(collect(range(0,stop=1,length=n)),1,m)
@test all(norm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n-1,m,l), verb)
y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l),n,m,l)
@test all(norm.(y1 .- 1/9) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),2)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m-1,l), verb)
y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l),n,m,l)
@test all(norm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),3)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m,l-1), verb)
y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l),n,m,l)
@test all(norm.(y1) .<= 1e-12)

n,m,l,i = 5,6,2,3
op = FiniteDiff(Float64,(n,m,l,i),1)
x1 = randn(n,m,l,i)
y1 = test_op(op, x1, randn(n-1,m,l,i), verb)
y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l*i),n,m,l,i)
@test all(norm.(y1 .- 1/(n-1)) .<= 1e-12)

n,m,l,i = 5,6,2,3
op = FiniteDiff(Float64,(n,m,l,i),4)
x1 = randn(n,m,l,i)
y1 = test_op(op, x1, randn(n,m,l,i-1), verb)
y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l*i),n,m,l,i)
@test norm(y1) <= 1e-12

@test_throws ErrorException FiniteDiff(Float64,(n,m,l), 4)

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

@test all(norm.(y1 .- x1[1:k]) .<= 1e-12)

n,m = 5,4
k = 3
op = GetIndex(Float64,(n,m),(1:k,:))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(k,m), verb)

@test all(norm.(y1 .- x1[1:k,:]) .<= 1e-12)

n,m = 5,4
op = GetIndex(Float64,(n,m),(:,2))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n), verb)

@test all(norm.(y1 .- x1[:,2]) .<= 1e-12)

n,m,l = 5,4,3
op = GetIndex(Float64,(n,m,l),(1:3,2,:))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(3,3), verb)

@test all(norm.(y1 .- x1[1:3,2,:]) .<= 1e-12)

# other constructors
GetIndex((n,m), (1:k,:))
GetIndex(x1, (1:k,:))

@test_throws ErrorException GetIndex(Float64,(n,m),(1:k,:,:))
op = GetIndex(Float64,(n,m),(1:n,1:m))
@test typeof(op) <: Eye

op = GetIndex(Float64,(n,),(1:k,))

##properties
@test is_sliced(op)           == true
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

####### MatrixOp ############

# real matrix, real input
n,m = 5,4
A = randn(n,m)
op = MatrixOp(Float64,(m,),A)
x1 = randn(m)
y1 = test_op(op, x1, randn(n), verb)
y2 = A*x1

# real matrix, complex input
n,m = 5,4
A = randn(n,m)
op = MatrixOp(Complex{Float64},(m,),A)
x1 = randn(m)+im.*randn(m)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = A*x1

# complex matrix, complex input
n,m = 5,4
A = randn(n,m)+im*randn(n,m)
op = MatrixOp(Complex{Float64},(m,),A)
x1 = randn(m)+im.*randn(m)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = A*x1

# complex matrix, real input
n,m = 5,4
A = randn(n,m)+im*randn(n,m)
op = MatrixOp(Float64,(m,),A)
x1 = randn(m)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = A*x1

@test all(norm.(y1 .- y2) .<= 1e-12)

# complex matrix, real matrix input
c = 3
op = MatrixOp(Float64,(m,c),A)
@test_throws ErrorException MatrixOp(Float64,(m,c,3),A)
@test_throws MethodError MatrixOp(Float64,(m,c),randn(n,m,2))
x1 = randn(m,c)
y1 = test_op(op, x1, randn(n,c).+randn(n,c), verb)
y2 = A*x1

# other constructors
op = MatrixOp(A)
op = MatrixOp(Float64, A)
op = MatrixOp(A, c)
op = MatrixOp(Float64, A, c)

op = convert(LinearOperator,A)
op = convert(LinearOperator,A,c)
op = convert(LinearOperator, Complex{Float64}, size(x1), A)

##properties
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(MatrixOp(randn(Random.seed!(0),3,4)))    == true
@test is_full_column_rank(MatrixOp(randn(Random.seed!(0),3,4))) == false


##### LMatrixOp ############

n,m = 5,6
b = randn(m)
op = LMatrixOp(Float64,(n,m),b)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n), verb)
y2 = x1*b

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m = 5,6
b = randn(m)+im*randn(m)
op = LMatrixOp(Complex{Float64},(n,m),b)
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, randn(n)+im*randn(n), verb)
y2 = x1*b

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 5,6,7
b = randn(m,l)
op = LMatrixOp(Float64,(n,m),b)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,l), verb)
y2 = x1*b

@test all(norm.(y1 .- y2) .<= 1e-12)

n,m,l = 5,6,7
b = randn(m,l)+im*randn(m,l)
op = LMatrixOp(Complex{Float64},(n,m),b)
x1 = randn(n,m)+im*randn(n,m)
y1 = test_op(op, x1, randn(n,l)+im*randn(n,l), verb)
y2 = x1*b

@test all(norm.(y1 .- y2) .<= 1e-12)

## other constructors
op = LMatrixOp(b,n)

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

######### MyLinOp ############

n,m = 5,4;
A = randn(n,m);
op = MyLinOp(Float64, (m,),(n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))
x1 = randn(m)
y1 = test_op(op, x1, randn(n), verb)
y2 = A*x1

# other constructors
op = MyLinOp(Float64, (m,), Float64, (n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))


######### MIMOFilt ############
m,n = 10,2
b = [[1.;0.;1.;0.;0.],[1.;0.;1.;0.;0.]]
a = [[1.;1.;1.],[2.;2.;2.]]
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,1), verb)
y2 = filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2])

@test all(norm.(y1 .- y2) .<= 1e-12)

m,n = 10,3; #time samples, number of inputs
b = [[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.], ];
a = [[1.;1.;1.],[2.;2.;2.],[      3.],[      4.],[      5.],[      6.], ];
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,2), verb)
y2 = [filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2])+filt(b[3],a[3],x1[:,3]) filt(b[4],a[4],x1[:,1])+filt(b[5],a[5],x1[:,2])+filt(b[6],a[6],x1[:,3])]

@test all(norm.(y1 .- y2) .<= 1e-12)

m,n = 10,3
b = [randn(10),randn(5),randn(10),randn(2),randn(10),randn(10)]
a = [[1.],[1.],[1.],[1.],[1.],[1.]]
op = MIMOFilt(Float64, (m,n), b, a)

x1 = randn(m,n)
y1 = test_op(op, x1, randn(m,2), verb)
y2 = [filt(b[1],a[1],x1[:,1])+filt(b[2],a[2],x1[:,2])+filt(b[3],a[3],x1[:,3]) filt(b[4],a[4],x1[:,1])+filt(b[5],a[5],x1[:,2])+filt(b[6],a[6],x1[:,3])]

@test all(norm.(y1 .- y2) .<= 1e-12)

## other constructors
MIMOFilt((10,3),  b, a)
MIMOFilt((10,3),  b)
MIMOFilt(x1,  b, a)
MIMOFilt(x1,  b)

#errors
@test_throws ErrorException MIMOFilt(Float64, (10,3,2) ,b,a)
a2 = [[1.0f0],[1.0f0],[1.0f0],[1.0f0],[1.0f0],[1.0f0]]
b2 = convert.(Array{Float32,1},b)
@test_throws ErrorException MIMOFilt(Float64, (m,n),b2,a2)
@test_throws ErrorException MIMOFilt(Float64, (m,n),b,a[1:end-1])
push!(a2,[1.0f0])
push!(b2,randn(Float32,10))
@test_throws ErrorException MIMOFilt(Float32, (m,n),b2,a2)
a[1][1] = 0.
@test_throws ErrorException MIMOFilt(Float64, (m,n) ,b,a)

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

y1 = op*repeat(collect(range(0,stop=1,length=n)),1,m)
@test all(norm.(y1[:,1] .- 1/(n-1) ) .<= 1e-12)
@test all(norm.(y1[:,2] ) .<= 1e-12)

Dx = spdiagm(0 => ones(n), -1 => -ones(n-1))
Dx[1,1],Dx[1,2] = -1,1
Dy = spdiagm(0 => ones(m), -1 => -ones(m-1))
Dy[1,1],Dy[1,2] = -1,1

Dxx = kron(sparse(I,m,m),Dx)
Dyy = kron(Dy,sparse(I,n,n))
TV = [Dxx;Dyy]

x1 = randn(n,m)
@test norm(op*x1-reshape(TV*(x1[:]),n*m,2))<1e-12

n,m,l = 10,5,3
op = Variation(Float64,(n,m,l))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(m*n*l,3), verb)

y1 = op*reshape(repeat(collect(range(0,stop=1,length=n)),1,m*l),n,m,l)
@test all(norm.(y1[:,1] .- 1/(n-1) ) .<= 1e-12)
@test all(norm.(y1[:,2] ) .<= 1e-12)
@test all(norm.(y1[:,3] ) .<= 1e-12)

### other constructors
Variation(Float64, n,m)
Variation((n,m))
Variation(n,m)
Variation(x1)

##errors
@test_throws ErrorException Variation(Float64,(n,))

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
y2 = xcorr(x1, h; padmode=:longest)

@test all(norm.(y1 .- y2) .<= 1e-12)
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
@test all(norm.(y1 .- [x1;zeros(5)] ) .<= 1e-12)

n = (3,2)
z = (5,3)
op = ZeroPad(Float64,n,z)
x1 = randn(n)
y1 = test_op(op, x1, randn(n.+z), verb)
y2 = zeros(n.+z)
y2[1:n[1],1:n[2]] = x1
@test all(norm.(y1 .- y2 ) .<= 1e-12)

n = (3,2,2)
z = (5,3,1)
op = ZeroPad(Float64,n,z)
x1 = randn(n)
y1 = test_op(op, x1, randn(n.+z), verb)
y2 = zeros(n.+z)
y2[1:n[1],1:n[2],1:n[3]] = x1
@test all(norm.(y1 .- y2 ) .<= 1e-12)

# other constructors
ZeroPad(n, z...)
ZeroPad(Float64, n, z...)
ZeroPad(n, z...)
ZeroPad(x1, z)
ZeroPad(x1, z...)

#errors
@test_throws ErrorException ZeroPad(Float64,n,(1,2))
@test_throws ErrorException ZeroPad(Float64,n,(1,-2,3))

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
