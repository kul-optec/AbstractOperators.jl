@printf("\nTesting linear operators\n")

verb = true

function test_op(A::LinearOperator, x, y, verb::Bool = false)

  verb && (println(); show(A); println())

  Ax = A*x
  Ax2 = AbstractOperators.deepsimilar(Ax)
  verb && println("forward preallocated")
  A_mul_B!(Ax2, A, x) #verify in-place linear operator works
  verb && @time A_mul_B!(Ax2, A, x)

  @test AbstractOperators.deepvecnorm(Ax .- Ax2) <= 1e-8

  Acy = A'*y
  Acy2 = AbstractOperators.deepsimilar(Acy)
  verb && println("adjoint preallocated")
  Ac_mul_B!(Acy2, A, y) #verify in-place linear operator works
  verb && @time Ac_mul_B!(Acy2, A, y)

  @test AbstractOperators.deepvecnorm(Acy .- Acy2) <= 1e-8

	s1 = AbstractOperators.deepvecdot(Ax2, y)
	s2 = AbstractOperators.deepvecdot(x, Acy2)

	@test abs( s1 - s2 ) < 1e-8

  return Ax
end

######### Conv ############
n,m = 5, 6
h = randn(m)
op = Conv(Float64,(n,),h)
x1 = randn(n)
y1 = test_op(op, x1, randn(n+m-1), verb)
y2 = conv(h,x1)

@test all(vecnorm.(y1 .- y2) .<= 1e-12)
# other constructors
op = Conv(x1,h)

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

######### DFT ############
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

######### Eye ############
n = 4
op = Eye(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)

@test all(vecnorm.(y1 .- x1) .<= 1e-12)

# other constructors
op = Eye(Float64, (n,))
op = Eye((n,)) 
op = Eye(n) 

######### Filt ############
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

######### FiniteDiff ############
n= 10
op = FiniteDiff(Float64,(n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y1 = op*collect(linspace(0,1,n))
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

n,m= 10,5
op = FiniteDiff(Float64,(n,m))
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m), verb)
y1 = op*repmat(collect(linspace(0,1,n)),1,m)
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

n,m= 10,5
op = FiniteDiff(Float64,(n,m),2)
x1 = randn(n,m)
y1 = test_op(op, x1, randn(n,m), verb)
y1 = op*repmat(collect(linspace(0,1,n)),1,m)
@test all(vecnorm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l))
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m,l), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1 .- 1/9) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),2)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m,l), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1) .<= 1e-12)

n,m,l= 10,5,7
op = FiniteDiff(Float64,(n,m,l),3)
x1 = randn(n,m,l)
y1 = test_op(op, x1, randn(n,m,l), verb)
y1 = op*reshape(repmat(collect(linspace(0,1,n)),1,m*l),n,m,l)
@test all(vecnorm.(y1) .<= 1e-12)

@test_throws ErrorException op = FiniteDiff(Float64,(n,m,l,3))
@test_throws ErrorException op = FiniteDiff(Float64,(n,m,l), 4)

## other constructors
FiniteDiff((n,m)) 
FiniteDiff(x1)

# GetIndex

######### MatrixOp ############

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
x1 = randn(m,c)
y1 = test_op(op, x1, randn(n,c), verb)
y2 = A*x1

# other constructors
op = MatrixOp(A)
op = MatrixOp(Float64, A) 
op = MatrixOp(A, c)
op = MatrixOp(Float64, A, c) 

# MIMOFilt

# Variation

# Xcorr

# ZeroPad

# Zeros
