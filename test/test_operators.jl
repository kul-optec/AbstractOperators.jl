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

# Conv

# DCT

# DFT

######### DiagOp ############
n = 4
d = randn(n)
opD = DiagOp(Float64,(n,),d)
x1 = randn(n)
y1 = test_op(opD, x1, randn(n), verb)
y2 = d.*x1

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

# other constructors
opD = DiagOp(d) 
opD = DiagOp(Float64, d)

######### Eye ############
n = 4
opI = Eye(Float64,(n,))
x1 = randn(n)
y1 = test_op(opI, x1, randn(n), verb)

@test all(vecnorm.(y1 .- x1) .<= 1e-12)

# other constructors
opI = Eye(Float64, (n,))
opI = Eye((n,)) 
opI = Eye(n) 

# Filt

# FiniteDiff

# GetIndex

######### MatrixOp ############

n,m = 5,4
A = randn(n,m)
opA = MatrixOp(Float64,(m,),A)
x1 = randn(m)
y1 = test_op(opA, x1, randn(n), verb)
y2 = A*x1

@test all(vecnorm.(y1 .- y2) .<= 1e-12)

c = 3
opA = MatrixOp(Float64,(m,c),A)
@test_throws ErrorException opA = MatrixOp(Float64,(m,c,3),A)
x1 = randn(m,c)
y1 = test_op(opA, x1, randn(n,c), verb)
y2 = A*x1

# other constructors
OpA = MatrixOp(A)
OpA = MatrixOp(Float64, A) 
OpA = MatrixOp(A, c)
OpA = MatrixOp(Float64, A, c) 

# MIMOFilt

# Variation

# Xcorr

# ZeroPad

# Zeros
