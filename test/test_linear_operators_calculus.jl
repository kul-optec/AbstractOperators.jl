@printf("\nTesting linear operators calculus rules\n")

##########################
##### test Compose #######
##########################
m1, m2, m3 = 4, 7, 3
A1 = randn(m2, m1)
A2 = randn(m3, m2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)

opC = Compose(opA2,opA1)
x = randn(m1)
y1 = test_op(opC, x, randn(m3), verb)
y2 = A2*A1*x
@test all(norm.(y1 .- y2) .<= 1e-12)

# test Compose longer
m1, m2, m3, m4 = 4, 7, 3, 2
A1 = randn(m2, m1)
A2 = randn(m3, m2)
A3 = randn(m4, m3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)

opC1 = Compose(opA3,Compose(opA2,opA1))
opC2 = Compose(Compose(opA3,opA2),opA1)
x = randn(m1)
y1 = test_op(opC1, x, randn(m4), verb)
y2 = test_op(opC2, x, randn(m4), verb)
y3 = A3*A2*A1*x
@test all(norm.(y1 .- y2) .<= 1e-12)
@test all(norm.(y3 .- y2) .<= 1e-12)

#test Compose special cases
@test typeof(opA1*Eye(m1)) == typeof(opA1) 
@test typeof(Eye(m2)*opA1) == typeof(opA1) 
@test typeof(Eye(m2)*Eye(m2)) == typeof(Eye(m2)) 

opS1 = Compose(opA2,opA1)
opS1c = Scale(pi,opS1)
@test typeof(opS1c.A[end]) <: Scale

#properties
@test is_sliced(opC)            == false
@test is_linear(opC1)           == true
@test is_null(opC1)             == false
@test is_eye(opC1)              == false
@test is_diagonal(opC1)         == false
@test is_AcA_diagonal(opC1)     == false
@test is_AAc_diagonal(opC1)     == false
@test is_orthogonal(opC1)       == false
@test is_invertible(opC1)       == false
@test is_full_row_rank(opC1)    == false
@test is_full_column_rank(opC1) == false

# properties special case
opC = DCT((5,))*GetIndex((10,), 1:5)
@test is_sliced(opC)           == true
@test is_AAc_diagonal(opC)     == true
@test diag_AAc(opC)            == 1.0

d = randn(5)
opC = DiagOp(d)*GetIndex((10,), 1:5)
@test is_sliced(opC)           == true
@test is_diagonal(opC)         == true
@test diag(opC)                == d

# displacement test
m1, m2, m3, m4, m5 = 4, 7, 3, 2, 11
A1 = randn(m2, m1)
A2 = randn(m3, m2)
A3 = randn(m4, m3)
A4 = randn(m5, m4)
d1 = randn(m2)
d2 = pi
d3 = 0.0
d4 = randn(m5)
opA1 = AffineAdd(MatrixOp(A1),d1)  
opA2 = AffineAdd(MatrixOp(A2),d2)
opA3 = MatrixOp(A3)
opA4 = AffineAdd(MatrixOp(A4),d4,false) 

opC  = Compose(Compose(Compose(opA4,opA3),opA2),opA1)

x = randn(m1)

@test norm( opC*x - (A4*(A3*( A2*( A1*x+d1 ) .+ d2 ) .+ d3)-d4) )  < 1e-9
@test norm( displacement(opC) - ( A4*(A3*(A2*d1 .+d2 ) .+ d3)-d4) ) < 1e-9

opA4 = MatrixOp(A4)
opC  = AffineAdd(Compose(Compose(Compose(opA4,opA3),opA2),opA1),d4)
@test norm( opC*x - (A4*(A3*( A2*( A1*x+d1 ) .+ d2 ) .+ d3)+d4) )  < 1e-9
@test norm( displacement(opC) - ( A4*(A3*(A2*d1 .+ d2) .+ d3)+d4) ) < 1e-9

@test norm( remove_displacement(opC)*x - (A4*(A3*( A2*( A1*x ) ) ) ) )  < 1e-9

#########################
#### test DCAT    #######
#########################

m1, n1, m2, n2 = 4, 7, 5, 2
A1 = randn(m1, n1)
A2 = randn(m2, n2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opD = DCAT(opA1, opA2)
x1 = randn(n1)
x2 = randn(n2)
y1 = test_op(opD, (x1, x2), (randn(m1),randn(m2)), verb)
y2 = (A1*x1, A2*x2)
@test all(norm.(y1 .- y2) .<= 1e-12)

# test DCAT longer

m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
A1 = randn(m1, n1)
A2 = randn(m2, n2)
A3 = randn(m3, n3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opD = DCAT(opA1, opA2, opA3)
x1 = randn(n1)
x2 = randn(n2)
x3 = randn(n3)
y1 = test_op(opD, (x1, x2, x3), (randn(m1),randn(m2),randn(m3)), verb)
y2 = (A1*x1, A2*x2, A3*x3)
@test all(norm.(y1 .- y2) .<= 1e-12)

#properties
@test is_linear(opD)           == true
@test is_null(opD)             == false
@test is_eye(opD)              == false
@test is_diagonal(opD)         == false
@test is_AcA_diagonal(opD)     == false
@test is_AAc_diagonal(opD)     == false
@test is_orthogonal(opD)       == false
@test is_invertible(opD)       == false
@test is_full_row_rank(opD)    == false
@test is_full_column_rank(opD) == false

# DCAT of Eye

n1, n2 = 4, 7 
x1 = randn(n1)
x2 = randn(n2)

opD = Eye((x1,x2))
y1 = test_op(opD, (x1, x2), (randn(n1),randn(n2)), verb)

#properties
@test is_linear(opD)           == true
@test is_null(opD)             == false
@test is_eye(opD)              == true 
@test is_diagonal(opD)         == true
@test is_AcA_diagonal(opD)     == true
@test is_AAc_diagonal(opD)     == true
@test is_orthogonal(opD)       == true
@test is_invertible(opD)       == true
@test is_full_row_rank(opD)    == true
@test is_full_column_rank(opD) == true

@test diag(opD) == 1
@test diag_AcA(opD) == 1
@test diag_AAc(opD) == 1

# displacement DCAT

m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
A1 = randn(m1, n1)
A2 = randn(m2, n2)
A3 = randn(m3, n3)
d1 = randn(m1)
d2 = randn(m2)
d3 = randn(m3)
opA1 = AffineAdd(MatrixOp(A1), d1)
opA2 = AffineAdd(MatrixOp(A2), d2)
opA3 = AffineAdd(MatrixOp(A3), d3)
opD = DCAT(opA1, opA2, opA3)
x1 = randn(n1)
x2 = randn(n2)
x3 = randn(n3)
y1 = opD*(x1,x2,x3)
y2 = (A1*x1+d1, A2*x2+d2, A3*x3+d3)
@test all(norm.(y1 .- y2) .<= 1e-12)
@test all(norm.(displacement(opD) .- (d1,d2,d3)) .<= 1e-12)
y1 = remove_displacement(opD)*(x1,x2,x3)
y2 = (A1*x1, A2*x2, A3*x3)
@test all(norm.(y1 .- y2) .<= 1e-12)

#######################
## test HCAT    #######
#######################

m, n1, n2 = 4, 7, 5
A1 = randn(m, n1)
A2 = randn(m, n2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opH = HCAT(opA1, opA2)
x1 = randn(n1)
x2 = randn(n2)
y1 = test_op(opH, (x1, x2), randn(m), verb)
y2 = A1*x1 + A2*x2
@test norm(y1-y2) <= 1e-12

#permuatation 
p = [2;1]
opHp = opH[p]
y1 = test_op(opHp, (x2, x1), randn(m), verb)
@test norm(y1-y2) <= 1e-12

# test HCAT longer

m, n1, n2, n3 = 4, 7, 5, 6
A1 = randn(m, n1)
A2 = randn(m, n2)
A3 = randn(m, n3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opH = HCAT(opA1, opA2, opA3)
x1 = randn(n1)
x2 = randn(n2)
x3 = randn(n3)
y1 = test_op(opH, (x1, x2, x3), randn(m), verb)
y2 = A1*x1 + A2*x2 + A3*x3
@test norm(y1-y2) <= 1e-12

# test HCAT of HCAT
opHH = HCAT(opH, opA2, opA3)
y1 = test_op(opHH, (x1, x2, x3, x2, x3), randn(m), verb)
y2 = A1*x1 + A2*x2 + A3*x3 + A2*x2 + A3*x3
@test norm(y1-y2) <= 1e-12

opHH = HCAT(opH, opH, opA3)
x = (x1, x2, x3, x1, x2, x3, x3)
y1 = test_op(opHH, x, randn(m), verb)
y2 = A1*x1 + A2*x2 + A3*x3 + A1*x1 + A2*x2 + A3*x3 + A3*x3
@test norm(y1-y2) <= 1e-12

opA3 = MatrixOp(randn(n1,n1))
@test_throws Exception HCAT(opA1,opA2,opA3)
opF = AbstractOperators.DFT(Complex{Float64},(m,))
@test_throws Exception HCAT(opA1,opF,opA2)

# test utilities

# permutation
p = randperm(ndoms(opHH,2))
opHP = AbstractOperators.permute(opHH,p)

xp = x[p] 

y1 = test_op(opHP, xp, randn(m), verb)

pp = randperm(ndoms(opHH,2))
opHPP = AbstractOperators.permute(opHH,pp)
xpp = x[pp] 
y1 = test_op(opHPP, xpp, randn(m), verb)

#properties
m, n1, n2, n3 = 4, 7, 5, 6
A1 = randn(m, n1)
A2 = randn(m, n2)
A3 = randn(m, n3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
op = HCAT(opA1, opA2, opA3)
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

d = randn(n1).+im.*randn(n1)
op = HCAT(DiagOp(d), AbstractOperators.DFT(Complex{Float64},n1))
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == true
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == true
@test is_full_column_rank(op) == false

@test diag_AAc(op) == d .* conj(d) .+ n1

y1 = randn(n1) .+ im .* randn(n1)
@test norm(op*(op'*y1).-diag_AAc(op).*y1) <1e-12

#test displacement

m, n1, n2 = 4, 7, 5
A1 = randn(m, n1)
A2 = randn(m, n2)
d1 = randn(m)
d2 = randn(m)
opA1 = AffineAdd(MatrixOp(A1), d1)
opA2 = AffineAdd(MatrixOp(A2), d2)
opH = HCAT(opA1, opA2)
x1 = randn(n1)
x2 = randn(n2)
y1 = opH*(x1,x2)
y2 = A1*x1+d1 + A2*x2+d2
@test norm(y1-y2) <= 1e-12
y1 = remove_displacement(opH)*(x1,x2)
y2 = A1*x1 + A2*x2
@test norm(y1-y2) <= 1e-12

########################
### test Reshape #######
########################

m, n = 8, 4
dim_out = (2, 2, 2)
A1 = randn(m, n)
opA1 = MatrixOp(A1)
opR = Reshape(opA1, dim_out)
opR = Reshape(opA1, dim_out...)
x1 = randn(n)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = reshape(A1*x1, dim_out)
@test norm(y1-y2) <= 1e-12

@test_throws Exception Reshape(opA1,(2,2,1))

@test is_null(opR)             == is_null(opA1)            
@test is_eye(opR)              == is_eye(opA1)             
@test is_diagonal(opR)         == is_diagonal(opA1)        
@test is_AcA_diagonal(opR)     == is_AcA_diagonal(opA1)    
@test is_AAc_diagonal(opR)     == is_AAc_diagonal(opA1)    
@test is_orthogonal(opR)       == is_orthogonal(opA1)      
@test is_invertible(opR)       == is_invertible(opA1)      
@test is_full_row_rank(opR)    == is_full_row_rank(opA1)   
@test is_full_column_rank(opR) == is_full_column_rank(opA1)

# testing displacement
m, n = 8, 4
dim_out = (2, 2, 2)
A1 = randn(m, n)
d1 = randn(m)
opA1 = AffineAdd(MatrixOp(A1),d1)
opR = Reshape(opA1, dim_out)
x1 = randn(n)
y1 = opR*x1
y2 = reshape(A1*x1+d1, dim_out)
@test norm(y1-y2) <= 1e-12
y1 = remove_displacement(opR)*x1
y2 = reshape(A1*x1, dim_out)
@test norm(y1-y2) <= 1e-12

#######################
## test Scale   #######
#######################

m, n = 8, 4
coeff = pi
A1 = randn(m, n)
opA1 = MatrixOp(A1)
opS = Scale(coeff, opA1)
x1 = randn(n)
y1 = test_op(opS, x1, randn(m), verb)
y2 = coeff*A1*x1
@test norm(y1-y2) <= 1e-12

coeff2 = 3
opS2 = Scale(coeff2, opS)
y1 = test_op(opS2, x1, randn(m), verb)
y2 = coeff2*coeff*A1*x1
@test norm(y1-y2) <= 1e-12

opF = AbstractOperators.DFT(m,n)
opS = Scale(coeff, opF)
x1 = randn(m,n)
y1 = test_op(opS, x1, fft(randn(m,n)), verb)
y2 = coeff*(fft(x1))
@test norm(y1-y2) <= 1e-12

opS = Scale(coeff, opA1)
@test is_null(opS)             == is_null(opA1)            
@test is_eye(opS)              == is_eye(opA1)             
@test is_diagonal(opS)         == is_diagonal(opA1)        
@test is_AcA_diagonal(opS)     == is_AcA_diagonal(opA1)    
@test is_AAc_diagonal(opS)     == is_AAc_diagonal(opA1)    
@test is_orthogonal(opS)       == is_orthogonal(opA1)      
@test is_invertible(opS)       == is_invertible(opA1)      
@test is_full_row_rank(opS)    == is_full_row_rank(opA1)   
@test is_full_column_rank(opS) == is_full_column_rank(opA1)

op = Scale(-4.0,AbstractOperators.DFT(10))
@test is_AAc_diagonal(op)     == true
@test diag_AAc(op) == 16*10

op = Scale(-4.0,ZeroPad((10,), 20))
@test is_AcA_diagonal(op)     == true
@test diag_AcA(op) == 16

# special case, Scale of DiagOp gets a DiagOp
d = randn(10)
op = Scale(3,DiagOp(d))
@test typeof(op) <: DiagOp
@test norm(diag(op) - 3 .*d) < 1e-12

# Scale with imaginary coeff gives error
m, n = 8, 4
coeff = im
A1 = randn(m, n)
opA1 = MatrixOp(A1)
@test_throws ErrorException Scale(coeff, opA1)

## testing displacement
m, n = 8, 4
coeff = pi
A1 = randn(m, n)
d1 = randn(m)
opA1 = AffineAdd(MatrixOp(A1),d1)
opS = Scale(coeff, opA1)
x1 = randn(n)
y1 = opS*x1
y2 = coeff*(A1*x1+d1)
@test norm(y1-y2) <= 1e-12
y1 = remove_displacement(opS)*x1
y2 = coeff*(A1*x1)
@test norm(y1-y2) <= 1e-12

#########################
#### test Sum     #######
#########################

m,n = 5,7
A1 = randn(m,n)
A2 = randn(m,n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opS = Sum(opA1,opA2)
x1 = randn(n)
y1 = test_op(opS, x1, randn(m), verb)
y2 = A1*x1+A2*x1
@test norm(y1-y2) <= 1e-12

#test Sum longer
m,n = 5,7
A1 = randn(m,n)
A2 = randn(m,n)
A3 = randn(m,n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opS = Sum(opA1,opA2,opA3)
x1 = randn(n)
y1 = test_op(opS, x1, randn(m), verb)
y2 = A1*x1+A2*x1+A3*x1
@test norm(y1-y2) <= 1e-12

opA3 = MatrixOp(randn(m,m))
@test_throws Exception Sum(opA1,opA3)
opF = AbstractOperators.DFT(Float64,(m,))
@test_throws Exception Sum(opF,opA3)

@test is_null(opS)             == false
@test is_eye(opS)              == false 
@test is_diagonal(opS)         == false
@test is_AcA_diagonal(opS)     == false
@test is_AAc_diagonal(opS)     == false
@test is_orthogonal(opS)       == false
@test is_invertible(opS)       == false
@test is_full_row_rank(opS)    == true
@test is_full_column_rank(opS) == false

d = randn(10)
op = Sum(Scale(-3.1,Eye(10)),DiagOp(d))
@test is_diagonal(op)         == true
@test norm(   diag(op) - (d .-3.1)  )<1e-12

#test displacement of sum
m,n = 5,7
A1 = randn(m,n)
A2 = randn(m,n)
A3 = randn(m,n)
d1 = randn(m)
d2 = pi
d3 = randn(m)
opA1 = AffineAdd(MatrixOp(A1), d1) 
opA2 = AffineAdd(MatrixOp(A2), d2)
opA3 = AffineAdd(MatrixOp(A3), d3) 
opS = Sum(opA1,opA2,opA3)
x1 = randn(n)
y2 = A1*x1+A2*x1+A3*x1+d1.+d2+d3
@test norm(opS*x1-y2) <= 1e-12
@test norm(displacement(opS) - (d1.+d2+d3)) <= 1e-12
y2 = A1*x1+A2*x1+A3*x1
@test norm(remove_displacement(opS)*x1-y2) <= 1e-12

###################################
###### test AdjointOperator #######
###################################

m,n = 5,7
A1 = randn(m,n)
opA1 = MatrixOp(A1)
opA1t = MatrixOp(A1')
opT = AdjointOperator(opA1)
x1 = randn(m)
y1 = test_op(opT, x1, randn(n), verb)
y2 = A1'*x1
@test norm(y1-y2) <= 1e-12

@test is_null(opT)             == is_null(opA1t)            
@test is_eye(opT)              == is_eye(opA1t)             
@test is_diagonal(opT)         == is_diagonal(opA1t)        
@test is_AcA_diagonal(opT)     == is_AcA_diagonal(opA1t)    
@test is_AAc_diagonal(opT)     == is_AAc_diagonal(opA1t)    
@test is_orthogonal(opT)       == is_orthogonal(opA1t)      
@test is_invertible(opT)       == is_invertible(opA1t)      
@test is_full_row_rank(opT)    == is_full_row_rank(opA1t)   
@test is_full_column_rank(opT) == is_full_column_rank(opA1t)

d = randn(3)
op = AdjointOperator(DiagOp(d))
@test is_diagonal(op) == true
@test diag(op) == d

op = AdjointOperator(ZeroPad((10,),5))
@test is_AcA_diagonal(op) == false
@test is_AAc_diagonal(op) == true
@test diag_AAc(op) == 1

#############################
######## test VCAT    #######
#############################

m1, m2, n = 4, 7, 5
A1 = randn(m1, n)
A2 = randn(m2, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opV = VCAT(opA1, opA2)
x1 = randn(n)
y1 = test_op(opV, x1, (randn(m1), randn(m2)), verb)
y2 = (A1*x1, A2*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

m1, n = 4, 5
A1 = randn(m1, n)+im*randn(m1, n)
opA1 = MatrixOp(A1)
opA2 = AbstractOperators.DFT(n)'
opV = VCAT(opA1, opA2)
x1 = randn(n)+im*randn(n)
y1 = test_op(opV, x1, (randn(m1)+im*randn(m1), randn(n)), verb)
y2 = (A1*x1, opA2*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

#test VCAT longer
m1, m2, m3, n = 4, 7, 3, 5
A1 = randn(m1, n)
A2 = randn(m2, n)
A3 = randn(m3, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opV = VCAT(opA1, opA2, opA3)
x1 = randn(n)
y1 = test_op(opV, x1, (randn(m1), randn(m2), randn(m3)), verb)
y2 = (A1*x1, A2*x1, A3*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

#test VCAT of VCAT
opVV = VCAT(opV,opA3)
y1 = test_op(opVV, x1, (randn(m1), randn(m2), randn(m3), randn(m3)), verb)
y2 = (A1*x1, A2*x1, A3*x1, A3*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

opVV = VCAT(opA1,opV,opA3)
y1 = test_op(opVV, x1, (randn(m1), randn(m1), randn(m2), randn(m3), randn(m3)), verb)
y2 = (A1*x1, A1*x1, A2*x1, A3*x1, A3*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

opA3 = MatrixOp(randn(m1,m1))
@test_throws Exception VCAT(opA1,opA2,opA3)
opF = AbstractOperators.DFT(Complex{Float64},(n,))
@test_throws Exception VCAT(opA1,opF,opA2)

###properties
m1, m2, m3, n = 4, 7, 3, 5
A1 = randn(m1, n)
A2 = randn(m2, n)
A3 = randn(m3, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
op = VCAT(opA1, opA2, opA3)
@test is_linear(op)           == true
@test is_null(op)             == false
@test is_eye(op)              == false
@test is_diagonal(op)         == false
@test is_AcA_diagonal(op)     == false
@test is_AAc_diagonal(op)     == false
@test is_orthogonal(op)       == false
@test is_invertible(op)       == false
@test is_full_row_rank(op)    == false
@test is_full_column_rank(op) == true

op = VCAT(AbstractOperators.DFT(Complex{Float64},10), Eye(Complex{Float64},10) )
@test is_AcA_diagonal(op)     == true
@test diag_AcA(op) == 11

##test displacement
m1, m2, n = 4, 7, 5
A1 = randn(m1, n)
A2 = randn(m2, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
d1 = randn(m1)
d2 = randn(m2)
opV = VCAT(AffineAdd(opA1,d1), AffineAdd(opA2,d2))
x1 = randn(n)
y1 = opV*x1
y2 = (A1*x1+d1, A2*x1+d2)
@test all(norm.(y1 .- y2) .<= 1e-12)
y1 = remove_displacement(opV)*x1
y2 = (A1*x1, A2*x1)
@test all(norm.(y1 .- y2) .<= 1e-12)

#########################
#### test BroadCast #####
#########################

m, n = 8, 4
dim_out = (m, 10)
A1 = randn(m, n)
opA1 = MatrixOp(A1)
opR = BroadCast(opA1, dim_out)
x1 = randn(n)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= A1*x1
@test norm(y1-y2) <= 1e-12

m, n, l, k = 8, 4, 5, 7
dim_out = (m, n, l, k)
opA1 = Eye(m,n)
opR = BroadCast(opA1, dim_out)
x1 = randn(m,n)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= x1
@test norm(y1-y2) <= 1e-12

@test_throws Exception BroadCast(opA1,(m,m))

m, l = 1, 5
dim_out = (m, l)
opA1 = Eye(m)
opR = BroadCast(opA1, dim_out)
x1 = randn(m)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= x1
@test norm(y1-y2) <= 1e-12

#colum in - matrix out
m, l = 4, 5
dim_out = (m, l)
opA1 = Eye(1,l)
opR = BroadCast(opA1, dim_out)
x1 = randn(1,l)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= x1
@test norm(y1-y2) <= 1e-12

op = HCAT(Eye(m,l),opR)
x1 = (randn(m,l),randn(1,l))
y1 = test_op(op, x1, randn(dim_out), verb)
y2 = x1[1].+x1[2]
@test norm(y1-y2) <= 1e-12

m, n, l  = 2, 5, 8
dim_out = (m, n, l)
opA1 = Eye(m)
opR = BroadCast(opA1, dim_out)
x1 = randn(m)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= x1
@test norm(y1-y2) <= 1e-12

m, n, l  = 1, 5, 8
dim_out = (m, n, l)
opA1 = Eye(m)
opR = BroadCast(opA1, dim_out)
x1 = randn(m)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= x1
@test norm(y1-y2) <= 1e-12

m, n, l  = 1, 5, 8
dim_out = (m, n, l)
opA1 = Scale(2.4,Eye(m))
opR = BroadCast(opA1, dim_out)
x1 = randn(m)
y1 = test_op(opR, x1, randn(dim_out), verb)
y2 = zeros(dim_out)
y2 .= 2.4*x1
@test norm(y1-y2) <= 1e-12

@test is_null(opR)             == is_null(opA1)            
@test is_eye(opR)              == false             
@test is_diagonal(opR)         == false 
@test is_AcA_diagonal(opR)     == false
@test is_AAc_diagonal(opR)     == false
@test is_orthogonal(opR)       == false
@test is_invertible(opR)       == false
@test is_full_row_rank(opR)    == false
@test is_full_column_rank(opR) == false

# test displacement

m, n = 8, 4
dim_out = (m, 10)
A1 = randn(m, n)
d1 = randn(m)
opA1 = AffineAdd(MatrixOp(A1), d1)
opR = BroadCast(opA1, dim_out)
x1 = randn(n)
y1 = opR*x1
y2 = zeros(dim_out)
y2 .= A1*x1+d1
@test norm(y1-y2) <= 1e-12
x1 = randn(n)
y1 = remove_displacement(opR)*x1
y2 = zeros(dim_out)
y2 .= A1*x1
@test norm(y1-y2) <= 1e-12

###########################
##### test AffineAdd  #####
###########################

n,m = 5,6
A = randn(n,m)
opA = MatrixOp(A)
d = randn(n)
T = AffineAdd(opA,d)

println(T)
x1 = randn(m)
y1 = T*x1
@test norm(y1-(A*x1+d)) <1e-9
r = randn(n)
@test norm(T'*r-(A'*r)) <1e-9
@test displacement(T) == d
@test norm(remove_displacement(T)*x1-A*x1) <1e-9

# with sign
T = AffineAdd(opA,d,false)
@test sign(T) == -1

println(T)
x1 = randn(m)
y1 = T*x1
@test norm(y1-(A*x1-d)) <1e-9
r = randn(n)
@test norm(T'*r-(A'*r)) <1e-9
@test displacement(T) == -d
@test norm(remove_displacement(T)*x1-A*x1) <1e-9

# with scalar
T = AffineAdd(opA,pi)
@test sign(T) == 1

println(T)
x1 = randn(m)
y1 = T*x1
@test norm(y1-(A*x1.+pi)) <1e-9
r = randn(n)
@test norm(T'*r-(A'*r)) < 1e-9
@test displacement(T) .- pi < 1e-9
@test norm(remove_displacement(T)*x1-A*x1) <1e-9

@test_throws DimensionMismatch AffineAdd(MatrixOp(randn(2,5)),randn(5))
@test_throws ErrorException AffineAdd(AbstractOperators.DFT(4),randn(4))
AffineAdd(AbstractOperators.DFT(4),pi)
@test_throws ErrorException AffineAdd(Eye(4),im*pi)

# with scalar and vector 
d = randn(n) 
T = AffineAdd(AffineAdd(opA,pi),d,false)

println(T)
x1 = randn(m)
y1 = T*x1
@test norm(y1-(A*x1 .+ pi .- d)) <1e-9
r = randn(n)
@test norm(T'*r-(A'*r)) < 1e-9
@test norm(displacement(T) .- (pi .-d )) < 1e-9

T2 = remove_displacement(T)
@test norm(T2*x1-(A*x1)) <1e-9

# permute AddAffine 
n,m = 5,6
A = randn(n,m)
d = randn(n)
opH = HCAT(Eye(n),MatrixOp(A))
x = (randn(n),randn(m))
opHT = AffineAdd(opH,d)

@test norm(opHT*x-(x[1]+A*x[2].+d)) < 1e-12
p = [2;1]
@test norm(AbstractOperators.permute(opHT,p)*x[p]-(x[1]+A*x[2].+d)) < 1e-12

#############################
######## test combin. #######
#############################

## test Compose of HCAT
m1, m2, m3, m4 = 4, 7, 3, 2
A1 = randn(m3, m1)
A2 = randn(m3, m2)
A3 = randn(m4, m3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opH = HCAT(opA1,opA2)
opC = Compose(opA3,opH)
x1, x2 = randn(m1), randn(m2)
y1 = test_op(opC, (x1,x2), randn(m4), verb)

y2 = A3*(A1*x1+A2*x2)

@test norm(y1-y2) < 1e-9

opCp = AbstractOperators.permute(opC,[2,1])
y1 = test_op(opCp, (x2,x1), randn(m4), verb)

@test norm(y1-y2) < 1e-9

## test HCAT of Compose of HCAT
m5 = 10
A4 = randn(m4,m5)
x3 = randn(m5)
opHC = HCAT(opC,MatrixOp(A4))
x = (x1,x2,x3)
y1 = test_op(opHC, x, randn(m4), verb)

@test norm(y1-(y2+A4*x3)) < 1e-9

p = randperm(ndoms(opHC,2))
opHP = AbstractOperators.permute(opHC,p)

xp = x[p] 

y1 = test_op(opHP, xp, randn(m4), verb)

pp = randperm(ndoms(opHC,2))
opHPP = AbstractOperators.permute(opHC,pp)
xpp = x[pp] 
y1 = test_op(opHPP, xpp, randn(m4), verb)


# test VCAT of HCAT's
m1, m2, n1 = 4, 7, 3
A1 = randn(n1, m1)
A2 = randn(n1, m2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opH1 = HCAT(opA1,opA2)

m1, m2, n2 = 4, 7, 5
A3 = randn(n2, m1)
A4 = randn(n2, m2)
opA3 = MatrixOp(A3)
opA4 = MatrixOp(A4)
opH2 = HCAT(opA3,opA4)

opV = VCAT(opH1,opH2)
x1, x2 = randn(m1), randn(m2)
y1 = test_op(opV, (x1,x2), (randn(n1),randn(n2)), verb)
y2 = (A1*x1+A2*x2,A3*x1+A4*x2)
@test all(norm.(y1 .- y2) .<= 1e-12)

# test VCAT of HCAT's with complex num
m1, m2, n1 = 4, 7, 5
A1 = randn(n1, m1)+im*randn(n1, m1)
opA1 = MatrixOp(A1)
opA2 = AbstractOperators.DFT(n1)
opH1 = HCAT(opA1,opA2)

m1, m2, n2 = 4, 7, 5
A3 = randn(n2, m1)+im*randn(n2,m1)
opA3 = MatrixOp(A3)
opA4 = AbstractOperators.DFT(n2)
opH2 = HCAT(opA3,opA4)

opV = VCAT(opH1,opH2)
x1, x2 = randn(m1)+im*randn(m1), randn(n2)
y1 = test_op(opV, (x1,x2), (randn(n1)+im*randn(n1),randn(n2)+im*randn(n2)), verb)
y2 = (A1*x1+fft(x2),A3*x1+fft(x2))
@test all(norm.(y1 .- y2) .<= 1e-12)

# test HCAT of VCAT's

n1, n2, m1, m2 = 3, 5, 4, 7
A = randn(m1, n1); opA = MatrixOp(A)
B = randn(m1, n2); opB = MatrixOp(B)
C = randn(m2, n1); opC = MatrixOp(C)
D = randn(m2, n2); opD = MatrixOp(D)
opV = HCAT(VCAT(opA, opC), VCAT(opB, opD))
x1 = randn(n1)
x2 = randn(n2)
y1 = test_op(opV, (x1, x2), (randn(m1), randn(m2)), verb)
y2 = (A*x1 + B*x2, C*x1 + D*x2)

@test all(norm.(y1 .- y2) .<= 1e-12)

# test Sum of HCAT's

m, n1, n2, n3 = 4, 7, 5, 3
A1 = randn(m, n1)
A2 = randn(m, n2)
A3 = randn(m, n3)
B1 = randn(m, n1)
B2 = randn(m, n2)
B3 = randn(m, n3)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA3 = MatrixOp(A3)
opB1 = MatrixOp(B1)
opB2 = MatrixOp(B2)
opB3 = MatrixOp(B3)
opHA = HCAT(opA1, opA2, opA3)
opHB = HCAT(opB1, opB2, opB3)
opS = Sum(opHA, opHB)
x1 = randn(n1)
x2 = randn(n2)
x3 = randn(n3)
y1 = test_op(opS, (x1, x2, x3), randn(m), verb)
y2 = A1*x1 + B1*x1 + A2*x2 + B2*x2 + A3*x3 + B3*x3

@test norm(y1-y2) <= 1e-12

p = [3;2;1]
opSp = AbstractOperators.permute(opS,p)
y1 = test_op(opSp, (x1, x2, x3)[p], randn(m), verb)

# test Sum of VCAT's

m1, m2, n = 4, 7, 5
A1 = randn(m1, n)
A2 = randn(m2, n)
B1 = randn(m1, n)
B2 = randn(m2, n)
C1 = randn(m1, n)
C2 = randn(m2, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opB1 = MatrixOp(B1)
opB2 = MatrixOp(B2)
opC1 = MatrixOp(C1)
opC2 = MatrixOp(C2)
opVA = VCAT(opA1, opA2)
opVB = VCAT(opB1, opB2)
opVC = VCAT(opC1, opC2)
opS = Sum(opVA, opVB, opVC)
x = randn(n)
y1 = test_op(opS, x, (randn(m1), randn(m2)), verb)
y2 = (A1*x + B1*x +C1*x, A2*x + B2*x + C2*x)

@test all(norm.(y1 .- y2) .<= 1e-12)

# test Scale of DCAT

m1, n1 = 4, 7
m2, n2 = 3, 5
A1 = randn(m1, n1)
A2 = randn(m2, n2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opD = DCAT(opA1, opA2)
coeff = randn()
opS = Scale(coeff, opD)
x1 = randn(n1)
x2 = randn(n2)
y = test_op(opS, (x1, x2), (randn(m1), randn(m2)), verb)
z = (coeff*A1*x1, coeff*A2*x2)

@test all(norm.(y .- z) .<= 1e-12)

# test Scale of VCAT

m1, m2, n = 4, 3, 7
A1 = randn(m1, n)
A2 = randn(m2, n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opV = VCAT(opA1, opA2)
coeff = randn()
opS = Scale(coeff, opV)
x = randn(n)
y = test_op(opS, x, (randn(m1), randn(m2)), verb)
z = (coeff*A1*x, coeff*A2*x)

@test all(norm.(y .- z) .<= 1e-12)

# test Scale of HCAT

m, n1, n2 = 4, 3, 7
A1 = randn(m, n1)
A2 = randn(m, n2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opH = HCAT(opA1, opA2)
coeff = randn()
opS = Scale(coeff, opH)
x1 = randn(n1)
x2 = randn(n2)
y = test_op(opS, (x1, x2), randn(m), verb)
z = coeff*(A1*x1 + A2*x2)

@test all(norm.(y .- z) .<= 1e-12)

# test DCAT of HCATs

m1, m2, n1, n2, l1, l2, l3 = 2,3,4,5,6,7,8
A1 = randn(m1, n1)
A2 = randn(m1, n2)
B1 = randn(m2, n1)
B2 = randn(m2, n2)
B3 = randn(m2, n2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opH1 = HCAT(opA1, opA2)
opB1 = MatrixOp(B1)
opB2 = MatrixOp(B2)
opB3 = MatrixOp(B3)
opH2 = HCAT(opB1, opB2, opB3)

op = DCAT(opA1, opH2)
x =  randn.(size(op,2))
y0 = randn.(size(op,1))
y = test_op(op, x, y0, verb)

op = DCAT(opH1, opH2)
x =  randn.(size(op,2))
y0 = randn.(size(op,1))
y = test_op(op, x, y0, verb)

p = randperm(ndoms(op,2))
y2 = op[p]*x[p]

@test AbstractOperators.blockvecnorm(y .- y2) <= 1e-8

# test Scale of Sum

m,n = 5,7
A1 = randn(m,n)
A2 = randn(m,n)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opS = Sum(opA1,opA2)
coeff = pi
opSS = Scale(coeff,opS)
x1 = randn(n)
y1 = test_op(opSS, x1, randn(m), verb)
y2 = coeff*(A1*x1+A2*x1)
@test norm(y1-y2) <= 1e-12

# test Scale of Compose

m1, m2, m3 = 4, 7, 3
A1 = randn(m2, m1)
A2 = randn(m3, m2)
opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)

coeff = pi
opC = Compose(opA2,opA1)
opS = Scale(coeff,opC)
x = randn(m1)
y1 = test_op(opS, x, randn(m3), verb)
y2 = coeff*(A2*A1*x)
@test all(norm.(y1 .- y2) .<= 1e-12)

