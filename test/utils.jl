########### Test for LinearOperators
function test_op(A::AbstractOperator, x, y, verb::Bool = false)

  verb && (println(); show(A); println())

  Ax = A*x
  Ax2 = AbstractOperators.blocksimilar(Ax)
  verb && println("forward preallocated")
  A_mul_B!(Ax2, A, x) #verify in-place linear operator works
  verb && @time A_mul_B!(Ax2, A, x)

  @test AbstractOperators.blockvecnorm(Ax .- Ax2) <= 1e-8

  Acy = A'*y
  Acy2 = AbstractOperators.blocksimilar(Acy)
  verb && println("adjoint preallocated")
  Ac_mul_B!(Acy2, A, y) #verify in-place linear operator works
  verb && @time Ac_mul_B!(Acy2, A, y)

  @test AbstractOperators.blockvecnorm(Acy .- Acy2) <= 1e-8

  s1 = real(AbstractOperators.blockvecdot(Ax2, y))
  s2 = real(AbstractOperators.blockvecdot(x, Acy2))
  @test abs( s1 - s2 ) < 1e-8

  return Ax
end

########### Test for LinearOperators
function test_NLop(A::AbstractOperator, x, y, verb::Bool = false)

	verb && (println(),println(A))

	Ax = A*x
	Ax2 = AbstractOperators.blocksimilar(Ax)
	verb && println("forward preallocated")
	A_mul_B!(Ax2, A, x) #verify in-place linear operator works
	verb && @time A_mul_B!(Ax2, A, x)

	@test_throws ErrorException A'

	@test AbstractOperators.blockvecnorm(Ax .- Ax2) <= 1e-8

	J = Jacobian(A,x)
	verb && println(J)

	grad = J'*y
	A_mul_B!(Ax2, A, x) #redo forward
	verb && println("jacobian Ac_mul_B! preallocated")
	grad2 = AbstractOperators.blocksimilar(grad)
	Ac_mul_B!(grad2, J, y) #verify in-place linear operator works
	verb && A_mul_B!(Ax2, A, x) #redo forward
	verb && @time Ac_mul_B!(grad2, J, y) 

	@test AbstractOperators.blockvecnorm(grad .- grad2) < 1e-8

	grad3 = gradient_fd(A,Ax,x,y) #calculate gradient using finite differences

	@test AbstractOperators.blockvecnorm(grad .- grad3) < 1e-4

	return Ax, grad
end

############# Finite Diff for Jacobian tests


function gradient_fd{A<:AbstractOperator}(op::A, 
					  y0::AbstractArray, 
					  x0::AbstractArray, 
					  r::AbstractArray) 
	
	y = copy(y0)
	J =  zeros(*(size(op,1)...),*(size(op,2)...))
	h = sqrt(eps())
	for i = 1:size(J,2)
		x = copy(x0)
		x[i] = x[i]+h
		A_mul_B!(y,op,x)
		J[:,i] .= ((y.-y0)./h)[:]
	end
	return reshape(J'*r[:],size(op,2))
end

function gradient_fd{N, A<:AbstractOperator}(op::A, 
					     y0::AbstractArray, 
					     x0::NTuple{N,AbstractArray},
					     r::AbstractArray) 
	

	y = copy(y0)
	grad = zeros.(x0)
	J =  [ zeros(*(size(op,1)...),*(sz2...)) for sz2 in size(op,2)]

	h = sqrt(eps())
	for ii in eachindex(J)
		for i = 1:size(J[ii],2)
			x = deepcopy(x0)
			x[ii][i] = x[ii][i]+h
			A_mul_B!(y,op,x)
			J[ii][:,i] .= ((y.-y0)./h)[:]
		end
		grad[ii] .= reshape(J[ii]'*r[:],size(op,2)[ii])
	end
	return grad
end

function gradient_fd{N, A<:AbstractOperator}(op::A, 
					     y0::NTuple{N,AbstractArray}, 
					     x0::AbstractArray, 
					     r::NTuple{N,AbstractArray}) 
	
	y = zeros.(y0)
	grad = zeros(x0)
	J = [ zeros(*(sz1...),*(size(op,2)...)) for sz1 in size(op,1)]

	h = sqrt(eps())
	for i in eachindex(x0)
		x = deepcopy(x0)
		x[i] = x[i]+h
		A_mul_B!(y,op,x)
		for ii in eachindex(J)
			J[ii][:,i] .= ((y[ii].-y0[ii])./h)[:]
		end
	end
	for ii in eachindex(J)
		grad .+= reshape(J[ii]'*r[ii],size(op,2))
	end
	return grad
end

function gradient_fd{N,M, A<:AbstractOperator}(op::A, 
					       y0::NTuple{N,AbstractArray}, 
					       x0::NTuple{M,AbstractArray}, 
					       r::NTuple{N,AbstractArray}) 
	grad = zeros.(x0)
	y    = zeros.(y0)
	J = [ zeros(*(size(op,1)[i]...),*(size(op,2)[ii]...)) for ii = 1:M, i = 1:N ]
				

	h = sqrt(eps())
	for i = 1:M
		for iii in eachindex(x[i])
			x = deepcopy(x0)
			x[i][iii] = x[i][iii]+h
			A_mul_B!(y,op,x)

			for ii = 1:N
				J[i,ii][:,iii] .= ((y[ii].-y0[ii])./h)[:]
			end
		end
	end

	for ii = 1:N, i = 1:M
		grad[i] .+= reshape(J[i,ii]'*r[ii],size(op,2)[i])
	end
	return grad

	
end

