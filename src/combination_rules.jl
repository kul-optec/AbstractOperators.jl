function can_be_combined(T1::AffineAdd, T2::AffineAdd)
	return is_linear(T1.A) && can_be_combined(T1.A, T2.A)
end
function combine(T1::AffineAdd{L1,D1,S1}, T2::AffineAdd{L2,D2,S2}) where {L1,D1,S1,L2,D2,S2}
	new_d = T1.A * T2.d
	if S1 == S2
		new_d .+= T1.d
	else
		new_d .-= T1.d
	end
	return AffineAdd(combine(T1.A, T2.A), new_d, S2)
end

can_be_combined(T1, T2::AffineAdd) = is_linear(T1) && can_be_combined(T1, T2.A)
function combine(T1, T2::AffineAdd{L,D,S}) where {L,D,S}
	new_d = if T2.d isa Number
		temp = allocate_in_domain(T1)
		temp .= T2.d
		T1 * temp
	else
		T1 * T2.d
	end
	return AffineAdd(combine(T1, T2.A), new_d, S)
end

can_be_combined(L, R::Compose) = can_be_combined(L, R.A[end])
can_be_combined(L::Compose, R::Compose) = can_be_combined(L.A[1], R.A[end])
function can_be_combined(L::Scale, R::Compose)
	can_be_combined(L.A, R.A[end]) || (
		is_linear(L.A) &&
		all(is_linear.(R.A)) &&
		any(op -> op isa Union{Scale,DiagOp,MatrixOp,LMatrixOp}, R.A)
	)
end
function can_be_combined(L::AdjointOperator{<:Scale}, R::Compose)
	can_be_combined(L.A.A', R.A[end]) || (
		is_linear(L.A.A') &&
		all(is_linear.(R.A)) &&
		any(op -> op isa Union{Scale,DiagOp,MatrixOp,LMatrixOp}, R.A)
	)
end
function can_be_combined(L::Compose, R::Scale)
	can_be_combined(L.A[end], R.A) || (
		is_linear(R.A) &&
		all(is_linear.(L.A)) &&
		any(op -> op isa Union{Scale,DiagOp,MatrixOp,LMatrixOp}, L.A)
	)
end
function can_be_combined(L::Compose, R::AdjointOperator{<:Scale})
	can_be_combined(L.A[end], R.A.A') || (
		is_linear(R.A.A') &&
		all(is_linear.(L.A)) &&
		any(op -> op isa Union{Scale,DiagOp,MatrixOp,LMatrixOp}, L.A)
	)
end

function combine(L, R::Compose)
	combined = combine(L, R.A[end])
	if combined isa Compose
		ops = (R.A[1:(end - 1)]..., combined.A...)
		bufs = (R.buf..., combined.buf...)
	else
		ops = (R.A[1:(end - 1)]..., combined)
		bufs = R.buf
	end
	return Compose(ops, bufs)
end
function combine(L::Compose, R::Compose)
	combined = combine(L.A[1], R.A[end])
	if combined isa Compose
		ops = (R.A[1:(end - 1)]..., combined.A..., L.A[2:end]...)
		bufs = (R.buf..., combined.buf..., L.buf...)
	else
		ops = (R.A[1:(end - 1)]..., combined, L.A[2:end]...)
		bufs = (R.buf..., L.buf...)
	end
	return Compose(ops, bufs)
end
function combine(L::Scale{T,O,Th}, R::Compose) where {T,O,Th}
	threaded = Th == FastBroadcast.True()
	if can_be_combined(L.A, R.A[end])
		return Scale(L.coeff, L.coeff_conj, combine(L.A, R); threaded) # forward optimization task to combine function
	else
		return Scale(L.coeff, L.A * R; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
	end
end
function combine(L::AdjointOperator{Scale{T,O,Th}}, R::Compose) where {T,O,Th}
	threaded = Th == FastBroadcast.True()
	if can_be_combined(L.A.A', R.A[end])
		return Scale(L.A.coeff_conj, L.A.coeff, combine(L.A.A', R); threaded) # forward optimization task to combine function
	else
		return Scale(L.A.coeff_conj, L.A.A' * R; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
	end
end
function combine(L::Compose, R::Scale{T,O,Th}) where {T<:Number,O<:AbstractOperator,Th}
	threaded = Th == FastBroadcast.True()
	if can_be_combined(L.A[1], R.A)
		S = Scale(R.coeff, R.coeff_conj, combine(L.A[1], R); threaded) # forward optimization task to combine function
		return Compose((S, L.A[2:end]...), L.buf)
	else
		return Scale(R.coeff, L * R.A; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
	end
end
function combine(
	L::Compose, R::AdjointOperator{Scale{T,O,Th}}
) where {T<:Number,O<:AbstractOperator,Th}
	threaded = Th == FastBroadcast.True()
	if can_be_combined(L.A[1], R.A.A')
		S = Scale(R.A.coeff_conj, R.A.coeff, combine(L.A[1], R.A.A'); threaded) # forward optimization task to combine function
		return Compose((S, L.A[2:end]...), L.buf)
	else
		return Scale(R.A.coeff_conj, L * R.A.A'; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
	end
end

function can_be_combined(L::DCAT, R::DCAT)
	return length(L.A) == length(R.A) &&
		   L.idxD == R.idxD &&
		   L.idxC == R.idxC &&
		   all(can_be_combined(L.A[i], R.A[i]) for i in eachindex(L.A))
end

function combine(L::DCAT, R::DCAT)
	return DCAT([combine(L.A[i], R.A[i]) for i in 1:ndoms(L, 1)]...)
end

function can_be_combined(L1, L2::HCAT)
	return is_diagonal(L1) && all(can_be_combined(L1, A) for A in L2.A)
end
function combine(L1, L2::HCAT)
	combined = tuple([combine(L1, A) for A in L2.A]...)
	return HCAT(combined, L2.idxs, L2.buf)
end

can_be_combined(L, R::Scale) = is_linear(L) && can_be_combined(L, R.A)
can_be_combined(L, R::AdjointOperator{<:Scale}) = can_be_combined(R.A.A, L')
combine(L, R::Scale) = Scale(R.coeff, combine(L, R.A))
combine(L, R::AdjointOperator{<:Scale}) = Scale(R.A.coeff, combine(R.A.A, L'))'

function can_be_combined(L, R::Sum)
	return is_linear(L) && all(can_be_combined(L, A) for A in R.A)
end
function combine(L, R::Sum)
	ops = tuple([combine(L, A) for A in R.A]...)
	return size(L, 1) == size(L, 2) ? Sum(ops, R.bufC, R.bufD) : Sum(ops...)
end

can_be_combined(T1::DiagOp, T2::DiagOp) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::AdjointOperator{<:DiagOp}) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::DiagOp, T2::DiagOp)
	return DiagOp(domain_type(T2), T2.dim_in, T1.d .* T2.d)
end
function combine(T1::DiagOp, T2::AdjointOperator{<:DiagOp})
	return DiagOp(domain_type(T2), size(T2, 2), T1.d .* conj.(T2.A.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::DiagOp)
	return DiagOp(domain_type(T2), size(T2, 2), conj.(T1.A.d) .* T2.d)
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp})
	return DiagOp(domain_type(T2), size(T2, 2), conj.(T1.A.d) .* conj.(T2.A.d))
end

can_be_combined(T1, ::Eye) = true
combine(T1, ::Eye) = T1

can_be_combined(::MatrixOp, ::MatrixOp) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::MatrixOp) = true
can_be_combined(::MatrixOp, ::AdjointOperator{<:MatrixOp}) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::AdjointOperator{<:MatrixOp}) = true
function combine(T1::MatrixOp, T2::MatrixOp)
	return MatrixOp(domain_type(T2), size(T2, 2), T1.A * T2.A)
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domain_type(T2), size(T2, 2), T1.A * T2.A.A')
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::MatrixOp)
	return MatrixOp(domain_type(T2), size(T2, 2), T1.A.A' * T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domain_type(T2), size(T2, 2), T1.A.A' * T2.A.A')
end

can_be_combined(T1::MatrixOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::Scale) = true
can_be_combined(T1::Scale, T2::MatrixOp) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::Scale, T2::AdjointOperator{<:MatrixOp}) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::MatrixOp) = true
can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp}) = true
function combine(T1::MatrixOp, T2::Scale)
	return Compose(MatrixOp(T2.coeff * T1.A), T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::Scale)
	return Compose(MatrixOp(T2.coeff * T1.A.A'), T2.A)
end
function combine(T1::Scale, T2::MatrixOp)
	if can_be_combined(T1.A, T2)
		return Scale(T1.coeff, combine(T1.A, T2))
	else
		return Compose(T1.A, MatrixOp(T1.coeff * T2.A))
	end
end
function combine(T1::Scale, T2::AdjointOperator{<:MatrixOp})
	if can_be_combined(T1.A, T2)
		return Scale(T1.coeff, combine(T1.A, T2))
	else
		return Compose(T1.A, MatrixOp(T1.coeff * T2.A.A'))
	end
end
function combine(T1::AdjointOperator{<:Scale}, T2::MatrixOp)
	return Compose(T1.A.A', MatrixOp(T1.A.coeff_conj * T2.A))
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp})
	return Compose(T1.A.A', MatrixOp(T1.A.coeff_conj * T2.A.A'))
end

can_be_combined(T1::Scale, T2::DiagOp) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::Scale, T2::DiagOp)
	if can_be_combined(T1.A, T2)
		return Scale(T1.coeff, combine(T1.A, T2))
	else
		scaled_diagop = T1.coeff * T2
		return T1.A * scaled_diagop
	end
end
function combine(T1::AdjointOperator{<:Scale}, T2::DiagOp)
	scaled_diagop = DiagOp(domain_type(T2), size(T2, 2), T1.A.coeff_conj * T2.d)
	return if can_be_combined(T1.A.A', scaled_diagop)
		combine(T1.A.A', scaled_diagop)
	else
		T1.A.A' * scaled_diagop
	end
end
function combine(T1::DiagOp, T2::Scale)
	scaled_diagop = DiagOp(domain_type(T1), size(T1, 2), T1.d .* T2.coeff)
	return if can_be_combined(scaled_diagop, T2.A)
		combine(scaled_diagop, T2.A)
	else
		scaled_diagop * T2.A
	end
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::Scale)
	scaled_diagop = DiagOp(domain_type(T1), size(T1, 2), conj.(T1.A.d) .* T2.coeff)
	return if can_be_combined(scaled_diagop, T2.A)
		combine(scaled_diagop, T2.A)
	else
		scaled_diagop * T2.A
	end
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp})
	scaled_diagop = DiagOp(domain_type(T2), size(T2, 2), T1.A.coeff_conj .* conj.(T2.A.d))
	return if can_be_combined(T1.A.A', scaled_diagop)
		combine(T1.A.A', scaled_diagop)
	else
		T1.A.A' * scaled_diagop
	end
end

can_be_combined(T1::DiagOp, T2::MatrixOp) = codomain_type(T1) == domain_type(T2)
can_be_combined(T1::MatrixOp, T2::DiagOp) = codomain_type(T1) == domain_type(T2)
function can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::MatrixOp)
	return codomain_type(T1) == domain_type(T2)
end
function can_be_combined(T1::DiagOp, T2::AdjointOperator{<:MatrixOp})
	return codomain_type(T1) == domain_type(T2)
end
function can_be_combined(T1::MatrixOp, T2::AdjointOperator{<:DiagOp})
	return codomain_type(T1) == domain_type(T2)
end
function can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::DiagOp)
	return codomain_type(T1) == domain_type(T2)
end
function can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:MatrixOp})
	return codomain_type(T1) == domain_type(T2)
end
function can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:DiagOp})
	return codomain_type(T1) == domain_type(T2)
end
combine_matrix(L::AbstractMatrix, R::AbstractMatrix) = L * R
combine_matrix(L::AbstractMatrix, R::AbstractVector) = L * Diagonal(R)
combine_matrix(L::AbstractVector, R::AbstractMatrix) = Diagonal(L) * R
combine_matrix(L::Number, R::AbstractMatrix) = L * R
combine_matrix(L::AbstractMatrix, R::Number) = R * L
function combine(T1::DiagOp, T2::MatrixOp)
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.d, T2.A))
end
function combine(T1::MatrixOp, T2::DiagOp)
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A, T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::MatrixOp)
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A))
end
function combine(T1::DiagOp, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.d, T2.A'))
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:DiagOp})
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A, conj(T2.A.d)))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::DiagOp)
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A.A', T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A.A'))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:DiagOp})
	return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A.A', conj(T2.A.d)))
end
