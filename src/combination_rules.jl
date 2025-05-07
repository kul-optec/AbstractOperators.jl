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

function can_be_combined(T1, T2::BroadCast)
	return size(T1, 1) == size(T1, 2) && can_be_combined(T1, T2.A)
end
combine(T1, T2::BroadCast) = BroadCast(combine(T1, T2.A), T2.dim_out, T2.bufC, T2.bufD)

can_be_combined(L, R::Compose) = can_be_combined(L, R.A[end])
can_be_combined(L::Compose, R::Compose) = can_be_combined(L[1], R.A[end])

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
		bufs = (R.buf[1:(end - 1)]..., allocate_in_domain(combined), L.buf[2:end]...)
	else
		ops = (R.A[1:(end - 1)]..., combined, L.A[2:end]...)
		bufs = (R.buf[1:(end - 1)]..., allocate_in_domain(combined), L.buf[2:end]...)
	end
	return Compose(ops, bufs)
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

can_be_combined(T1::IDCT, T2::DCT) = true
can_be_combined(T1::DCT, T2::IDCT) = true
can_be_combined(T1::DCT, T2::AdjointOperator{<:DCT}) = true
can_be_combined(T1::IDCT, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::IDCT) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::DCT) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::AdjointOperator{<:DCT}) = true
combine(::CosineTransform, T2::CosineTransform) = Eye(allocate_in_domain(T2))
function combine(::CosineTransform, T2::AdjointOperator{<:CosineTransform})
	return Eye(allocate_in_domain(T2))
end
function combine(::AdjointOperator{<:CosineTransform}, T2::CosineTransform)
	return Eye(allocate_in_domain(T2))
end
function combine(
	::AdjointOperator{<:CosineTransform}, T2::AdjointOperator{<:CosineTransform}
)
	return Eye(allocate_in_domain(T2))
end

can_be_combined(T1::IDFT, T2::DFT) = true
can_be_combined(T1::DFT, T2::IDFT) = true
can_be_combined(T1::DFT, T2::AdjointOperator{<:DFT}) = true
can_be_combined(T1::IDFT, T2::AdjointOperator{<:IDFT}) = true
can_be_combined(T1::AdjointOperator{<:DFT}, T2::DFT) = true
can_be_combined(T1::AdjointOperator{<:IDFT}, T2::IDFT) = true
can_be_combined(T1::AdjointOperator{<:DFT}, T2::AdjointOperator{<:IDFT}) = true
can_be_combined(T1::AdjointOperator{<:IDFT}, T2::AdjointOperator{<:DFT}) = true
function combine(::FourierTransform, T2::FourierTransform)
	return Scale(diag_AAc(T2), Eye(allocate_in_domain(T2)))
end
function combine(::FourierTransform, T2::AdjointOperator{<:FourierTransform})
	return Scale(diag_AAc(T2), Eye(allocate_in_domain(T2)))
end
function combine(::AdjointOperator{<:FourierTransform}, T2::FourierTransform)
	return Scale(diag_AAc(T2), Eye(allocate_in_domain(T2)))
end
function combine(
	::AdjointOperator{<:FourierTransform}, T2::AdjointOperator{<:FourierTransform}
)
	return Scale(diag_AAc(T2), Eye(allocate_in_domain(T2)))
end

can_be_combined(T1::DiagOp, T2::DiagOp) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::AdjointOperator{<:DiagOp}) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::DiagOp, T2::DiagOp)
	return DiagOp(domainType(T2), T2.dim_in, T1.d .* T2.d)
end
function combine(T1::DiagOp, T2::AdjointOperator{<:DiagOp})
	return DiagOp(domainType(T2), size(T2, 2), T1.d .* conj.(T2.A.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::DiagOp)
	return DiagOp(domainType(T2), size(T2, 2), conj.(T1.A.d) .* T2.d)
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp})
	return DiagOp(domainType(T2), size(T2, 2), conj.(T1.A.d) .* conj.(T2.A.d))
end

can_be_combined(T1, ::Eye) = true
can_be_combined(T1, ::AdjointOperator{<:Eye}) = true
combine(T1, ::Eye) = T1
combine(T1, ::AdjointOperator{<:Eye}) = T1

can_be_combined(::MatrixOp, ::MatrixOp) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::MatrixOp) = true
can_be_combined(::MatrixOp, ::AdjointOperator{<:MatrixOp}) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::AdjointOperator{<:MatrixOp}) = true
function combine(T1::MatrixOp, T2::MatrixOp)
	return MatrixOp(domainType(T2), size(T2, 2), T1.A * T2.A)
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domainType(T2), size(T2, 2), T1.A * T2.A.A')
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::MatrixOp)
	return MatrixOp(domainType(T2), size(T2, 2), T1.A.A' * T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domainType(T2), size(T2, 2), T1.A.A' * T2.A.A')
end

can_be_combined(T1::MatrixOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::Scale) = true
can_be_combined(T1::Scale, T2::MatrixOp) = can_be_combined(T1.A, T2)
can_be_combined(T1::Scale, T2::AdjointOperator{<:MatrixOp}) = can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::MatrixOp) = can_be_combined(T1.A.A', T2)
function can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp})
	return can_be_combined(T1.A.A', T2)
end
function combine(T1::MatrixOp, T2::Scale)
	return combine(MatrixOp(domainType(T1), size(T1, 2), T2.coeff * T1.A), T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::Scale)
	return combine(MatrixOp(domainType(T1), size(T1, 2), T2.coeff * T1.A.A'), T2.A)
end
function combine(T1::Scale, T2::MatrixOp)
	return Scale(T1.coeff, combine(T1.A, T2))
end
function combine(T1::Scale, T2::AdjointOperator{<:MatrixOp})
	return Scale(T1.coeff, combine(T1.A, T2))
end
function combine(T1::AdjointOperator{<:Scale}, T2::MatrixOp)
	return Scale(T1.A.coeff_conj, combine(T1.A.A', T2))
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp})
	return Scale(
		T1.A.coeff_conj, combine(T1.A.A', MatrixOp(domainType(T2), size(T2, 2), T2.A'))
	)
end

can_be_combined(T1::Scale, T2::DiagOp) = can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::Scale, T2::DiagOp)
	return Scale(T1.coeff, combine(T1.A, T2))
end
function combine(T1::AdjointOperator{<:Scale}, T2::DiagOp)
	scaled_diagop = DiagOp(domainType(T2), size(T2, 2), T1.A.coeff_conj * T2.d)
	return can_be_combined(T1.A.A', scaled_diagop) ? combine(T1.A.A', scaled_diagop) : T1.A.A' * scaled_diagop
end
function combine(T1::DiagOp, T2::Scale)
	scaled_diagop = DiagOp(domainType(T1), size(T1, 2), T1.d .* T2.coeff)
	return can_be_combined(scaled_diagop, T2.A) ? combine(scaled_diagop, T2.A) : scaled_diagop * T2.A
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::Scale)
	scaled_diagop = DiagOp(domainType(T1), size(T1, 2), conj.(T1.A.d) .* T2.coeff)
	return can_be_combined(scaled_diagop, T2.A) ? combine(scaled_diagop, T2.A) : scaled_diagop * T2.A
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp})
	scaled_diagop = DiagOp(domainType(T2), size(T2, 2), T1.A.coeff_conj .* conj.(T2.A.d))
	return can_be_combined(T2.A.A', scaled_diagop) ? combine(T2.A.A', scaled_diagop) : T2.A.A' * scaled_diagop
end

can_be_combined(T1::DiagOp, T2::MatrixOp) = codomainType(T1) == domainType(T2)
can_be_combined(T1::MatrixOp, T2::DiagOp) = codomainType(T1) == domainType(T2)
function can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::MatrixOp)
	return codomainType(T1) == domainType(T2)
end
function can_be_combined(T1::DiagOp, T2::AdjointOperator{<:MatrixOp})
	return codomainType(T1) == domainType(T2)
end
function can_be_combined(T1::MatrixOp, T2::AdjointOperator{<:DiagOp})
	return codomainType(T1) == domainType(T2)
end
function can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::DiagOp)
	return codomainType(T1) == domainType(T2)
end
function can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:MatrixOp})
	return codomainType(T1) == domainType(T2)
end
function can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:DiagOp})
	return codomainType(T1) == domainType(T2)
end
combine_matrix(L::AbstractMatrix, R::AbstractMatrix) = L * R
combine_matrix(L::AbstractMatrix, R::AbstractVector) = L * Diagonal(R)
combine_matrix(L::AbstractVector, R::AbstractMatrix) = Diagonal(L) * R
combine_matrix(L::Number, R::AbstractMatrix) = L * R
combine_matrix(L::AbstractMatrix, R::Number) = R * L
function combine(T1::DiagOp, T2::MatrixOp)
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.d, T2.A))
end
function combine(T1::MatrixOp, T2::DiagOp)
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.A, T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::MatrixOp)
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A))
end
function combine(T1::DiagOp, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.d, T2.A'))
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:DiagOp})
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.A, conj(T2.A.d)))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::DiagOp)
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.A.A', T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:MatrixOp})
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A.A'))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:DiagOp})
	return MatrixOp(domainType(T2), size(T2, 2), combine_matrix(T1.A.A', conj(T2.A.d)))
end
