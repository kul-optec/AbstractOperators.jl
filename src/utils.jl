const thread_count_functions = Ref{Vector{Pair{Function,Function}}}(Pair{Function,Function}[
	BLAS.get_num_threads => BLAS.set_num_threads,
])

function set_thread_counts_expr(thread_count_expr, body_expr)
	quote
		local prev_thread_counts = [pair.first() for pair in AbstractOperators.thread_count_functions[]]
		for pair in AbstractOperators.thread_count_functions[]
			pair.second($thread_count_expr)
		end
		local res
		try
			if $thread_count_expr == 1
				res = disable_polyester_threads() do
					$(esc(body_expr))
				end
			else
				# Full threading enabled
				res = $(esc(body_expr))
			end
		finally
			# Restore previous thread counts
			for (i, pair) in enumerate(AbstractOperators.thread_count_functions[])
				pair.second(prev_thread_counts[i])
			end
		end
		res
	end
end

macro enable_full_threading(expr)
	return set_thread_counts_expr(nthreads(), expr)
end

macro restrict_threading(expr)
	return set_thread_counts_expr(1, expr)
end

function check(codomain_array, op, domain_array)
	if domain_array isa AbstractArray === false
		throw(ArgumentError("Input must be an AbstractArray"))
	end
	if codomain_array isa AbstractArray === false
		throw(ArgumentError("Output must be an AbstractArray"))
	end
	if (ndoms(op, 2) > 1) != (domain_array isa ArrayPartition)
		throw(ArgumentError("Input must be an ArrayPartition if and only if operator has multiple input domains"))
	end
	if domain_array isa ArrayPartition
		dtype = eltype.(domain_array.x)
	else
		dtype = eltype(domain_array)
	end
	if dtype != domain_type(op)
		throw(
			ArgumentError(
				"Input type $(dtype) does not match operator input type $(domain_type(op))",
			),
		)
	end
	dim_in = domain_array isa ArrayPartition ? size.(domain_array.x) : size(domain_array)
	if dim_in != size(op, 2)
		throw(
			ArgumentError(
				"Input size $(dim_in) does not match operator input size $(size(op, 2))",
			),
		)
	end
	if (ndoms(op, 1) > 1) != (codomain_array isa ArrayPartition)
		throw(ArgumentError("Output must be an ArrayPartition if and only if operator has multiple output domains"))
	end
	if codomain_array isa ArrayPartition
		dtype = eltype.(codomain_array.x)
	else
		dtype = eltype(codomain_array)
	end
	if dtype != codomain_type(op)
		throw(
			ArgumentError(
				"Output type $(dtype) does not match operator output type $(codomain_type(op))",
			),
		)
	end
	dim_out = codomain_array isa ArrayPartition ? size.(codomain_array.x) : size(codomain_array)
	if dim_out != size(op, 1)
		throw(
			ArgumentError(
				"Output size $(dim_out) does not match operator output size $(size(op, 1))",
			),
		)
	end
end
