const thread_count_functions = Ref([
	BLAS.get_num_threads => BLAS.set_num_threads,
	FFTW.get_num_threads => FFTW.set_num_threads,
])

function set_thread_counts_expr(thread_count_expr, body_expr)
	quote
		local prev_thread_counts = [pair.first() for pair in thread_count_functions[]]
		for pair in thread_count_functions[]
			pair.second($thread_count_expr)
		end
		local res = $(esc(body_expr))
		for (i, pair) in enumerate(thread_count_functions[])
			pair.second(prev_thread_counts[i])
		end
		res
	end
end

macro enable_threading(expr)
	return set_thread_counts_expr(nthreads(), expr)
end

macro disable_threading(expr)
	return set_thread_counts_expr(1, expr)
end
