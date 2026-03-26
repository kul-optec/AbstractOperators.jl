# Compile-time ndoms from operator type, for use in @generated constructors.
# Specializations for HCAT/VCAT/DCAT are added in their respective files.
_ndoms_from_type(::Type{<:AbstractOperator}, dim::Int) = 1

const thread_count_functions = Ref{Vector{Pair{Function, Function}}}(
    Pair{Function, Function}[
        BLAS.get_num_threads => BLAS.set_num_threads,
    ]
)

# Non-inlined helpers so that the abstract Function dispatch is contained in
# AbstractOperators and not inlined into the calling module (which would cause
# JET @test_opt findings when target_modules excludes AbstractOperators).
@noinline function _save_thread_counts()
    return [pair.first() for pair in thread_count_functions[]]
end

@noinline function _apply_thread_counts(n::Int)
    for pair in thread_count_functions[]
        pair.second(n)
    end
end

@noinline function _restore_thread_counts(prev::Vector)
    for (i, pair) in enumerate(thread_count_functions[])
        pair.second(prev[i])
    end
end

function set_thread_counts_expr(thread_count_expr, body_expr)
    return quote
        local prev_thread_counts = AbstractOperators._save_thread_counts()
        AbstractOperators._apply_thread_counts($thread_count_expr)
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
            AbstractOperators._restore_thread_counts(prev_thread_counts)
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
    # Use isequal instead of != to avoid Union{Missing,Bool} from tuple comparisons
    if !isequal(dtype, domain_type(op))
        throw(
            ArgumentError(
                "Input type $(dtype) does not match operator input type $(domain_type(op))",
            ),
        )
    end
    dim_in = domain_array isa ArrayPartition ? size.(domain_array.x) : size(domain_array)
    if !isequal(dim_in, size(op, 2))
        throw(
            DimensionMismatch(
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
    if !isequal(dtype, codomain_type(op))
        throw(
            ArgumentError(
                "Output type $(dtype) does not match operator output type $(codomain_type(op))",
            ),
        )
    end
    dim_out = codomain_array isa ArrayPartition ? size.(codomain_array.x) : size(codomain_array)
    return if !isequal(dim_out, size(op, 1))
        throw(
            DimensionMismatch(
                "Output size $(dim_out) does not match operator output size $(size(op, 1))",
            ),
        )
    end
end
