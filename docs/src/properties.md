# Properties and Traits

## Size and Domains

```@docs
size
ndims
ndoms
domain_type
codomain_type
domain_storage_type
codomain_storage_type
```

## Traits

The functions in this package allows querying properties of the operators to be used by other packages for optimization and trait-based custumized behavior.

### LinearAlgebra traits

AbstractOperators.jl's operators is defined for following functions from standard library LinearAlgebra.

```@docs
diag
opnorm
```

### OperatorCore.jl traits

AbstractOperators.jl's operators satisfies the interface for matrix-like objects provided by [OperatorCore.jl](https://github.com/hakkelt/OperatorCore.jl) with the following functions:

```@docs
is_linear
is_eye
is_null
is_symmetric
is_diagonal
is_AcA_diagonal
diag_AcA
is_AAc_diagonal
diag_AAc
is_orthogonal
is_invertible
is_full_row_rank
is_full_column_rank
```

### Additional traits

AbstractOperators.jl's operators also defines the following functions:

```@docs
displacement
remove_displacement
is_sliced
remove_slicing
is_thread_safe
estimate_opnorm
```
