
########## WaveletOp ############
n = 8
op = WaveletOp(Float64, wavelet(WT.db4), (n,))
x1 = randn(n)
y1 = test_op(op, x1, randn(n), verb)
y2 = dwt(x1, wavelet(WT.db4))

@test all(norm.(y1 .- y2) .<= 1e-12)

n = 8
op = WaveletOp(ComplexF64, wavelet(WT.db4), (n,))
x1 = randn(ComplexF64, n)
y1 = test_op(op, x1, randn(ComplexF64, n), verb)
y2 = dwt(x1, wavelet(WT.db4))

@test all(norm.(y1 .- y2) .<= 1e-12)
