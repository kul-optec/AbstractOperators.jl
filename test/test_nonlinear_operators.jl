verb && @printf("\nTesting non linear operators\n")

# Sigmoid
n = 4
x = randn(n)
r = randn(n)
op = Sigmoid(Float64, (n,), 2)

y, grad = test_NLop(op, x, r, verb)

n, m, l = 4, 5, 6
x = randn(n, m)
r = randn(n, m)
op = Sigmoid((n, m), 2)

y, grad = test_NLop(op, x, r, verb)

n = 10
x = randn(n)
r = randn(n)
op = SoftMax(Float64, (n,))

y, grad = test_NLop(op, x, r, verb)

# SoftMax
n, m, l = 4, 5, 6
x = randn(n, m, l)
r = randn(n, m, l)
op = SoftMax(Float64, (n, m, l))

y, grad = test_NLop(op, x, r, verb)

# SoftPlus
n = 10
x = randn(n)
r = randn(n)
op = SoftPlus(Float64, (n,))

n, m, l = 4, 5, 6
x = randn(n, m, l)
r = randn(n, m, l)
op = SoftPlus(Float64, (n, m, l))

y, grad = test_NLop(op, x, r, verb)

# Exp
n, m, l = 4, 5, 6
x = randn(n, m, l)
r = randn(n, m, l)
op = Exp(n, m, l)
op = Exp(Float64, (n, m, l))

y, grad = test_NLop(op, x, r, verb)

## Sin
n, m, l = 4, 5, 6
x = randn(n, m, l)
r = randn(n, m, l)
op = Sin(n, m, l)
op = Sin(Float64, (n, m, l))

y, grad = test_NLop(op, x, r, verb)

## Cos
n, m, l = 4, 5, 6
x = randn(n, m, l)
r = randn(n, m, l)
op = Cos(n, m, l)
op = Cos(Float64, (n, m, l))

y, grad = test_NLop(op, x, r, verb)

# Atan
n = 10
x = randn(n)
r = randn(n)
op = Atan(n)
op = Atan(Float64, (n,))

y, grad = test_NLop(op, x, r, verb)

# Tanh
n = 10
x = randn(n)
r = randn(n)
op = Tanh(n)
op = Tanh(Float64, (n,))

y, grad = test_NLop(op, x, r, verb)

# Sech
n = 10
x = randn(n)
r = randn(n)
op = Sech(n)
op = Sech(Float64, (n,))

y, grad = test_NLop(op, x, r, verb)

# Pow
n = 10
x = randn(n)
r = randn(n)
op = Pow(Float64, (n,), 2)

y, grad = test_NLop(op, x, r, verb)

n = 10
x = abs.(randn(n))
r = abs.(randn(n))
op = Pow(Float64, (n,), 0.5)

y, grad = test_NLop(op, x, r, verb)
