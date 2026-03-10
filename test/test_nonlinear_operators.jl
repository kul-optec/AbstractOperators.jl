@testitem "NonlinearOp: Sigmoid" tags = [:nonlinearoperator, :Sigmoid] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

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
end

@testitem "NonlinearOp: SoftMax" tags = [:nonlinearoperator, :SoftMax] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = SoftMax(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = SoftMax(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: SoftPlus" tags = [:nonlinearoperator, :SoftPlus] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = SoftPlus(Float64, (n,))

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = SoftPlus(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Exp" tags = [:nonlinearoperator, :Exp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Exp(n, m, l)
    op = Exp(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Sin" tags = [:nonlinearoperator, :Sin] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Sin(n, m, l)
    op = Sin(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Cos" tags = [:nonlinearoperator, :Cos] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Cos(n, m, l)
    op = Cos(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Atan" tags = [:nonlinearoperator, :Atan] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Atan(n)
    op = Atan(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Tanh" tags = [:nonlinearoperator, :Tanh] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Tanh(n)
    op = Tanh(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Sech" tags = [:nonlinearoperator, :Sech] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Sech(n)
    op = Sech(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Pow" tags = [:nonlinearoperator, :Pow] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Pow(Float64, (n,), 2)
    y, grad = test_NLop(op, x, r, verb)

    x = abs.(randn(n))
    r = abs.(randn(n))
    op = Pow(Float64, (n,), 0.5)
    y, grad = test_NLop(op, x, r, verb)
end
