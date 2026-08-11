@testsnippet GPUNLTestUtils begin
    using Test
    using LinearAlgebra
    using RecursiveArrayTools
    using AbstractOperators

    _to_cpu(x::AbstractArray) = collect(x)
    _to_cpu(x::RecursiveArrayTools.ArrayPartition) =
        RecursiveArrayTools.ArrayPartition(collect.(x.x)...)

    function _assert_cpu_approx(x, y; atol = 1.0e-8)
        @test norm(_to_cpu(x) .- _to_cpu(y)) <= atol
    end

    function test_NLop_gpu(A::AbstractOperator, x, y, verb::Bool = false)
        verb && (println(), println(A))

        Ax = A * x
        Ax2 = similar(Ax)
        mul!(Ax2, A, x)
        _assert_cpu_approx(Ax, Ax2)

        @test_throws ErrorException A'

        J = Jacobian(A, x)
        grad = J' * y
        mul!(Ax2, A, x)
        grad2 = similar(grad)
        mul!(grad2, J', y)
        _assert_cpu_approx(grad, grad2; atol = 1.0e-8)

        return Ax, grad
    end
end
