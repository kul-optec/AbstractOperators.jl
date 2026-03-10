using TestItemRunner

# Run all tests. Tags enable selective filtering for interactive development:
#
#   julia --project=test -e '
#       using TestItemRunner
#       TestItemRunner.run_tests(pwd(); filter = ti -> :calculus in ti.tags)
#   '
#
# Available tags: :calculus, :linearoperator, :nonlinearoperator,
#                 :batching, :misc, :quality, :jet

@run_package_tests
