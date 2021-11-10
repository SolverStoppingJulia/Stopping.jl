##############################################################################
#
# Test problem
#
#############################################################################

function rosenbrock(x)
  n = 6

  # Initializations
  f = 0

  evenIdx = 2:2:n
  oddIdx = 1:2:(n - 1)

  f1 = x[evenIdx] .- x[oddIdx] .^ 2
  f2 = 1 .- x[oddIdx]

  # Function
  f = sum(f1 .^ 2 .+ f2 .^ 2)

  return f
end
