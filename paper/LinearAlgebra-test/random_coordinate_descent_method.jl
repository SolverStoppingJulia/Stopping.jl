using LinearAlgebra, Krylov, LinearOperators, SparseArrays, Stopping

#Krylov @kdot
macro kdot(n, x, y)
  return esc(:(Krylov.krylov_dot($n, $x, 1, $y, 1)))
end

using SolverTools, Logging
"""
Randomized coordinate descent

Sect. 3.7 in Gower, R. M., & Richtárik, P. (2015).
Randomized iterative methods for linear systems.
SIAM Journal on Matrix Analysis and Applications, 36(4), 1660-1690.
"""
function RandomizedCD(
  A::AbstractMatrix,
  b::AbstractVector{T};
  is_zero_start::Bool = true,
  x0::AbstractVector{T} = zeros(T, size(A, 2)),
  atol::AbstractFloat = 1e-7,
  rtol::AbstractFloat = 1e-15,
  max_iter::Int = size(A, 2)^2,
  max_time::Float64 = 60.0,
  max_cntrs = Stopping._init_max_counters_linear_operators(quick = 20000),
  verbose::Int = 100,
  kwargs...,
) where {T <: AbstractFloat}
  m, n = size(A)
  x = copy(x0)
  res = is_zero_start ? b : b - A * x
  nrm0 = norm(res)

  time_init = time()
  elapsed_time = time_init

  cntrs = LACounters()
  max_f = false

  OK = nrm0 <= atol
  k = 0

  #@info log_header([:iter, :un, :time], [Int, T, T])
  #@info log_row(Any[0, res[1], elapsed_time])
  while !OK && (k <= max_iter) && (elapsed_time - time_init <= max_time) && !max_f

    #rand a number between 1 and n
    #224.662 ns (4 allocations: 79 bytes) - independent of the n
    i = mod(k, n) + 1#Int(floor(rand() * n) + 1)
    Ai = A[:, i]

    #ei = zeros(n); ei[i] = 1.0 #unit vector in R^n
    #xk  = Ai == 0 ? x0 : x0 - dot(Ai,res)/norm(Ai,2)^2 * ei
    Aires = @kdot(m, Ai, res)
    nAi = @kdot(m, Ai, Ai)
    x[i] -= Aires / nAi

    #res = b - A*x
    res += Ai * Aires / nAi
    cntrs.nprod += 1
    nrm = norm(res, Inf)
    OK = nrm <= atol + nrm0 * rtol

    sum, max_f = 0, false
    for f in [:nprod, :ntprod, :nctprod]
      ff = getfield(cntrs, f)
      max_f = max_f || (ff > max_cntrs[f])
      sum += ff
    end
    max_f = max_f || sum > max_cntrs[:neval_sum]
    k += 1
    elapsed_time = time()
    if mod(k, verbose) == 0 #print every 20 iterations
      # @info log_row(Any[k, res[1], elapsed_time])
    end
  end

  return x, OK, k
end
