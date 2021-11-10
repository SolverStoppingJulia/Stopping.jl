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

Using Stopping
"""
function StopRandomizedCD(
  A::AbstractMatrix,
  b::AbstractVector{T};
  is_zero_start::Bool = true,
  x0::AbstractVector{T} = zeros(T, size(A, 2)),
  atol::AbstractFloat = 1e-7,
  rtol::AbstractFloat = 1e-15,
  max_iter::Int = size(A, 2)^2,
  verbose::Int = 100,
  kwargs...,
) where {T <: AbstractFloat}
  stp = LAStopping(
    LinearSystem(A, b),
    GenericState(x0, similar(b)),
    max_cntrs = Stopping._init_max_counters_linear_operators(),
    atol = atol,
    rtol = rtol,
    max_iter = max_iter,
    tol_check = (atol, rtol, opt0) -> (atol + rtol * opt0),
    retol = false,
    optimality_check = (pb::LinearSystem, state::GenericState{Vector{T}, Vector{T}}) -> state.res,
    kwargs...,
  )

  return StopRandomizedCD(stp, is_zero_start = is_zero_start, verbose = verbose, kwargs...)
end

function StopRandomizedCD(
  stp::AbstractStopping;
  is_zero_start::Bool = true,
  verbose::Int = 100,
  kwargs...,
)
  state = stp.current_state

  A, b = stp.pb.A, stp.pb.b
  m, n = size(A)
  x = state.x
  T = eltype(x)

  res = is_zero_start ? b : b - A * x

  OK = update_and_start!(stp, res = res)

  #@info log_header([:iter, :nrm, :time], [Int, T, T],
  #                 hdr_override=Dict(:nrm=>"||Ax-b||"))
  #@info log_row(Any[0, state.current_score[1], state.current_time])

  while !OK

    #rand a number between 1 and n
    #224.662 ns (4 allocations: 79 bytes) - independent of the n
    i = mod(stp.meta.nb_of_stop, n) + 1#Int(floor(rand() * n) + 1)
    Ai = A[:, i]

    #ei = zeros(n); ei[i] = 1.0 #unit vector in R^n
    #xk  = Ai == 0 ? x0 : x0 - dot(Ai,res)/norm(Ai,2)^2 * ei
    Aires = @kdot(m, Ai, res)
    nAi = @kdot(m, Ai, Ai)
    x[i] -= Aires / nAi

    res += Ai * Aires / nAi

    OK = update_and_stop!(stp, x = x, res = res)

    if mod(stp.meta.nb_of_stop, verbose) == 0 #print every 20 iterations
      #     @info log_row(Any[stp.meta.nb_of_stop, state.current_score[1], state.current_time])
    end
  end

  return stp
end
