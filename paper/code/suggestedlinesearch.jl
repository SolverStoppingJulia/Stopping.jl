################################################################################
#VERSION 1
#linesearch is an intermediary between NewtonStopLS and a 1D solver.
function preparelinesearch(
  stp::NLPStopping,
  x₀::AbstractVector,
  d::AbstractVector,
  f₀::AbstractFloat,
  ∇f₀::AbstractVector,
  τ₀::Real,
  τ₁::Real,
)

  # initialize the LSModel
  g₀ = dot(∇f₀, d)
  ϕ = LSModel(stp.pb, x₀, d, τ₀, f₀, g₀)

  # convert the Armijo and Wolfe criteria to an asymetric interval [α,β]
  α = (τ₁ - τ₀) * g₀
  β = -(τ₁ + τ₀) * g₀

  # reuse the stopping
  ϕstp = LS_Stopping(
    ϕ,
    LSAtT(0.0, ht = f₀, gt = ∇f₀),
    optimality_check = (p, s) -> optim_check_LS(p, s, α, β),
    main_stp = stp,
    max_iter = 40,
    atol = 0.0, # to rely only on the Armijo-Wolfe conditions
    rtol = 0.0, # otherwise, tolerance may prohibit convergence
    unbounded_threshold = 1e100,
  )

  # optimize in the interval [0.0,Inf]
  ϕstp = min_1D(ϕ, 0.0, Inf, α, β, stp = ϕstp)

  # unpack the results
  if ϕstp.meta.optimal
    t, ft, gt = ϕstp.current_state.x, ϕstp.current_state.ht, ϕstp.current_state.gt
  else
    t, ft, gt = 0.0, f₀, ∇f₀
    stp.meta.fail_sub_pb = true
  end

  return t, ft, gt
end

function optim_check_LS(p, s::LSAtT, α::Float64, β::Float64)
  return max(α - s.gt, s.gt - β, 0.0)
end

################################################################################
#VERSION 2
#Not an equivalent way would be with tol_check:
function preparelinesearch(
  stp::NLPStopping,
  x₀::AbstractVector,
  d::AbstractVector,
  f₀::AbstractFloat,
  ∇f₀::AbstractVector,
  τ₀::Real,
  τ₁::Real,
)

  # initialize the LSModel
  g₀ = dot(∇f₀, d)
  ϕ = LSModel(stp.pb, x₀, d, τ₀, f₀, g₀)
  #Instead of redefining LSModel we can use the known ADNLPModel ? (not optimal though)
  #ϕ = ADNLPModel(t -> (obj(stp.pb, x₀ + t*d) - f₀ - τ₀ * t * g₀), [0.0], lvar = [0.0], uvar = [Inf])

  # convert the Armijo and Wolfe criteria to an asymetric interval [α,β]
  α = (τ₁ - τ₀) * g₀
  β = -(τ₁ + τ₀) * g₀

  # reuse the stopping
  ϕstp = NLPStopping(
    ϕ,
    NLPAtX([0.0], fx = f₀, gx = ∇f₀),
    optimality_check = unconstrained_check,
    main_stp = stp,
    max_iter = 40,
    unbounded_threshold = 1e100,
    tol_check = (a, b, c) -> β,
    tol_check_neg = (a, b, c) -> α,
  )

  # optimize in the interval [0.0,Inf]
  ϕstp = min_1D(ϕ, 0.0, Inf, stp = ϕstp)

  # unpack the results
  if ϕstp.meta.optimal
    t, ft, gt = ϕstp.current_state.x, ϕstp.current_state.ht, ϕstp.current_state.gt
  else
    t, ft, gt = 0.0, f₀, ∇f₀
    stp.meta.fail_sub_pb = true
  end

  return t, ft, gt
end
