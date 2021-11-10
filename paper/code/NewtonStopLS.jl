function Newton_StopLS(
  nlp::AbstractNLPModel,
  x₀::AbstractVector;
  τ₀::Float64 = 0.0005,
  τ₁::Float64 = 0.99,
  stp::NLPStopping = NLPStopping(nlp, NLPAtX(x₀)),
  optimality_check = unconstrained_check,
  atol = 1e-6,
  max_iter = 200,
)
  x = copy(x₀)
  f, g = obj(nlp, x), grad(nlp, x)

  OK = update_and_start!(stp, x = x, fx = f, gx = g)

  d = similar(x)
  ϕ, ϕstp = prepare_LS(stp, x, d, τ₀, f, g)

  while !OK
    Hx = hess(nlp, x)
    H = Matrix(Symmetric(Hx, :L))
    Δ, O = eigen(H)

    # Boost negative values of Δ to 1e-8
    D = Δ .+ max.((1e-8 .- Δ), 0.0)

    d = -O * diagm(1.0 ./ D) * O' * g

    # Simple line search call
    t, f, g = linesearch(ϕ, ϕstp, x, d, f, g, τ₀, τ₁)

    x += t * d
    OK = update_and_stop!(stp, x = x, gx = g, Hx = H)
  end

  if !stp.meta.optimal
    @warn "Optimality not reached"
    @info status(stp, list = true)
  end

  return x, f, norm(g), stp
end
