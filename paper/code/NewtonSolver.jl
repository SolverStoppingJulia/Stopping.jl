function Newton_Spectral(
  nlp::AbstractNLPModel,
  x₀::AbstractVector;
  τ₀::Float64 = 0.0005,
  ϵ::Float64 = 1e-6,
  maxiter::Int = 200,
)
  x = copy(x₀)
  iter = 0
  f, g = obj(nlp, x), grad(nlp, x)

  while (norm(g, Inf) > ϵ) && (iter <= maxiter)
    H = Matrix(Symmetric(hess(nlp, x), :L))
    Δ, O = eigen(H)

    # Boost negative values of Δ to 1e-8
    D = Δ .+ max.((1e-8 .- Δ), 0.0)

    d = -O * diagm(1.0 ./ D) * O' * g

    # Simple Armijo backtracking
    hp0 = g' * d
    t = 1.0
    ft = obj(nlp, x + t * d)
    while ft > (f + τ₀ * t * hp0)
      t /= 2.0
      ft = obj(nlp, x + t * d)
    end
    x += t * d
    f, g = ft, grad(nlp, x)
    iter += 1
  end
  if iter > maxiter
    @warn "Iteration limit"
  end

  return x, f, norm(g), iter
end
