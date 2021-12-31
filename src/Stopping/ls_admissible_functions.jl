export armijo, wolfe, armijo_wolfe, shamanskii_stop, goldstein

"""
    `armijo(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.01), kwargs...) where {S, T}`

Check if a step size is admissible according to the Armijo criterion.

Armijo criterion: `f(x + θd) - f(x) - τ₀ θ ∇f(x+θd)d < 0`

This function returns the maximum between the left-hand side and 0.

Note: `fx`, `f₀` and `g₀` are required in the `OneDAtX`.

See also `wolfe`, `armijo_wolfe`, `shamanskii_stop`, `goldstein`
"""
function armijo(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.01), kwargs...) where {S, T}
  if isnan(h_at_t.fx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
    return throw(error("fx, f₀ and g₀ are mandatory in the state."))
  else
    fact = -T(0.8)
    Eps = T(1e-10)
    hgoal = h_at_t.fx - h_at_t.f₀ - h_at_t.g₀ * h_at_t.x * τ₀
    # Armijo = (h_at_t.fx <= hgoal)# || ((h_at_t.fx <= h_at_t.f₀ + Eps * abs(h_at_t.f₀)) & (h_at_t.gx <= fact * h_at_t.g₀))
    # positive = h_at_t.x > 0.0   # positive step
    return max(hgoal, zero(T))
  end
end

"""
    `wolfe(h::Any, h_at_t::OneDAtX{S, T}; τ₁::T = T(0.99), kwargs...) where {S, T}`

Check if a step size is admissible according to the Wolfe criterion.

Strong Wolfe criterion: `|∇f(x+θd)| - τ₁||∇f(x)|| < 0`.

This function returns the maximum between the left-hand side and 0.

Note: `gx` and `g₀` are required in the `OneDAtX`.

See also `armijo`, `armijo_wolfe`, `shamanskii_stop`, `goldstein`
"""
function wolfe(h::Any, h_at_t::OneDAtX{S, T}; τ₁::T = T(0.99), kwargs...) where {S, T}
  if isnan(h_at_t.g₀) || isnan(h_at_t.gx)
    return throw(error("gx and g₀ are mandatory in the state."))
  else
    wolfe = abs(h_at_t.gx) - τ₁ * abs(h_at_t.g₀)
    #positive = h_at_t.x > 0.0   # positive step
    return max(wolfe, zero(T))
  end
end

"""
    `armijo_wolfe(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.01), τ₁::T = T(0.99), kwargs...) where {S, T}`

Check if a step size is admissible according to the Armijo and Wolfe criteria.

Note: `fx`, `f₀`, `gx` and `g₀` are required in the `OneDAtX`.

See also `armijo`, `wolfe`, `shamanskii_stop`, `goldstein`
"""
function armijo_wolfe(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.01), τ₁::T = T(0.99), kwargs...) where {S, T}
  if isnan(h_at_t.fx) || isnan(h_at_t.gx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
    return throw(error("fx, f₀, gx and g₀ are mandatory."))
  else
    wolfe = abs(h_at_t.gx) - τ₁ * abs(h_at_t.g₀)
    armijo = h_at_t.fx - h_at_t.f₀ - h_at_t.g₀ * h_at_t.x * τ₀
    return max(armijo, wolfe, zero(T))
  end
end

"""
    `shamanskii_stop(h :: Any, h_at_t :: OneDAtX; γ :: Float64 = 1.0e-09, kwargs...)`

Check if a step size is admissible according to the "Shamanskii" criteria.

This criteria was proposed in:
> Lampariello, F., & Sciandrone, M. (2001).
> Global convergence technique for the Newton method with periodic Hessian evaluation.
> Journal of optimization theory and applications, 111(2), 341-358.

Note: 
- `h.d` accessible (specific `LineModel`).
- `fx`, `f₀` are required in the `OneDAtX`.

See also `armijo`, `wolfe`, `armijo_wolfe`, `goldstein`
"""
function shamanskii_stop(h::Any, h_at_t::OneDAtX{S, T}; γ::T = T(1.0e-09), kwargs...) where {S, T}
  admissible = h_at_t.fx - h_at_t.f₀ - γ * (h_at_t.x)^3 * norm(h.d)^3
  return max(admissible, zero(T))
end

"""
    `goldstein(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.0001), τ₁::T = T(0.9999), kwargs...) where {S, T}`

Check if a step size is admissible according to the Goldstein criteria.

Note: `fx`, `f₀` and `g₀` are required in the `OneDAtX`.

See also `armijo`, `wolfe`, `armijo_wolfe`, `shamanskii_stop`
"""
function goldstein(h::Any, h_at_t::OneDAtX{S, T}; τ₀::T = T(0.0001), τ₁::T = T(0.9999), kwargs...) where {S, T}
  if isnan(h_at_t.fx) || isnan(h_at_t.gx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
    return throw(error("fx, f₀, gx and g₀ are mandatory."))
  else
    goldstein = max(
      h_at_t.f₀ + h_at_t.x * (1 - τ₀) * h_at_t.g₀ - h_at_t.fx,
      h_at_t.fx - (h_at_t.f₀ + h_at_t.x * τ₀ * h_at_t.g₀),
    )
    # positive = h_at_t.x > 0.0   # positive step
    return max(goldstein, zero(T)) #&& positive
  end
end
