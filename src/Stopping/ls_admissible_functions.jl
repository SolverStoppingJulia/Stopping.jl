export armijo, wolfe, armijo_wolfe, shamanskii_stop, goldstein

"""
armijo: check if a step size is admissible according to the Armijo criterion.

Armijo criterion: f(x + θd) - f(x) - τ₀ θ ∇f(x+θd)d < 0

`armijo(h :: Any, h_at_t :: OneDAtX; τ₀ :: Float64 = 0.01, kwargs...)`

Note: fx, f₀ and g₀ are required in the OneDAtX

See also `wolfe`, `armijo_wolfe`, `shamanskii_stop`, `goldstein`
"""
function armijo(h      :: Any,
                h_at_t :: OneDAtX;
                τ₀     :: Float64 = 0.01,
                kwargs...)

    if isnan(h_at_t.fx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
     return throw(error("Nothing entries in the State. fx, f₀ and g₀ are mandatory."))
    else
     fact = -0.8
     Eps = 1e-10
     hgoal = h_at_t.fx - h_at_t.f₀ - h_at_t.g₀ * h_at_t.x * τ₀
     # Armijo = (h_at_t.fx <= hgoal)# || ((h_at_t.fx <= h_at_t.f₀ + Eps * abs(h_at_t.f₀)) & (h_at_t.gx <= fact * h_at_t.g₀))
	 # Armijo_HZ =
	 # positive = h_at_t.x > 0.0   # positive step
     return max(hgoal, 0.0)
    end
end

"""
wolfe: check if a step size is admissible according to the Wolfe criterion.

Strong Wolfe criterion: |∇f(x+θd)| < τ₁||∇f(x)||.

`wolfe(h :: Any, h_at_t :: OneDAtX; τ₁ :: Float64 = 0.99, kwargs...)`

Note: gx and g₀ are required in the OneDAtX

See also `armijo`, `armijo_wolfe`, `shamanskii_stop`, `goldstein`
"""
function wolfe(h      :: Any,
               h_at_t :: OneDAtX;
               τ₁     :: Float64 = 0.99,
               kwargs...)

    if isnan(h_at_t.g₀) || isnan(h_at_t.gx)
     return throw(error("Nothing entries in the State."))
    else

     #wolfe = (τ₁ .* h_at_t.g₀) - (abs(h_at_t.gx))
     wolfe = abs(h_at_t.gx) - τ₁ * abs(h_at_t.g₀)
	 #positive = h_at_t.x > 0.0   # positive step
     return max(wolfe, 0.0)
    end
end

"""
armijo_wolfe: check if a step size is admissible according to the Armijo and Wolfe criteria.

`armijo_wolfe(h :: Any, h_at_t :: OneDAtX; τ₀ :: Float64 = 0.01, τ₁ :: Float64 = 0.99, kwargs...)`

Note: fx, f₀, gx and g₀ are required in the OneDAtX

See also `armijo`, `wolfe`, `shamanskii_stop`, `goldstein`
"""
function armijo_wolfe(h      :: Any,
                      h_at_t :: OneDAtX;
                      τ₀     :: Float64 = 0.01,
                      τ₁     :: Float64 = 0.99,
                      kwargs...)

   if isnan(h_at_t.fx) || isnan(h_at_t.gx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
    return throw(error("Nothing entries in the State. fx, f₀, gx and g₀ are mandatory."))
   else
    wolfe  = abs(h_at_t.gx) - τ₁ * abs(h_at_t.g₀)
    armijo = h_at_t.fx - h_at_t.f₀ - h_at_t.g₀ * h_at_t.x * τ₀
    return max(armijo, wolfe, 0.0)
   end
end

"""
shamanskii_stop: check if a step size is admissible according to the "Shamanskii" criteria.
This criteria was proposed in:
Lampariello, F., & Sciandrone, M. (2001).
Global convergence technique for the Newton method with periodic Hessian evaluation.
Journal of optimization theory and applications, 111(2), 341-358.

`shamanskii_stop(h :: Any, h_at_t :: OneDAtX; γ :: Float64 = 1.0e-09, kwargs...)`

Note: 
- h.d accessible (specific LineModel)
- fx, f₀ are required in the OneDAtX

See also `armijo`, `wolfe`, `armijo_wolfe`, `goldstein`
"""
function shamanskii_stop(h      :: Any,
                         h_at_t :: OneDAtX;
                         γ      :: Float64 = 1.0e-09,
                         kwargs...)

    admissible = h_at_t.fx - h_at_t.f₀ - γ * (h_at_t.x)^3 * norm(h.d)^3
    return max(admissible, 0.0)
end

"""
goldstein: check if a step size is admissible according to the Goldstein criteria.

`goldstein(h :: Any, h_at_t :: OneDAtX; τ₀ :: Float64 = 0.0001, τ₁ :: Float64 = 0.9999, kwargs...)`

Note: fx, f₀ and g₀ are required in the OneDAtX

See also `armijo`, `wolfe`, `armijo_wolfe`, `shamanskii_stop`
"""
function goldstein(h      :: Any,
                   h_at_t :: OneDAtX;
                   τ₀     :: Float64 = 0.0001,
                   τ₁     :: Float64 = 0.9999,
                   kwargs...)
  if isnan(h_at_t.fx) || isnan(h_at_t.gx) || isnan(h_at_t.f₀) || isnan(h_at_t.g₀)
    return throw(error("Nothing entries in the State. fx, f₀, gx and g₀ are mandatory."))
  else
 	goldstein = max(h_at_t.f₀ + h_at_t.x * (1 - τ₀) * h_at_t.g₀ - h_at_t.fx,
                    h_at_t.fx - (h_at_t.f₀ + h_at_t.x *  τ₀ * h_at_t.g₀))
 	# positive = h_at_t.x > 0.0   # positive step
 	return max(goldstein, 0.0) #&& positive
  end
end
