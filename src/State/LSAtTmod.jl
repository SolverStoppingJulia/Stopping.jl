import Base.copy

"""
A structure designed to track line search information from one iteration to
another. If we have f : ℜⁿ → ℜ, then we define h(θ) = f(x + θ*d) where x and d
are vectors of same dimension and θ is a scalar, more specifically our step size.

Tracked data can include:
 - x : our current step size
 - ht : h(θ) at the current iteration
 - gt : h'(θ)
 - h₀ : h(0)
 - g₀ : h'(0)
 - current_time: the time at which the line search algorithm started.

Unless they are defined otherwise, the default value for all parameter is NaN
(except for x). They can be updated through the update! function.

Example:
```
ls_a_t = LSAtT(1.0)
update!(ls_a_t, x = 0.0, h₀ = obj(h, 0.0), g₀ = grad(h, 0.0))
```
"""
mutable struct 	LSAtT <: AbstractState

    x            :: Number
    ht           :: FloatVoid  # h(θ)
    gt           :: FloatVoid  # h'(θ)
    h₀           :: FloatVoid  # h(0)
    g₀           :: FloatVoid  # h'(0)

    current_time   :: FloatVoid

 function LSAtT(t            :: Number;
                ht           :: FloatVoid = nothing,
                gt           :: FloatVoid = nothing,
                h₀           :: FloatVoid = nothing,
                g₀           :: FloatVoid = nothing,
                current_time :: FloatVoid = nothing)

  return new(t, ht, gt, h₀, g₀, current_time)
 end
end

"""
Copy a LSAtT
useful?
"""
function copy(ls_at_t :: LSAtT)
    return LSAtT(copy(ls_at_t.x),
                 ht = copy(ls_at_t.ht),
                 gt = copy(ls_at_t.gt),
                 h₀ = copy(ls_at_t.h₀),
                 g₀ = copy(ls_at_t.g₀),
                 current_time = copy(ls_at_t.current_time))
end
