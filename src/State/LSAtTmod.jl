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
 - start_time: the time at which the line search algorithm started.

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
    ht           :: Number  # h(θ)
    gt           :: Number  # h'(θ)
    h₀           :: Number  # h(0)
    g₀           :: Number  # h'(0)

    start_time   :: Number

 function LSAtT(t          :: Number;
                ht         :: Number = NaN,
                gt         :: Number = NaN,
                h₀         :: Number = NaN,
                g₀         :: Number = NaN,
                start_time :: Number = NaN)

  return new(t, ht, gt, h₀, g₀, start_time)
 end
end

function update!(ls_at_t :: LSAtT;
                 x       :: FloatVoid = nothing,
                 ht      :: FloatVoid = nothing,
                 gt      :: FloatVoid = nothing,
                 h₀      :: FloatVoid = nothing,
                 g₀      :: FloatVoid = nothing,
                 tmps    :: FloatVoid = nothing,
                 kwargs...)

    if x != nothing
       ls_at_t.x  = x
    end

    if ht != nothing
       ls_at_t.ht = ht
    end

    ls_at_t.gt = gt == nothing ? ls_at_t.gt : gt
    ls_at_t.h₀ = h₀ == nothing ? ls_at_t.h₀ : h₀
    ls_at_t.g₀ = g₀ == nothing ? ls_at_t.g₀ : g₀

    ls_at_t.start_time = tmps == nothing ? ls_at_t.start_time : tmps

    return ls_at_t
end

function copy(ls_at_t :: LSAtT)
    return LSAtT(copy(ls_at_t.x),
                 ht = copy(ls_at_t.ht),
                 gt = copy(ls_at_t.gt),
                 h₀ = copy(ls_at_t.h₀),
                 g₀ = copy(ls_at_t.g₀),
                 start_time = copy(ls_at_t.start_time))
end

# function convert_ls(T, ls_at_t :: LSAtT)
#     ls_a_t_T = LSAtT(T.(copy(ls_at_t.x)))
#
#     ls_a_t_T.ht = typeof(ls_at_t.ht) != Nothing ? convert.(T, ls_at_t.ht) : ls_at_t.ht
#     ls_a_t_T.gt = typeof(ls_at_t.gt) != Nothing ? convert.(T, ls_at_t.gt) : ls_at_t.gt
#     ls_a_t_T.h₀ = typeof(ls_at_t.h₀) != Nothing ? convert.(T, ls_at_t.h₀) : ls_at_t.h₀
#     ls_a_t_T.g₀ = typeof(ls_at_t.g₀) != Nothing ? convert.(T, ls_at_t.g₀) : ls_at_t.g₀
#
#     return ls_a_t_T
# end
