"""
Type: LSAtT

Methods: update!, reinit!, copy

A structure designed to track line search information from one iteration to
another. Given f : ℜⁿ → ℜ, define h(θ) = f(x + θ*d) where x and d
are vectors of same dimension and θ is a scalar, more specifically the step size.

Tracked data can include:
 - x             : the current step size
 - ht [opt]      : h(θ) at the current iteration
 - gt [opt]      : h'(θ)
 - h₀ [opt]      : h(0)
 - g₀ [opt]      : h'(0)
 - current_time  :  the time at which the line search algorithm started.
 - current_score : the score at which the line search algorithm started.

Constructors:
 `LSAtT(:: T, :: S; ht :: T = _init_field(T), gt :: T = _init_field(T), h₀ :: T = _init_field(T), g₀ :: T = _init_field(T), current_time :: Float64 = NaN) where {S, T <: Number}`

 `LSAtT(:: T; ht :: T = _init_field(T), gt :: T = _init_field(T), h₀ :: T = _init_field(T), g₀ :: T = _init_field(T), current_time :: Float64 = NaN, current_score :: T = _init_field(T))  where T <: Number`

Note: 
 - By default, unknown entries are set using `_init_field`.  
 - By default the type of `current_score` is `eltype(x)` and cannot be changed once the State is created. 
   To have a vectorized `current_score` of length n, try something like `GenericState(x, Array{eltype(x),1}(undef, n))`.
"""
mutable struct 	LSAtT{S, T <: Number} <: AbstractState{S, T}

  x            :: T
  ht           :: T  # h(θ)
  gt           :: T  # h'(θ)
  h₀           :: T  # h(0)
  g₀           :: T  # h'(0)

  current_time   :: Float64
  current_score  :: S

  function LSAtT(t             :: T,
                 current_score :: S;
                 ht            :: T = _init_field(T),
                 gt            :: T = _init_field(T),
                 h₀            :: T = _init_field(T),
                 g₀            :: T = _init_field(T),
                 current_time  :: Float64 = NaN) where {S, T <: Number}

    return new{S, T}(t, ht, gt, h₀, g₀, current_time, current_score)
  end
end

function LSAtT(t             :: T;
               ht            :: T = _init_field(T),
               gt            :: T = _init_field(T),
               h₀            :: T = _init_field(T),
               g₀            :: T = _init_field(T),
               current_time  :: Float64 = NaN,
               current_score :: T = _init_field(T))  where T <: Number

  return LSAtT(t, current_score, ht = ht, gt = gt, h₀ = h₀, g₀ = g₀, current_time = current_time)
end
