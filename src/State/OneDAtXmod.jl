"""
Type: OneDAtX

Methods: update!, reinit!, copy

A structure designed to track line search information from one iteration to
another. Given f : ℜⁿ → ℜ, define h(θ) = f(x + θ*d) where x and d
are vectors of same dimension and θ is a scalar, more specifically the step size.

Tracked data can include:
 - x             : the current step size
 - fx [opt]      : h(θ) at the current iteration
 - gx [opt]      : h'(θ)
 - f₀ [opt]      : h(0)
 - g₀ [opt]      : h'(0)
 - d [opt]       : search direction
 - res [opt]     : residual
 - current_time  : the time at which the line search algorithm started.
 - current_score : the score at which the line search algorithm started.

Constructors:
 `OneDAtX(:: T, :: S; fx :: T = _init_field(T), gx :: T = _init_field(T), f₀ :: T = _init_field(T), g₀ :: T = _init_field(T), current_time :: Float64 = NaN) where {S, T <: Number}`

 `OneDAtX(:: T; fx :: T = _init_field(T), gx :: T = _init_field(T), f₀ :: T = _init_field(T), g₀ :: T = _init_field(T), current_time :: Float64 = NaN, current_score :: T = _init_field(T))  where T <: Number`

Note: 
 - By default, unknown entries are set using `_init_field`.  
 - By default the type of `current_score` is `eltype(x)` and cannot be changed once the State is created. 
   To have a vectorized `current_score` of length n, use `OneDAtX(x, Array{eltype(x),1}(undef, n))`.
"""
mutable struct OneDAtX{S, T <: Number} <: AbstractState{S, T}
  x::T
  fx::T  # h(θ)
  gx::T  # h'(θ)
  f₀::T  # h(0)
  g₀::T  # h'(0)

  d::T
  res::T

  current_time::Float64
  current_score::S

  function OneDAtX(
    t::T,
    current_score::S;
    fx::T = _init_field(T),
    gx::T = _init_field(T),
    f₀::T = _init_field(T),
    g₀::T = _init_field(T),
    d::T = _init_field(T),
    res::T = _init_field(T),
    current_time::Float64 = NaN,
  ) where {S, T <: Number}
    return new{S, T}(t, fx, gx, f₀, g₀, d, res, current_time, current_score)
  end
end

function OneDAtX(
  t::T;
  fx::T = _init_field(T),
  gx::T = _init_field(T),
  f₀::T = _init_field(T),
  g₀::T = _init_field(T),
  d::T = _init_field(T),
  res::T = _init_field(T),
  current_time::Float64 = NaN,
  current_score::T = _init_field(T),
) where {T <: Number}
  return OneDAtX(
    t,
    current_score,
    fx = fx,
    gx = gx,
    f₀ = f₀,
    g₀ = g₀,
    d = d,
    res = res,
    current_time = current_time,
  )
end

for field in fieldnames(OneDAtX)
  meth = Symbol("get_", field)
  @eval begin
    @doc """
        $($meth)(state)
    Return the value $($(QuoteNode(field))) from the state.
    """
    $meth(state::OneDAtX) = getproperty(state, $(QuoteNode(field)))
  end
  @eval export $meth
end

function set_current_score!(state::OneDAtX{S, T}, current_score::S) where {S, T}
  state.current_score .= current_score
  return state
end

function set_current_score!(state::OneDAtX{S, T}, current_score::S) where {S <: Number, T}
  state.current_score = current_score
  return state
end

function set_x!(state::OneDAtX{S, T}, x::T) where {S, T}
  if length(state.x) == length(x)
    state.x .= x
  else
    state.x = x
  end
  return state
end

function set_d!(state::OneDAtX{S, T}, d::T) where {S, T}
  if length(state.d) == length(d)
    state.d .= d
  else
    state.d = d
  end
  return state
end

function set_res!(state::OneDAtX{S, T}, res::T) where {S, T}
  if length(state.res) == length(res)
    state.res .= res
  else
    state.res = res
  end
  return state
end
