"""
Type: NLPAtX

Methods: update!, reinit!

NLPAtX contains the information concerning a nonlinear optimization model at
the iterate x.

min_{x ∈ ℜⁿ} f(x) subject to lcon <= c(x) <= ucon, lvar <= x <= uvar.

Tracked data include:
 - x             : the current iterate
 - fx [opt]      : function evaluation at x
 - gx [opt]      : gradient evaluation at x
 - Hx [opt]      : hessian evaluation at x

 - mu [opt]      : Lagrange multiplier of the bounds constraints

 - cx [opt]      : evaluation of the constraint function at x
 - Jx [opt]      : jacobian matrix of the constraint function at x
 - lambda        : Lagrange multiplier of the constraints

 - d [opt]       : search direction
 - res [opt]     : residual

 - current_time  : time
 - current_score : score
 (import the type NLPModels.Counters)

Constructors:
 `NLPAtX(:: T, :: T, :: S; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), cx :: T = _init_field(T), Jx :: SparseMatrixCSC{eltype(T), Int64} = _init_field(SparseMatrixCSC{eltype(T), Int64}), d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64 = NaN) where {S, T <: AbstractVector}`

 `NLPAtX(:: T; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), current_time :: Float64 = NaN, current_score :: Union{T,eltype(T)} = _init_field(eltype(T))) where {T <: AbstractVector}`

 `NLPAtX(:: T, :: T; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), cx :: T = _init_field(T), Jx :: SparseMatrixCSC{eltype(T), Int64} = _init_field(SparseMatrixCSC{eltype(T), Int64}), d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64  = NaN, current_score :: Union{T,eltype(T)} = _init_field(eltype(T))) where T <: AbstractVector`

Note:
 - By default, unknown entries are set using `_init_field`.  
 - By default the type of `current_score` is `eltype(x)` and cannot be changed once the State is created.  
    To have a vectorized `current_score` of length n, try something like `GenericState(x, Array{eltype(x),1}(undef, n))`.  
 - All these information (except for `x` and `lambda`) are optionnal and need to be update when
    required. The update is done through the `update!` function.  
 - `x` and `lambda` are mandatory entries. If no constraints `lambda = []`.  
 - The constructor check the size of the entries.  

See also: `GenericState`, `update!`, `update_and_start!`, `update_and_stop!`, `reinit!`
"""
mutable struct NLPAtX{Score, S, T <: AbstractVector} <: AbstractState{S, T}

  #Unconstrained State
  x::T     # current point
  fx::S # objective function
  gx::T  # gradient size: x
  Hx  # hessian size: |x| x |x|

  #Bounds State
  mu::T # Lagrange multipliers with bounds size of |x|

  #Constrained State
  cx::T # vector of constraints lc <= c(x) <= uc
  Jx  # jacobian matrix, size: |lambda| x |x|
  lambda::T    # Lagrange multipliers

  d::T #search direction
  res::T #residual

  #Resources State
  current_time::Float64
  current_score::Score

  function NLPAtX(
    x::T,
    lambda::T,
    current_score::Score;
    fx::eltype(T) = _init_field(eltype(T)),
    gx::T = _init_field(T),
    Hx = _init_field(Matrix{eltype(T)}),
    mu::T = _init_field(T),
    cx::T = _init_field(T),
    Jx = _init_field(SparseMatrixCSC{eltype(T), Int64}),
    d::T = _init_field(T),
    res::T = _init_field(T),
    current_time::Float64 = NaN,
  ) where {Score, S, T <: AbstractVector}
    _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

    return new{Score, eltype(T), T}(
      x,
      fx,
      gx,
      Hx,
      mu,
      cx,
      Jx,
      lambda,
      d,
      res,
      current_time,
      current_score,
    )
  end
end

function NLPAtX(
  x::T,
  lambda::T;
  fx::eltype(T) = _init_field(eltype(T)),
  gx::T = _init_field(T),
  Hx = _init_field(Matrix{eltype(T)}),
  mu::T = _init_field(T),
  cx::T = _init_field(T),
  Jx = _init_field(SparseMatrixCSC{eltype(T), Int64}),
  d::T = _init_field(T),
  res::T = _init_field(T),
  current_time::Float64 = NaN,
  current_score::Union{T, eltype(T)} = _init_field(eltype(T)),
) where {T <: AbstractVector}
  _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

  return NLPAtX(
    x,
    lambda,
    current_score,
    fx = fx,
    gx = gx,
    Hx = Hx,
    mu = mu,
    cx = cx,
    Jx = Jx,
    d = d,
    res = res,
    current_time = current_time,
  )
end

function NLPAtX(
  x::T;
  fx::eltype(T) = _init_field(eltype(T)),
  gx::T = _init_field(T),
  Hx = _init_field(Matrix{eltype(T)}),
  mu::T = _init_field(T),
  d::T = _init_field(T),
  res::T = _init_field(T),
  current_time::Float64 = NaN,
  current_score::Union{T, eltype(T)} = _init_field(eltype(T)),
) where {T <: AbstractVector}
  _size_check(
    x,
    zeros(eltype(T), 0),
    fx,
    gx,
    Hx,
    mu,
    _init_field(T),
    _init_field(SparseMatrixCSC{eltype(T), Int64}),
  )

  return NLPAtX(
    x,
    zeros(eltype(T), 0),
    current_score,
    fx = fx,
    gx = gx,
    Hx = Hx,
    mu = mu,
    d = d,
    res = res,
    current_time = current_time,
  )
end

for field in fieldnames(NLPAtX)
  meth = Symbol("get_", field)
  @eval begin
    @doc """
        $($meth)(state)
    Return the value $($(QuoteNode(field))) from the state.
    """
    $meth(state::NLPAtX) = getproperty(state, $(QuoteNode(field)))
  end
  @eval export $meth
end

function set_current_score!(state::NLPAtX{Score, S, T}, current_score::Score) where {Score, S, T}
  if length(state.current_score) == length(current_score)
    state.current_score .= current_score
  else
    state.current_score = current_score
  end
  return state
end

function Stopping.set_current_score!(
  state::NLPAtX{Score, S, T},
  current_score::Score,
) where {Score <: Number, S, T}
  state.current_score = current_score
  return state
end

function set_x!(state::NLPAtX{Score, S, T}, x::T) where {Score, S, T}
  if length(state.x) == length(x)
    state.x .= x
  else
    state.x = x
  end
  return state
end

function set_d!(state::NLPAtX{Score, S, T}, d::T) where {Score, S, T}
  if length(state.d) == length(d)
    state.d .= d
  else
    state.d = d
  end
  return state
end

function set_res!(state::NLPAtX{Score, S, T}, res::T) where {Score, S, T}
  if length(state.res) == length(res)
    state.res .= res
  else
    state.res = res
  end
  return state
end

function set_lambda!(state::NLPAtX{Score, S, T}, lambda::T) where {Score, S, T}
  if length(state.lambda) == length(lambda)
    state.lambda .= lambda
  else
    state.lambda = lambda
  end
  return state
end

function set_mu!(state::NLPAtX{Score, S, T}, mu::T) where {Score, S, T}
  if length(state.mu) == length(mu)
    state.mu .= mu
  else
    state.mu = mu
  end
  return state
end

function set_fx!(state::NLPAtX{Score, S, T}, fx::S) where {Score, S, T}
  state.fx = fx
  return state
end

function set_gx!(state::NLPAtX{Score, S, T}, gx::T) where {Score, S, T}
  if length(state.gx) == length(gx)
    state.gx .= gx
  else
    state.gx = gx
  end
  return state
end

function set_cx!(state::NLPAtX{Score, S, T}, cx::T) where {Score, S, T}
  if length(state.cx) == length(cx)
    state.cx .= cx
  else
    state.cx = cx
  end
  return state
end

function Stopping._domain_check(
  stateatx::NLPAtX{Score, S, T};
  current_score = false,
  x = false,
) where {Score, S, T}
  if !x && Stopping._check_nan_miss(get_x(stateatx))
    return true
  end
  if !current_score && Stopping._check_nan_miss(get_current_score(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_d(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_res(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_fx(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_gx(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_mu(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_cx(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_lambda(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_Jx(stateatx))
    return true
  end
  if Stopping._check_nan_miss(get_Hx(stateatx))
    return true
  end
  return false
end

"""
reinit!: function that set all the entries at void except the mandatory x

`reinit!(:: NLPAtX, x :: AbstractVector, l :: AbstractVector; kwargs...)`

`reinit!(:: NLPAtX; kwargs...)`

Note: if `x` or `lambda` are given as keyword arguments they will be
prioritized over the existing `x`, `lambda` and the default `Counters`.
"""
function reinit!(stateatx::NLPAtX{Score, S, T}, x::T, l::T; kwargs...) where {Score, S, T}
  for k ∈ fieldnames(NLPAtX)
    if k ∉ [:x, :lambda]
      setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
    end
  end

  setfield!(stateatx, :x, x)
  setfield!(stateatx, :lambda, l)

  if length(kwargs) == 0
    return stateatx #save the update! call if no other kwargs than x
  end

  return update!(stateatx; kwargs...)
end

function reinit!(stateatx::NLPAtX{Score, S, T}, x::T; kwargs...) where {Score, S, T}
  for k ∈ fieldnames(NLPAtX)
    if k ∉ [:x, :lambda]
      setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
    end
  end

  setfield!(stateatx, :x, x)

  if length(kwargs) == 0
    return stateatx #save the update! call if no other kwargs than x
  end

  return update!(stateatx; kwargs...)
end

function reinit!(stateatx::NLPAtX; kwargs...)
  for k ∈ fieldnames(NLPAtX)
    if k ∉ [:x, :lambda]
      setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
    end
  end

  return update!(stateatx; kwargs...)
end

"""
_size_check!: check the size of the entries in the State

`_size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)`
"""
function _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)
  if length(gx) != 0 && length(gx) != length(x)
    throw(error("Wrong size of gx in the NLPAtX."))
  end
  if size(Hx) != (0, 0) && size(Hx) != (length(x), length(x))
    throw(error("Wrong size of Hx in the NLPAtX."))
  end
  if length(mu) != 0 && length(mu) != length(x)
    throw(error("Wrong size of mu in the NLPAtX."))
  end

  if length(lambda) != 0
    if length(cx) != 0 && length(cx) != length(lambda)
      throw(error("Wrong size of cx in the NLPAtX."))
    end
    if size(Jx) != (0, 0) && size(Jx) != (length(lambda), length(x))
      throw(error("Wrong size of Jx in the NLPAtX."))
    end
  end
end
