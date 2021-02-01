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
 - evals [opt]   : number of evaluations of the function
 (import the type NLPModels.Counters)

Constructors:
 `NLPAtX(:: T, :: T, :: S; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), cx :: T = _init_field(T), Jx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64 = NaN, evals :: Counters = Counters()) where {S, T <: AbstractVector}`

 `NLPAtX(:: T; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), current_time :: Float64 = NaN, current_score :: Union{T,eltype(T)} = _init_field(eltype(T)), evals :: Counters  = Counters()) where {T <: AbstractVector}`

 `NLPAtX(:: T, :: T; fx :: eltype(T) = _init_field(eltype(T)), gx :: T = _init_field(T), Hx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), mu :: T = _init_field(T), cx :: T = _init_field(T), Jx :: Matrix{eltype(T)} = _init_field(Matrix{eltype(T)}), d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64  = NaN, current_score :: Union{T,eltype(T)} = _init_field(eltype(T)), evals :: Counters = Counters()) where T <: AbstractVector`

Note:
 - By default, unknown entries are set using `_init_field` (except evals).  
 - By default the type of `current_score` is `eltype(x)` and cannot be changed once the State is created.  
    To have a vectorized `current_score` of length n, try something like `GenericState(x, Array{eltype(x),1}(undef, n))`.  
 - All these information (except for `x` and `lambda`) are optionnal and need to be update when
    required. The update is done through the `update!` function.  
 - `x` and `lambda` are mandatory entries. If no constraints `lambda = []`.  
 - The constructor check the size of the entries.  

See also: `GenericState`, `update!`, `update_and_start!`, `update_and_stop!`, `reinit!`
"""
mutable struct 	NLPAtX{S, T <: AbstractVector, 
                       MT <: AbstractMatrix}  <: AbstractState{S, T}

#Unconstrained State
    x            :: T     # current point
    fx           :: eltype(T) # objective function
    gx           :: T  # gradient size: x
    Hx           :: MT  # hessian size: |x| x |x|

#Bounds State
    mu           :: T # Lagrange multipliers with bounds size of |x|

#Constrained State
    cx           :: T # vector of constraints lc <= c(x) <= uc
    Jx           :: MT  # jacobian matrix, size: |lambda| x |x|
    lambda       :: T    # Lagrange multipliers

    d            :: T #search direction
    res          :: T #residual

 #Resources State
    current_time   :: Float64
    current_score  :: S
    evals          :: Counters

 function NLPAtX(x             :: T,
                 lambda        :: T,
                 current_score :: S;
                 fx            :: eltype(T) = _init_field(eltype(T)),
                 gx            :: T = _init_field(T),
                 Hx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                 mu            :: T = _init_field(T),
                 cx            :: T = _init_field(T),
                 Jx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                 d             :: T = _init_field(T),
                 res           :: T = _init_field(T),
                 current_time  :: Float64 = NaN,
                 evals         :: Counters = Counters()
                 ) where {S, T <: AbstractVector}

  _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

  return new{S, T, Matrix{eltype(T)}}(x, fx, gx, Hx, mu, cx, Jx, lambda, d, 
                                      res, current_time, current_score, evals)
 end
end

function NLPAtX(x             :: T,
                lambda        :: T;
                fx            :: eltype(T) = _init_field(eltype(T)),
                gx            :: T = _init_field(T),
                Hx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                mu            :: T = _init_field(T),
                cx            :: T = _init_field(T),
                Jx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                d             :: T = _init_field(T),
                res           :: T = _init_field(T),
                current_time  :: Float64  = NaN,
                current_score :: Union{T,eltype(T)} = _init_field(eltype(T)),
                evals         :: Counters     = Counters()
                ) where T <: AbstractVector

 _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

 return NLPAtX(x, lambda, current_score, fx = fx, gx = gx,
               Hx = Hx, mu = mu, current_time = current_time,
               evals = evals)
end

function NLPAtX(x             :: T;
                fx            :: eltype(T) = _init_field(eltype(T)),
                gx            :: T = _init_field(T),
                Hx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                mu            :: T = _init_field(T),
                current_time  :: Float64 = NaN,
                current_score :: Union{T,eltype(T)} = _init_field(eltype(T)),
                evals         :: Counters     = Counters()
                ) where {T <: AbstractVector}

    _size_check(x, zeros(eltype(T),0), fx, gx, Hx, mu, 
                _init_field(T), _init_field(Matrix{eltype(T)}))

	return NLPAtX(x, zeros(eltype(T),0), current_score, fx = fx, gx = gx,
                  Hx = Hx, mu = mu, current_time = current_time,
                  evals = evals)
end

"""
reinit!: function that set all the entries at void except the mandatory x

`reinit!(:: NLPAtX, x :: AbstractVector, l :: AbstractVector; kwargs...)`

`reinit!(:: NLPAtX; kwargs...)`

Note: if *x*, *lambda* or *evals* are given as keyword arguments they will be
prioritized over the existing *x*, *lambda* and the default *Counters*.
"""
function reinit!(stateatx :: NLPAtX{S, T, MT}, 
                 x        :: T, 
                 l        :: T; 
                 kwargs...) where {S, T, MT}

 for k ∈ fieldnames(NLPAtX)
   if k ∉ [:x,:lambda] 
       setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k)))) 
   end
 end
 
 setfield!(stateatx, :x, x)
 setfield!(stateatx, :lambda, l)
 
 if length(kwargs)==0 
     return stateatx #save the update! call if no other kwargs than x
 end

 return update!(stateatx; kwargs...)
end

function reinit!(stateatx :: NLPAtX{S, T, MT}, 
                 x        :: T; 
                 kwargs...) where {S, T, MT}

 for k ∈ fieldnames(NLPAtX)
   if k ∉ [:x,:lambda] 
       setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k)))) 
   end
 end
 
 setfield!(stateatx, :x, x)
 
 if length(kwargs)==0 
     return stateatx #save the update! call if no other kwargs than x
 end

 return update!(stateatx; kwargs...)
end

function reinit!(stateatx :: NLPAtX; kwargs...)
 
 for k ∈ fieldnames(NLPAtX)
   if k ∉ [:x,:lambda] 
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

    if length(gx) != 0 &&  length(gx) != length(x)
     throw(error("Wrong size of gx in the NLPAtX."))
    end
    if size(Hx) != (0,0) && size(Hx) != (length(x), length(x))
     throw(error("Wrong size of Hx in the NLPAtX."))
    end
    if length(mu) != 0 && length(mu) != length(x)
     throw(error("Wrong size of mu in the NLPAtX."))
    end

    if lambda != zeros(0)
        if length(cx) != 0 && length(cx) != length(lambda)
         throw(error("Wrong size of cx in the NLPAtX."))
        end
        if size(Jx) != (0,0) && size(Jx) != (length(lambda), length(x))
         throw(error("Wrong size of Jx in the NLPAtX."))
        end
    end

end
