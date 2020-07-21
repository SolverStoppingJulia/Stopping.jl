"""
Type: NLPAtX

Methods: update!, reinit!

NLPAtX contains the information concerning a nonlinear optimization model at
the iterate x.

min_{x ∈ ℜⁿ} f(x) subject to lcon <= c(x) <= ucon, lvar <= x <= uvar.

Tracked data include:
 - x : the current iterate
 - fx [opt] : function evaluation at x
 - gx [opt] : gradient evaluation at x
 - Hx [opt] : hessian evaluation at x

 - mu [opt] : Lagrange multiplier of the bounds constraints

 - cx [opt] : evaluation of the constraint function at x
 - Jx [opt] : jacobian matrix of the constraint function at x
 - lambda : Lagrange multiplier of the constraints

 - current_time [opt]  : time
 - current_score [opt] : score
 - evals [opt] : number of evaluations of the function (import the type NLPModels.Counters)

Constructor:
`NLPAtX(:: AbstractVector, :: AbstractVector; fx :: FloatVoid = nothing, gx :: Iterate = nothing, Hx :: MatrixType = nothing, mu :: Iterate = nothing, cx :: Iterate = nothing, Jx :: MatrixType = nothing, current_time :: FloatVoid = nothing, current_score :: FloatVoid = nothing, evals :: Counters = Counters())`

`NLPAtX(:: AbstractVector; fx :: FloatVoid = nothing, gx :: Iterate = nothing, Hx :: MatrixType = nothing, mu :: Iterate = nothing, current_time :: FloatVoid = nothing, current_score :: FloatVoid = nothing, evals :: Counters = Counters())`

Note:
      - By default, unknown entries are set to *nothing* (except evals).
      - All these information (except for *x* and *lambda*) are optionnal and need to be update when
        required. The update is done through the update! function.
      - *x* and *lambda* are mandatory entries. If no constraints `lambda = []`.
      - The constructor check the size of the entries.
"""
mutable struct 	NLPAtX <: AbstractState

#Unconstrained State
    x            :: AbstractVector     # current point
    fx           :: FloatVoid   # objective function
    gx           :: Iterate     # gradient size: x
    Hx           :: MatrixType  # hessian size: |x| x |x|

#Bounds State
    mu           :: Iterate     # Lagrange multipliers with bounds size of |x|

#Constrained State
    cx           :: Iterate     # vector of constraints lc <= c(x) <= uc
    Jx           :: MatrixType  # jacobian matrix, size: |lambda| x |x|
    lambda       :: AbstractVector    # Lagrange multipliers

 #Resources State
    current_time   :: FloatVoid
    current_score  :: FloatVoid
    evals          :: Counters

 function NLPAtX(x             :: AbstractVector,
                 lambda        :: AbstractVector;
                 fx            :: FloatVoid    = nothing,
                 gx            :: Iterate      = nothing,
                 Hx            :: MatrixType   = nothing,
                 mu            :: Iterate      = nothing,
                 cx            :: Iterate      = nothing,
                 Jx            :: MatrixType   = nothing,
                 current_time  :: FloatVoid    = nothing,
                 current_score :: FloatVoid    = nothing,
                 evals         :: Counters     = Counters())

  _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

  return new(x, fx, gx, Hx, mu, cx, Jx, lambda, current_time, current_score, evals)
 end
end

function NLPAtX(x             :: AbstractVector;
                fx            :: FloatVoid    = nothing,
                gx            :: Iterate      = nothing,
                Hx            :: MatrixType   = nothing,
                mu            :: Iterate      = nothing,
                current_time  :: FloatVoid    = nothing,
                current_score :: FloatVoid    = nothing,
                evals         :: Counters     = Counters())

    _size_check(x, zeros(0), fx, gx, Hx, mu, nothing, nothing)

	return NLPAtX(x, zeros(0), fx = fx, gx = gx,
                  Hx = Hx, mu = mu, current_time = current_time,
                  current_score = current_score, evals = evals)
end

"""
reinit!: function that set all the entries at void except the mandatory x

`reinit!(:: NLPAtX, x :: AbstractVector, l :: AbstractVector; kwargs...)`

`reinit!(:: NLPAtX; kwargs...)`

Note: if *x*, *lambda* or *evals* are given as keyword arguments they will be
prioritized over the existing *x*, *lambda* and the default *Counters*.
"""
function reinit!(stateatx :: NLPAtX, x :: AbstractVector, l :: AbstractVector; kwargs...)

 for k ∈ fieldnames(typeof(stateatx))
   if !(k ∈ [:x,:lambda,:evals]) setfield!(stateatx, k, nothing) end
 end

 return update!(stateatx; x=x, lambda = l, evals = Counters(), kwargs...)
end

function reinit!(stateatx :: NLPAtX; kwargs...)
 return reinit!(stateatx, stateatx.x, stateatx.lambda; kwargs...)
end

"""
_size_check!: check the size of the entries in the State

`_size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)`
"""
function _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

    if gx != nothing && length(gx) != length(x)
     throw(error("Wrong size of gx in the NLPAtX."))
    end
    if Hx != nothing && size(Hx) != (length(x), length(x))
     throw(error("Wrong size of Hx in the NLPAtX."))
    end
    if mu != nothing && length(mu) != length(x)
     throw(error("Wrong size of mu in the NLPAtX."))
    end

    if lambda != zeros(0)
        if cx != nothing && length(cx) != length(lambda)
         throw(error("Wrong size of cx in the NLPAtX."))
        end
        if Jx != nothing && size(Jx) != (length(lambda), length(x))
         throw(error("Wrong size of Jx in the NLPAtX."))
        end
    end

end
