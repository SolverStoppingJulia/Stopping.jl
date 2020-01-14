import NLPModels: Counters

"""
NLPAtX contains the important information concerning a non linear problem at
the iteration x. Basic information is:
 - x the current candidate for solution to our original problem
 - f(x) which is the funciton evaluation at x
 - g(x) which is the gradient evaluation at x
 - Hx which is the hessian representation at x

 - mu : Lagrange multiplier of the bounds constraints

 - cx : evaluation of the constraint function at x
 - Jx : jacobian matrix of the constraint function at x
 - lambda : Lagrange multiplier of the constraints

 - start_time : Default is a NaN, can be updated to fit the start of the algorithm.
 - evals : number of evaluations of the function (import the type NLPModels.Counters)

 All these information (except for x) are optionnal and need to be update when
 required. The update is done trhough the update! function.
"""
mutable struct 	NLPAtX <: AbstractState

#Unconstrained State
    x            :: Vector     # current point
    fx           :: FloatVoid   # objective function
    gx           :: Iterate     # gradient, size: x
    Hx           :: MatrixType  # Accurate? size: |x| x |x|

#Bounds State
    mu           :: Iterate     # Lagrange multipliers with bounds size of |x|

#Constrained State
    cx           :: Iterate     # vector of constraints lc <= c(x) <= uc
    Jx           :: MatrixType  # jacobian matrix, size: |lambda| x |x|
    lambda       :: Vector    # Lagrange multipliers

 #Resources State
    start_time   :: FloatVoid
    evals        :: Counters

 function NLPAtX(x          :: Vector,
                 lambda     :: Vector;
                 fx         :: FloatVoid    = nothing,
                 gx         :: Iterate      = nothing,
                 Hx         :: MatrixType   = nothing,
                 mu         :: Iterate      = nothing,
                 cx         :: Iterate      = nothing,
                 Jx         :: MatrixType   = nothing,
                 start_time :: FloatVoid    = nothing,
                 evals      :: Counters     = Counters())

  _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

  return new(x, fx, gx, Hx, mu, cx, Jx, lambda, start_time, evals)
 end
end

"""
An additional constructor for unconstrained problems
"""
function NLPAtX(x          :: Vector;
                fx         :: FloatVoid    = nothing,
                gx         :: Iterate      = nothing,
                Hx         :: MatrixType   = nothing,
                mu         :: Iterate      = nothing,
                start_time :: FloatVoid    = nothing,
                evals      :: Counters     = Counters())

    _size_check(x, zeros(0), fx, gx, Hx, mu, nothing, nothing)

	return NLPAtX(x, zeros(0), fx = fx, gx = gx,
                  Hx = Hx, mu = mu, start_time = start_time, evals = evals)
end

"""
Updates the (desired) values of an object of type NLPAtX.
Inputs:
 - An NLPAtX object
 - Any keywords that needs to be updated.
"""
function update!(nlpatx :: NLPAtX;
                 x      :: Iterate    = nothing,
                 fx     :: FloatVoid  = nothing,
                 gx     :: Iterate    = nothing,
                 Hx     :: MatrixType = nothing,
                 mu     :: Iterate    = nothing,
                 cx     :: Iterate    = nothing,
                 Jx     :: MatrixType = nothing,
                 lambda :: Iterate    = nothing,
                 tmps   :: FloatVoid  = nothing,
                 evals  :: Union{Counters, Nothing}  = nothing)

    _size_check(nlpatx.x, nlpatx.lambda, fx, gx, Hx, mu, cx, Jx)

    nlpatx.x   = x  == nothing  ? nlpatx.x   : x
    nlpatx.fx  = fx == nothing  ? nlpatx.fx  : fx
    nlpatx.gx  = gx == nothing  ? nlpatx.gx  : gx
    nlpatx.Hx  = Hx == nothing  ? nlpatx.Hx  : Hx
    nlpatx.mu  = mu == nothing  ? nlpatx.mu  : mu
    nlpatx.cx  = cx == nothing  ? nlpatx.cx  : cx
    nlpatx.Jx  = Jx == nothing  ? nlpatx.Jx  : Jx

    nlpatx.lambda     = lambda == nothing  ? nlpatx.lambda    : lambda

    nlpatx.start_time = tmps   == nothing ? nlpatx.start_time : tmps
    nlpatx.evals      = evals  == nothing ? nlpatx.evals      : evals

    return nlpatx
end

"""
Check the size of the entries in the State
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

# function convert_nlp(T,  nlpatx :: NLPAtX)
#
#     nlpatxT         = NLPAtX(zeros(T, length(nlpatx.x)))
#     nlpatxT.x       = typeof(nlpatx.x)      != Nothing ? convert.(T, nlpatx.x)      : nlpatx.x
#     nlpatxT.fx      = typeof(nlpatx.fx)     != Nothing ? convert.(T, nlpatx.fx)     : nlpatx.fx
#     nlpatxT.gx      = typeof(nlpatx.gx)     != Nothing ? convert.(T, nlpatx.gx)     : nlpatx.gx
#     nlpatxT.Hx      = typeof(nlpatx.Hx)     != Nothing ? convert.(T, nlpatx.Hx)     : nlpatx.Hx
#     nlpatxT.mu      = typeof(nlpatx.mu)     != Nothing ? convert.(T, nlpatx.mu)     : nlpatx.mu
#     nlpatxT.cx      = typeof(nlpatx.cx)     != Nothing ? convert.(T, nlpatx.cx)     : nlpatx.cx
#     nlpatxT.Jx      = typeof(nlpatx.Jx)     != Nothing ? convert.(T, nlpatx.Jx)     : nlpatx.Jx
#     nlpatxT.lambda  = typeof(nlpatx.lambda) != Nothing ? convert.(T, nlpatx.lambda) : nlpatx.lambda
#
#     return nlpatxT
# end
