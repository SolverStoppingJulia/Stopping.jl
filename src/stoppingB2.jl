# A stopping manager for iterative solvers, bound constraints compatible
export TStoppingB, start!, stop


type TResult   # simplified state for bound constrained problems
    x :: Vector  # current iterate
    f :: Float64 # current objective value
    ∇f :: Vector # current objective gradient
    pg :: Vector # current objective projected gradient
    λ :: Vector  # current Lagrange multipliers
    # etc
end
include("count_utils.jl")

type TStoppingB <: AbstractStopping
    atol :: Float64                  # absolute tolerance
    rtol :: Float64                  # relative tolerance
    unbounded_threshold :: Float64   # below this value, the problem is declared unbounded
    stalled_x_threshold :: Float64
    stalled_f_threshold :: Float64
    # fine grain control on ressources
    max_counters :: NLPModels.Counters
    # global control on ressources
    max_eval :: Int                  # max evaluations (f+g+H+Hv) allowed
    max_iter :: Int                  # max iterations allowed
    max_time :: Float64              # max elapsed time allowed
    # global information to the stopping manager
    iter :: Int
    start_time :: Float64            # starting time of the execution of the method
    optimality0 :: Float64           # value of the optimality residual at starting point
    optimality_residual :: Function  # function to compute the optimality residual
    # diagnostic
    elapsed_time :: Float64
    optimal :: Bool
    tired :: Bool
    unbounded :: Bool
    stalled :: Bool
    feasible :: Bool                 # Used for algos stopping when reaching feasibility
    unfeasible :: Bool               # Used for algos interrupted when loosing feasibility
    #
    nlp :: AbstractNLPModel
    #
    nlp_at_x :: TResult


    function TStoppingB(nlp :: AbstractNLPModel;
                        atol :: Float64 = 1.0e-8,
                        rtol :: Float64 = 1.0e-6,
                        unbounded_threshold :: Float64 = -1.0e50,
                        stalled_x_threshold :: Float64 = eps(),
                        stalled_f_threshold :: Float64 = eps(),
                        max_eval :: Int = 20000,
                        max_iter :: Int = 5000,
                        max_time :: Float64 = 600.0, # 10 minutes
                        optimality_residual :: Function = optim_check_unconstrained,
                        kwargs...)
        
        max_counts = NLPModels.Counters()
        put_at_inf!(max_counts)
        
        return new(atol, rtol, unbounded_threshold, stalled_x_threshold, stalled_f_threshold,
                   max_counts, max_eval,
                   max_iter, max_time, 0, NaN, Inf, optimality_residual, 0.0,
                   false, false, false, false, false, false,
                   nlp,
                   TResult([],NaN,[],[],[]))
    end
end

include("check_tired.jl")
include("check_optim.jl")


proj(ub :: Vector, lb :: Vector, x :: Vector) = max.(min.(x,ub),lb)

gradproj(ub :: Vector, lb :: Vector, g::Vector, x :: Vector) =  x - proj(ub, lb, x-g)

function start!( s :: TStoppingB,
                x₀ :: Array{Float64,1})
    
    f₀,∇f₀ = objgrad(s.nlp,x₀)
    s.start_time  = time()
    s.iter = 0

    s.nlp_at_x.x = x₀
    s.nlp_at_x.f = f₀
    s.nlp_at_x.∇f = ∇f₀
    
    s.optimality0 = s.optimality_residual(s)
    return s, ∇f₀
end


function stop(s :: TStoppingB,
              iter :: Int,
              x :: Vector,
              f :: Float64,
              ∇f :: Vector;
              pg :: Vector = [],
              λ ::  Vector = [])

    # verify stalled before updating nlp_at_x
    if iter == 0
        s.stalled = false
    else
        stalled_x = (norm(x-s.nlp_at_x.x)/norm(x)) < s.stalled_x_threshold
        stalled_f = (abs(f-s.nlp_at_x.f)/abs(f))   < s.stalled_f_threshold
        s.stalled = stalled_x | stalled_f
    end

    # update nlp_at_x
    s.nlp_at_x.x  = x
    s.nlp_at_x.f  = f
    s.nlp_at_x.∇f = ∇f
    s.nlp_at_x.pg = pg
    s.nlp_at_x.λ  = λ
    s.iter        = iter

    optimality = s.optimality_residual(s)
    s.optimal = (optimality < s.atol) | (optimality < (s.rtol * s.optimality0))

    s.unbounded =  f <= s.unbounded_threshold

    s.tired = tired_check(s)

    return (s.optimal || s.unbounded || s.tired || s.stalled || s.feasible || s.unfeasible)
end
