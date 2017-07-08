# A stopping manager for iterative solvers
export TStopping, start!, stop

type TStopping
    atol :: Float64                  # absolute tolerance
    rtol :: Float64                  # relative tolerance
    unbounded_threshold :: Float64   # below this value, the problem is declared unbounded
    # fine grain control on ressources
    max_obj_f :: Int                 # max objective function (f) evaluations allowed
    max_obj_grad :: Int              # max objective gradient (g) evaluations allowed
    max_obj_hess :: Int              # max objective hessian (H) evaluations allowed
    max_obj_hv :: Int                # max objective H*v (HV) evaluations allowed
    # global control on ressources
    max_eval :: Int                  # max evaluations (f+g+H+Hv) allowed
    max_iter :: Int                  # max iterations allowed
    max_time :: Float64              # max elapsed time allowed
    # global information to the stopping manager
    start_time :: Float64            # starting time of the execution of the method
    optimality0 :: Float64           # value of the optimality residual at starting point
    optimality_residual :: Function  # function to compute the optimality residual


    function TStopping(;atol :: Float64 = 1.0e-8,
                       rtol :: Float64 = 1.0e-6,
                       unbounded_threshold :: Float64 = -1.0e50,
                       max_obj_f :: Int = typemax(Int),
                       max_obj_grad :: Int = typemax(Int),
                       max_obj_hess :: Int = typemax(Int),
                       max_obj_hv :: Int = typemax(Int),
                       max_eval :: Int = 20000,
                       max_iter :: Int = 5000,
                       max_time :: Float64 = 600.0, # 10 minutes
                       optimality_residual :: Function = x -> norm(x,Inf),
                       kwargs...)
        
        return new(atol, rtol, unbounded_threshold,
                   max_obj_f, max_obj_grad, max_obj_hess, max_obj_hv, max_eval,
                   max_iter, max_time, NaN, Inf, optimality_residual)
    end
end




function start!(nlp :: AbstractNLPModel,
                s :: TStopping,
                x₀ :: Array{Float64,1} )

    ∇f₀ = grad(nlp,x₀)
    s.optimality0 = s.optimality_residual(∇f₀)
    s.start_time  = time()
    return s, ∇f₀
end


function stop(nlp :: AbstractNLPModel,
              s :: TStopping,
              iter :: Int,
              x :: Array{Float64,1},
              f :: Float64,
              ∇f :: Array{Float64,1},
              )

    counts = nlp.counters
    calls = [counts.neval_obj,  counts.neval_grad, counts.neval_hess, counts.neval_hprod]

    optimality = s.optimality_residual(∇f)

    optimal = (optimality < s.atol) | (optimality <( s.rtol * s.optimality0))
    #optimal = optimality < s.atol +s.rtol*s.optimality0
    unbounded =  f <= s.unbounded_threshold


    # fine grain limits
    max_obj_f  = calls[1] > s.max_obj_f
    max_obj_g  = calls[2] > s.max_obj_grad
    max_obj_H  = calls[3] > s.max_obj_hess
    max_obj_Hv = calls[4] > s.max_obj_hv

    max_total = sum(calls) > s.max_eval

    # global evaluations diagnostic
    max_calls = (max_obj_f) | (max_obj_g) | (max_obj_H) | (max_obj_Hv) | (max_total)

    elapsed_time = time() - s.start_time

    max_iter = iter >= s.max_iter
    max_time = elapsed_time > s.max_time

    # global user limit diagnostic
    tired = (max_iter) | (max_calls) | (max_time)


    # return everything. Most users will use only the first four fields, but return
    # the fine grained information nevertheless.
    return optimal, unbounded, tired, elapsed_time,
           max_obj_f, max_obj_g, max_obj_H, max_obj_Hv, max_total, max_iter, max_time
end
