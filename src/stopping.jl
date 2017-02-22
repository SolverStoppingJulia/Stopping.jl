# A stopping manager for iterative solvers
export TStopping, start!, stop

type TStopping
    nlp :: AbstractNLPModel        # the model
    atol :: Float64                # absolute tolerance
    rtol :: Float64                # relative tolerance    
    unbounded_threshold :: Float64 #
    max_obj_f :: Int    
    max_obj_grad :: Int    
    max_obj_hess :: Int
    max_obj_hv :: Int
    max_eval :: Int
    max_iter :: Int
    max_time :: Float64
    start_time :: Float64
    optimality :: Float64
    optimality_residual :: Function

    function TStopping(nlp :: AbstractNLPModel;
                      atol :: Float64 = 1.0e-8,
                      rtol :: Float64 = 1.0e-6,
                      unbounded_threshold :: Float64 = -1.0e50,
                      max_obj_f :: Int = typemax(Int),    
                      max_obj_grad :: Int = typemax(Int),   
                      max_obj_hess :: Int = typemax(Int),
                      max_obj_hv :: Int = typemax(Int),
                      max_eval :: Int = 20000,
                      max_iter :: Int = 5000,
                      max_time :: Float64 = 600.0, # 10 minutes
                      optimality_residual :: Function = x -> norm(x,Inf)
                     )

        return new(nlp, atol, rtol, unbounded_threshold, 
                   max_obj_f, max_obj_grad, max_obj_hess, max_obj_hv, max_eval, 
                   max_iter, max_time, NaN, Inf, optimality_residual)
    end
end


function start!(s :: TStopping,
               x₀ :: Array{Float64,1} )

    s.optimality = s.optimality_residual(grad(s.nlp,x₀))
    s.start_time = time()
    return s
end


function stop(s :: TStopping,
              iter :: Int,
              x :: Array{Float64,1},
              f :: Float64,
              ∇f :: Array{Float64,1},
              )

    counts = s.nlp.counters
    calls = [counts.neval_obj,  counts.neval_grad, counts.neval_hess, counts.neval_hprod]

    optimality = s.optimality_residual(∇f)

    optimal = (optimality < s.atol) | (optimality <( s.rtol * s.optimality)) 
    unbounded =  f < s.unbounded_threshold


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
