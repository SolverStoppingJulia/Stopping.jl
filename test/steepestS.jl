export steepestS

function steepestS(nlp :: AbstractNLPModel;
                  s :: TStopping = TStopping(),
                  verbose :: Bool=true,
                  linesearch :: Function = Newarmijo_wolfe,
                  kwargs...)
                  
    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)

    iter = 0

    s = start!(nlp,s,x)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, norm(∇f))

    optimal, unbounded, tired, elapsed_time = stop(nlp,s,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    while (OK && !(optimal || tired || unbounded) )
        d = - ∇f
        slope = ∇f ⋅ d
        if slope > 0.0
            stalled_ascent_dir = true
            #println("Not a descent direction! slope = ", slope)
        else
            verbose && @printf("  %8.1e", slope)

            # Perform improved Armijo linesearch.
            h = C1LineFunction(nlp, x, d)
            t, good_grad, ft, nbk, nbW, stalled_linesearch  = linesearch(h, f, slope, ∇ft, verbose=false; kwargs...)
            #!stalled_linesearch || println("Max number of Armijo backtracking ",nbk)
            verbose && @printf("  %4d\n", nbk)

            xt = x + t*d
            good_grad || (∇ft = grad!(nlp, xt, ∇ft))

            # Move on.
            x = xt
            f = ft
            ∇f = ∇ft
            iter = iter + 1

            verbose && @printf("%4d  %8.1e  %7.1e", iter, f, norm(∇f))

            optimal, unbounded, tired, elapsed_time = stop(nlp,s,iter,x,f,∇f)
        end
        OK = !stalled_linesearch & !stalled_ascent_dir
    end
    verbose && @printf("\n")


    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    return (x, f, s.optimality_residual(∇f), iter, optimal, tired, status, elapsed_time)
end
