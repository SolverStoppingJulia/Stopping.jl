export Proj_LS

function Proj_LS(h :: LineModel,
                 h₀ :: Float64,
                 slope :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 bk_max :: Int=50,
                 nbWM :: Int=50,
                 verboseLS :: Bool=false,
                 check_slope :: Bool = false,
                 kwargs...)
    
    if check_slope
        (abs(slope - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
        verboseLS && @show h₀ obj(h, 0.0) slope grad(h,0.0)
    end
    
    # Perform improved Armijo projected linesearch.
    nbk = 0
    nbW = 0
    t = 1.0

    # First try to increase t to satisfy loose Wolfe condition
    ht = obj(h, t)
    slope_t = grad!(h, t, g)
    while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < nbWM)
        t *= 5.0
        ht = obj(h, t)
        slope_t = grad!(h, t, g)

        nbW += 1
        verboseLS && @printf(" W  %4d  slope  %4d slope_t %4d\n", nbW, slope, slope_t);
    end

    hgoal = h₀ + slope * t * τ₀;
    fact = -0.8
    ϵ = 1e-10

    # Enrich Armijo's condition with Hager & Zhang numerical trick
    Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
    good_grad = true
    while !Armijo && (nbk < bk_max)
        t *= 0.4
        ht = obj(h, t)
        hgoal = h₀ + slope * t * τ₀;

        # avoids unused grad! calls
        Armijo = false
        good_grad = false
        if ht <= hgoal
            Armijo = true
        elseif ht <= h₀ + ϵ * abs(h₀)
            slope_t = grad!(h, t, g)
            good_grad = true
            if slope_t <= fact * slope
                Armijo = true
            end
        end

        nbk += 1
        verboseLS && @printf(" A  %4d  h0  %4e ht %4e\n", nbk, h₀, ht);
    end

    verboseLS && @printf("  %4d %4d %8e\n", nbk, nbW, t);
    stalled = (nbk == bk_max)
    @assert (t > 0.0) && (!isnan(t)) "invalid step"
    return (t, t, good_grad, ht, nbk, nbW, stalled)#,h.f_eval,h.g_eval,h.h_eval)
end
