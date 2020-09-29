function prepare_LS(stp, x₀, d, τ₀, f₀, ∇f₀)
    # extract the nlp
    nlp = stp.pb
    # construct the line search model, which will be rebased at each iteration'current data
    ϕ = LSModel(nlp, x₀, d, τ₀, f₀, dot(∇f₀,d) )
    # instantiate stp, which will be adjusted at each iteration's current data
    ϕstp = LS_Stopping(ϕ, LSAtT(0.0),
                       optimality_check = (p,s) -> optim_check_LS(p,s), #Tanj: already set optim_check_LS(p,s,α,β)?
                       main_stp = stp,
                       max_iter = 40,
                       atol = 0.0, # to rely only on the Armijo-Wolfe conditions
                       rtol = 0.0, # otherwise, tolerance may prohibit convergence
                       unbounded_threshold = 1e100)

    return ϕ, ϕstp
end
