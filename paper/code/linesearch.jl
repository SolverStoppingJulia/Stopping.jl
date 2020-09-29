#linesearch is an intermediary between NewtonStopLS and a 1D solver.
#change the name?
function linesearch(ϕ    :: LSModel,
                    ϕstp :: AbstractStopping,  # be more specific for stp
                    x₀   :: AbstractVector,
                    d    :: AbstractVector,
                    f₀   :: AbstractFloat,
                    ∇f₀  :: AbstractVector,
                    τ₀   :: Real,
                    τ₁   :: Real)
    # rebase the LSModel
    g₀ = dot(∇f₀,d)
    rebase!(ϕ, x₀, d, τ₀, f₀, g₀)

    # convert the Armijo and Wolfe criteria to an asymetric interval [α,β]
    α =  (τ₁-τ₀)*g₀
    β = -(τ₁+τ₀)*g₀

    # reuse the stopping
    reinit!(ϕstp)
    ϕstp.pb = ϕ
    # redefine the optimality_check function using α and β
    ϕstp.optimality_check =  (p,s) -> optim_check_LS(p,s,α,β)

    # optimize in the interval [0.0,Inf]
    ϕstp = min_1D(ϕ, 0.0, Inf, α, β, stp = ϕstp)

    # unpack the results
    t = ϕstp.current_state.x #Tanj: Shouldn't we check if ϕstp.meta.optimal = true?
    ft = ϕ.f   # must rely on the solver (min_1D) so that the last evaluation was
    gt = ϕ.∇f  # at the optimal step, so that the stored value for f and ∇f are valid

    return t, ft, gt, ϕstp
end
