export optim_check_bounded,
    optim_check_bounded2,
    optim_check_bounded3,
    optim_check_unconstrained,
    optim_check_U_feasible

# 2-norm
optim_check_bounded(s) = norm(gradproj(s.nlp.meta.uvar,s.nlp.meta.lvar,s.nlp_at_x.∇f,s.nlp_at_x.x))

# Inf-norm and computing pg when needed
function optim_check_bounded2(s::TStoppingB)
    # compute pg if needed
    if s.nlp_at_x.pg == []
        s.nlp_at_x.pg = gradproj(s.nlp.meta.uvar,s.nlp.meta.lvar,s.nlp_at_x.∇f,s.nlp_at_x.x)
    end
        return norm(s.nlp_at_x.pg,Inf)
end


#use Lagrange multipliers for assessing optimality.
function optim_check_bounded3(s::TStoppingB)
    x = s.nlp_at_x.x
    n = length(x)
    A = sparse([speye(n);-speye(n)])

    ϵ = 0.0   #  ϵ-active constraints. In checking optimality conditions, no need to use ϵ!=0.
    if s.nlp_at_x.λ == []
        # compute λ since needed
        Auϵ = find((s.nlp.meta.uvar - x) .<= ϵ)       # first n multipliers for upper bounds
        Alϵ = n .+ find((x - s.nlp.meta.lvar) .<= ϵ)  # n+1 to 2n for lower bounds

        λ = zeros(2*n)
        λ[Auϵ] = max.(- s.nlp_at_x.∇f[Auϵ],0.0)      # multipliers for upper bounds in [1:n]
        λ[Alϵ] = max.(  s.nlp_at_x.∇f[Alϵ .- n],0.0) # multipliers for lower bounds in [n+1:2n]

        s.nlp_at_x.λ = λ
    end
    
    λ = s.nlp_at_x.λ
    L = s.nlp_at_x.∇f + A'*λ
    return norm(L,Inf)
end


optim_check_unconstrained(s) = norm(s.nlp_at_x.∇f,Inf)

# set unfeasible flag to trigger stopping. Useful in active set algorithms.
function optim_check_U_feasible(s::TStoppingB)
    # check feasibility (bounds)
    if any(s.nlp_at_x.x .<  s.nlp.meta.lvar) | any(s.nlp_at_x.x .>  s.nlp.meta.uvar)
        s.unfeasible = true
        return Inf
    else
        return norm(s.nlp_at_x.∇f,Inf)
    end
end
