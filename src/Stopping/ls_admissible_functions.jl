export armijo, wolfe, armijo_wolfe, shamanskii_stop

"""
Check if a step size is admissible according to the Armijo criterion.
Inputs: Any, #LineModel and LSAtT.
Outputs: admissibility in the armijo sense
Armijo criterion: f(x + θd) - f(x) < τ₀ ∇f(x+θd)d
"""
function armijo(h      :: Any, #LineModel,  # never used?
                h_at_t :: LSAtT;
                τ₀ 	   :: Float64 = 0.01)

    fact = -0.8
    Eps = 1e-10
    hgoal = h_at_t.ht - h_at_t.h₀ - h_at_t.g₀ * h_at_t.x * τ₀
    # Armijo = (h_at_t.ht <= hgoal)# || ((h_at_t.ht <= h_at_t.h₀ + Eps * abs(h_at_t.h₀)) & (h_at_t.gt <= fact * h_at_t.g₀))
	# Armijo_HZ =
	# positive = h_at_t.x > 0.0   # positive step
    return max(hgoal, 0.0)
end

"""
Check if a step size is admissible according to the Wolfe criterion.
Inputs: Any, #LineModel and LSAtT.
Outputs: admissibility in the Strong Wolfe sense.
Strong Wolfe criterion: |∇f(x+θd)| < τ₁||∇f(x)||.
"""
function wolfe(h      :: Any, #LineModel,
               h_at_t :: LSAtT;
               τ₁ 	  :: Float64 = 0.99)

    wolfe = (τ₁ .* h_at_t.g₀) - (abs(h_at_t.gt))
	#positive = h_at_t.x > 0.0   # positive step
    return max(wolfe, 0.0)
end

"""
Check if a step size is admissible according to the "Shamanskii" criteria.
This criteria was proposed in
	GLOBAL CONVERGENCE TECHNIQUE FOR THE NEWTON METHOD WITH PERIODIC HESSIAN EVALUATION
	by
	F. LAMPARIELLO and M. SCIANDRONE
Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the "Shamanskii" sense (Bool).
More documentation needed.
"""
function shamanskii_stop(h      :: Any, #LineModel,
                         h_at_t :: LSAtT;
                         γ      :: Float64 = 1.0e-09)
    admissible = h_at_t.ht - h_at_t.h₀ - γ * (h_at_t.x)^3 * norm(h.d)^3
    return max(admissible, 0.0)
end


"""
Check if a step size is admissible according to both the Armijo and Wolfe criterion.
Inputs: Any, #LineModel and LSAtT.
Outputs: admissibility in the Strong Wolfe sense.
Strong Wolfe criterion: |∇f(x+θd)| < τ₁||∇f(x)||.
"""
function armijo_wolfe(h      :: Any, #LineModel,
                      h_at_t :: LSAtT;
                      τ₁     :: Float64 = 0.99)

    wolfe = (τ₁ .* h_at_t.g₀) - (abs(h_at_t.gt))
    armijo = h_at_t.ht - h_at_t.h₀ - h_at_t.g₀ * h_at_t.x * τ₀
    return max(armijo, wolfe, 0.0)
end

# """
# Check if a step size is admissible according to the Goldstein criteria.
# # Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the Goldstein sense (Bool).
# """
# function goldstein(h      :: Any, #LineModel,
#                    h_at_t :: LSAtT;
#                    τ₀     :: Float64 = 0.0001,
#                    τ₁     :: Float64 = 0.9999)
#
# 	goldstein = (h_at_t.h₀ + h_at_t.x * (1 - τ₀) * h_at_t.g₀) <= (h_at_t.ht) && (h_at_t.ht) <= (h_at_t.h₀ + h_at_t.x *  τ₀ * h_at_t.g₀)
# 	# positive = h_at_t.x > 0.0   # positive step
# 	return goldstein #&& positive
# end
