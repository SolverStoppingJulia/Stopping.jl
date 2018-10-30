export armijo, wolfe, goldstein, shamanskii_stop

""" Check if a step size is admissible according to the Armijo criteria.
Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the armijo sense (Bool)
Armijo criterion: f(x + θd) - f(x) < τ₀∇f(x+θd)d"""
function armijo(h      :: Any, #LineModel,  # on s'en sert nulle part...
				h_at_t :: LSAtT;
				τ₀ 	   :: Float64 = 0.01)
	# print_with_color(:bold, "on est dans armijo!!! \n")
	fact = -0.8
  	Eps = 1e-10
    hgoal = h_at_t.h₀ + h_at_t.g₀ * h_at_t.x * τ₀
    Armijo_HZ = (h_at_t.ht <= hgoal) || ((h_at_t.ht <= h_at_t.h₀ + Eps * abs(h_at_t.h₀)) & (h_at_t.gt <= fact * h_at_t.g₀))
	positive = h_at_t.x > 0.0   # on veut que le pas de déplacement soit positif
    return Armijo_HZ && positive
end

""" Check if a step size is admissible according to the Wolfe criteria.
Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the Strong Wolfe sense (Bool).
Strong Wolfe criterion: |∇f(x+θd)| < τ₁||∇f(x)||."""
function wolfe(h 	:: Any, #LineModel,
			   h_at_t :: LSAtT;
			   τ₁ 	:: Float64 = 0.99)

	wolfe = (abs(h_at_t.gt) <= -τ₁*h_at_t.g₀)
	positive = h_at_t.x > 0.0   # on veut que le pas de déplacement soit positif
    return wolfe  && positive
end

""" Check if a step size is admissible according to the Goldstein criteria.
Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the Goldstein sense (Bool)."""
function goldstein(h 	:: Any, #LineModel,
			   	   h_at_t :: LSAtT;
			   	   τ₀ 	:: Float64 = 0.0001,
			   	   τ₁ 	:: Float64 = 0.9999)

	goldstein = (h_at_t.h₀ + h_at_t.x * (1 - τ₀) * h_at_t.g₀) <= (h_at_t.ht) && (h_at_t.ht) <= (h_at_t.h₀ + h_at_t.x *  τ₀ * h_at_t.g₀)
	positive = h_at_t.x > 0.0   # on veut que le pas de déplacement soit positif
	return goldstein && positive
end

""" Check if a step size is admissible according to the "Shamanskii" criteria.
Inputs: Any, #LineModel and LSAtT. Outpus: admissibility in the "Shamanskii" sense (Bool).
More documentation needed."""
function shamanskii_stop(h 	:: Any, #LineModel,
			   	   		 h_at_t :: LSAtT;
			   	   		 τ₀ 	:: Float64 = 0.0001,
			   	   		 τ₁ 	:: Float64 = 0.9999)

	admissible = (h_at_t.ht) <= (h_at_t.h₀ - τ₀ * (h_at_t.x)^3 * norm(h.d)^3) # s'assurer qu'il y ait un using LinearAlgebra en quelque part
	positive = h_at_t.x > 0.0   # on veut que le pas de déplacement soit positif
	return admissible && positive
end
