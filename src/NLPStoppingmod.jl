export NLPStopping, unconstrained, fill_in!

################################################################################
#  LS_Stopping est un sous-type de AbstractStopping
#  Est-ce qu'on doit ajouter lower_bound dans le LS_Stopping?
#  Car on l'utilise dans optimality_check qui est une fonction de LS_Stopping
#  donc une fonction de LS_Stopping doit appeler un param de LS_Stopping....
################################################################################

################################################################################
# Si on veut ajouter le ls_at_t ou peut importe son nom a LS_Stopping
# Sam
################################################################################

"""Stopping structure for non-linear (unconstrained?) programming problems.
Inputs:
 - An AbstractNLPModel
 - a stopping criterion through an admissibility function
 - the current state of the problem (i.e an NLPAtX)"""
mutable struct NLPStopping <: AbstractStopping
	# probleme
	pb :: AbstractNLPModel

	# critère d'arrêt propre au Line search
	optimality_check :: Function # critère qu'on va mettre dans optimality_check

	# information que le line search a en commun avec les autres stopping
	meta :: StoppingMeta

	# information courante sur le Line search
	current_state :: NLPAtX

	function NLPStopping(pb         	:: AbstractNLPModel,
						 admissible 	:: Function,
						 current_state 	:: NLPAtX;
						 meta       	:: StoppingMeta = StoppingMeta())

		return new(pb, admissible, meta, current_state)
	end

end

"""
fill_in! : A function that fill in the required values in the State
"""
function fill_in!(stp  :: NLPStopping,
                  x    :: Iterate;
                  fx   :: Iterate    = nothing,
                  gx   :: Iterate    = nothing)

 obfx = fx == nothing  ? obj(stp.pb, x)   : fx
 grgx = gx == nothing  ? grad(stp.pb, x)  : gx

 return update!(stp.current_state, x=x, fx = obfx, gx = grgx)
end

function _unbounded_check!(stp  :: NLPStopping,
                           x    :: Iterate)
 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x

 if stp.current_state.fx == nothing
	 stp.current_state.fx = obj(stp.pb, x)
 end
 f_too_large = stp.current_state.fx <= stp.meta.unbounded_threshold

 stp.meta.unbounded = x_too_large || f_too_large

 return stp
end

function _optimality_check(stp  :: NLPStopping)

 optimality = stp.optimality_check(stp.pb, stp.current_state)

 return optimality
end

################################################################################
# Différentes fonctions de conditions d'admissibilité
################################################################################
include("nlp_admissible_functions.jl")
