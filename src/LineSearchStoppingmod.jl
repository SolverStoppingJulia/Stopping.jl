export LS_Stopping

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
# import LSAtTmod.LSAtT
# import StoppingMetamod.StoppingMeta
# import LineModel2mod.LineModel

mutable struct LS_Stopping <: AbstractStopping
	# probleme
	pb :: Any       # si on met LineModel il faut l'importer...
					# ou peut-être AffineModel éventuellement
	                # pas juste un AbstractNLPModel, car pb.d nécessaire.

	# critère d'arrêt propre au Line search
	optimality_check :: Function # critère qu'on va mettre dans optimality_check

	# information que le line search a en commun avec les autres stopping
	meta :: StoppingMeta

	# information courante sur le Line search
	current_state :: LSAtT

	function LS_Stopping(pb         	:: Any,             # LineModel pose problème car on doit l'importer
						 admissible 	:: Function,
						 current_state 	:: LSAtT;
						 meta       	:: StoppingMeta = StoppingMeta())

		return new(pb, admissible, meta, current_state)
	end

end


function _unbounded_check!(stp  :: LS_Stopping,
                           x    :: Iterate)
 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x
 if isnan(stp.current_state.ht)
	 stp.current_state.ht = obj(stp.pb, x)
 end
 f_too_large = stp.current_state.ht <= stp.meta.unbounded_threshold

 stp.meta.unbounded = x_too_large || f_too_large

 return stp
end

#### Pourquoi on avait fait ça?
#### Un linesearch peut piétinner et c'est mal... peut-être que la fonction
#### de base est pas adapté mais celle la fait rien :/
# import GenericStoppingmod._stalled_check!
# """_stalled_check. Checks if the optimization algorithm is stalling."""
# function _stalled_check!(stp    :: LS_Stopping,
#                          x      :: Iterate;
#                          dx     :: Iterate = Float64[],
#                          df     :: Iterate = Float64[])
#
#  stp.meta.stalled = false
#
#  return stp
# end

function _optimality_check(stp  :: LS_Stopping)

 # print_with_color(:red, "on passe par ici dans le _optimality_check de LineSearch \n")

 optimality = stp.optimality_check(stp.pb, stp.current_state)

 return 1. - optimality # A MODIFIER QUAND ON AURA TRANSFORMER LES ADMISSIBLES EN TEST A 0
end

################################################################################
# Différentes fonctions de conditions d'admissibilité
################################################################################
include("admissible_functions.jl")
