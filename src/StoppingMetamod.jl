# module StoppingMetamod

export AbstractStoppingMeta, StoppingMeta, add_stop!
################################################################################
# Paramètre d'arrêt commun à tout les algos d'optim (1D, line search, sans
# contraintes, etc.)
# On a dit à la dernière rencontre qu'on voulait splitter le Meta en plusieurs
# petits meta. Comment on le split? En attendant j'ai juste ajouter le x
# dans les trucs du stoppingMeta
# Sam
################################################################################

"""Common stopping criterion for "all" optimization algorithms such as:
    - absolute and relative tolerance
    - threshold for unboundedness
    - time limit to let the algorithm run
    - maximum number of function (and derivatives) evaluations"""
abstract type AbstractStoppingMeta end

mutable struct StoppingMeta <: AbstractStoppingMeta # mutable ? ou immutable?
                                                    # veut-on changer la tolérance
                                                    # en cours de route?
	# problem tolerances
    atol :: Float64                # absolute tolerance
    rtol :: Float64                # relative tolerance

    unbounded_threshold :: Float64 # below this value, the problem is declared unbounded
    unbounded_x         :: Float64 # beyond this value, x is unbounded

    rtol_x              :: Float64 # algorithm is stalled move is beyond this value
    rtol_f              :: Float64 # algorithm is stalled move is beyond this value

    # fine grain control on ressources
    max_f           :: Int     # max function evaluations allowed

    # global control on ressources
    max_eval            :: Int     # max evaluations (f+g+H+Hv) allowed
    max_iter            :: Int     # max iterations allowed
    max_time            :: Float64 # max elapsed time allowed

    #intern Counters
    nb_of_stop :: Int

    # stopping properties            # mettre dans une fonction status! ?
	stalled_linesearch  :: Bool
    unbounded           :: Bool
    tired               :: Bool
    stalled             :: Bool
    optimal             :: Bool
    feasible            :: Bool

    # Information on the problem at the current iterate
    #nlp_at_x :: InterfaceResult

	function StoppingMeta(;atol                :: Float64  = 1.0e-6,
						   rtol                :: Float64  = 1.0e-15,
						   unbounded_threshold :: Float64  = -1.0e50,
						   unbounded_x         :: Float64  = 1.0e50,
						   max_f               :: Int      = typemax(Int),
						   max_eval            :: Int      = 20000,
						   max_iter            :: Int      = 5000,
						   max_time            :: Float64  = 300.0,
					       kwargs...)
		stalled_linesearch = false
		unbounded = false
		tired     = false
		stalled   = false
		optimal   = false

		rtol_x    = rtol
		rtol_f    = -eps(Float64) #desactivate by default

        nb_of_stop = 0

		return new(atol, rtol, unbounded_threshold, unbounded_x,
				   rtol_x, rtol_f, max_f, max_eval, max_iter, max_time,
                   nb_of_stop, stalled_linesearch,
				   unbounded, tired, stalled, optimal)
	end
end

"""
Fonction called everytime stop! is called. In theory should be called once every
iteration of an algorithm
"""
function add_stop!(meta :: StoppingMeta)
	meta.nb_of_stop += 1
end

#end of module
# end
