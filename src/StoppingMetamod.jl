# module StoppingMetamod

export AbstractStoppingMeta, StoppingMeta, add_stop!
################################################################################
# Common stopping parameters to all optimization algorithms
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
    atol         :: Number                # absolute tolerance
    rtol         :: Number                # relative tolerance
	optimality0  :: Number                # value of the optimality residual at starting point

    unbounded_threshold :: Number # below this value, the problem is declared unbounded
    unbounded_x         :: Number # beyond this value, x is unbounded

    rtol_x              :: Number # algorithm is stalled move is beyond this value
    rtol_f              :: Number # algorithm is stalled move is beyond this value

    # fine grain control on ressources
    max_f           :: Int     # max function evaluations allowed

    # global control on ressources
    max_eval            :: Int     # max evaluations (f+g+H+Hv) allowed
    max_iter            :: Int     # max iterations allowed
    max_time            :: Number # max elapsed time allowed

    #intern Counters
    nb_of_stop :: Int

    # stopping properties status of the problem)
 	optimal_sub_pb      :: Bool
    unbounded           :: Bool
    tired               :: Bool
    stalled             :: Bool
    optimal             :: Bool
    feasible            :: Bool

	function StoppingMeta(;atol                :: Number   = 1.0e-6,
						   rtol                :: Number   = 1.0e-15,
						   optimality0         :: Number   = 1.0,
						   unbounded_threshold :: Number   = -1.0e50,
						   unbounded_x         :: Number   = 1.0e50,
						   max_f               :: Int      = typemax(Int),
						   max_eval            :: Int      = 20000,
						   max_iter            :: Int      = 5000,
						   max_time            :: Number   = 300.0,
					       kwargs...)
		optimal_sub_pb = false
		unbounded = false
		tired     = false
		stalled   = false
		optimal   = false

		rtol_x    = rtol
		rtol_f    = -eps(typeof(rtol_x)) #desactivate by default

        nb_of_stop = 0

		return new(atol, rtol, optimality0, unbounded_threshold, unbounded_x,
				   rtol_x, rtol_f, max_f, max_eval, max_iter, max_time,
                   nb_of_stop, optimal_sub_pb,
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
