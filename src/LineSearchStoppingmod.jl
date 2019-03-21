export LS_Stopping

################################################################################
# Line search stopping module
################################################################################

mutable struct LS_Stopping <: AbstractStopping
	# problem
	pb :: Any       # hard to define a proper type to avoid circular dependencies
					# I don't know the right solution to this situation...

	# stopping criterion proper to linesearch
	optimality_check :: Function

	# shared information with linesearch and other stopping
	meta :: StoppingMeta

	# current information on linesearch
	current_state :: LSAtT

	function LS_Stopping(pb         	:: Any,
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

 optimality = stp.optimality_check(stp.pb, stp.current_state)

 return 1. - optimality # TO CHANGE. Should change admissible functions
end

################################################################################
# line search admissibility functions
################################################################################
include("admissible_functions.jl")
