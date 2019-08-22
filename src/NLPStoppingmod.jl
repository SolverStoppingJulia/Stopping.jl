export NLPStopping, unconstrained, fill_in!

################################################################################
# Specific stopping module for non linear problems
################################################################################

"""Stopping structure for non-linear (unconstrained?) programming problems.
Inputs:
 - An AbstractNLPModel
 - a stopping criterion through an admissibility function
 - the current state of the problem (i.e an NLPAtX)"""
mutable struct NLPStopping <: AbstractStopping
	# problem
	pb :: AbstractNLPModel

	# stopping criterion
	optimality_check :: Function # will be put in optimality_check

	# common line search parameters
	meta :: StoppingMeta

	# current state of the line search Algorithm
	current_state :: AbstractState

	function NLPStopping(pb         	:: AbstractNLPModel,
						 admissible 	:: Function,
						 current_state 	:: AbstractState;
						 meta       	:: StoppingMeta = StoppingMeta(),
						 kwargs...)

		if !(isempty(kwargs))
			meta = StoppingMeta(;kwargs...)
		end

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
# non linear problems admissibility functions
################################################################################
include("nlp_admissible_functions.jl")
