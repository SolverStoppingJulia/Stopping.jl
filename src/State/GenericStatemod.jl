"""
Type: GenericState

Methods: update!, reinit!

A generic State to describe the state of a problem at a point x.

Tracked data include:
 - x                   : current iterate
 - current_time [opt]  : time
 - current_score [opt] : score

Constructor: `GenericState(:: AbstractVector; current_time :: FloatVoid = nothing, current_score :: FloatVoid = nothing)`

Note: By default, unknown entries are set to *nothing*.

Examples:
GenericState(x)
GenericState(x, current\\_time = 1.0)
GenericState(x, current\\_score = 1.0)
"""
mutable struct GenericState <: AbstractState

    x :: AbstractVector

    #Current time
    current_time  :: FloatVoid
    #Current score
    current_score :: FloatVoid

    function GenericState(x             :: AbstractVector;
                          current_time  :: FloatVoid = nothing,
                          current_score :: FloatVoid = nothing)

      return new(x, current_time, current_score)
   end
end

"""
update!: generic update function for the State

`update!(:: AbstractState; convert = false, kwargs...)`

The function compares the kwargs and the entries of the State.
If the type of the kwargs is the same as the entry or the entry is *nothing*, then
it is updated.

Set kargs *convert* to true to update even incompatible types.

Examples:
update!(state1)
update!(state1, current\\_time = 2.0)
update!(state1, convert = true, current\\_time = 2.0)
"""
function update!(stateatx :: AbstractState; convert = false, kwargs...)

 kwargs = Dict(kwargs)

 for k ∈ fieldnames(typeof(stateatx))
  if (k ∈ keys(kwargs)) && (convert || getfield(stateatx, k) == nothing || typeof(kwargs[k]) ∈ [typeof(getfield(stateatx, k)), Nothing])
   setfield!(stateatx, k, kwargs[k])
  end
 end

 return stateatx
end

"""
reinit!: function that set all the entries at *nothing* except the mandatory *x*.

`reinit!(:: AbstractState, :: Iterate; kwargs...)`

Note: If *x* is given as a kargs it will be prioritized over
the second argument.

Examples:
reinit!(state2, zeros(2))
reinit!(state2, zeros(2), current_time = 1.0)

There is a shorter version of reinit! reusing the *x* in the state

`reinit!(:: AbstractState; kwargs...)`

Examples:
reinit!(state2)
reinit!(state2, current_time = 1.0)
"""
function reinit!(stateatx :: AbstractState, x :: Iterate; kwargs...)

 for k ∈ fieldnames(typeof(stateatx))
   if k != :x setfield!(stateatx, k, nothing) end
 end

 return update!(stateatx; x=x, kwargs...)
end

function reinit!(stateatx :: AbstractState; kwargs...)
 return reinit!(stateatx, stateatx.x; kwargs...)
end

"""
\\_domain\\_check: returns true if there is a NaN in the State entries, false otherwise

`_domain_check(:: AbstractState)`

Examples:
\\_domain\\_check(state1)
"""
function _domain_check(stateatx :: AbstractState)
 domainerror = false

 for k ∈ fieldnames(typeof(stateatx))
   try domainerror = domainerror || (true in isnan.(getfield(stateatx, k))) catch end
 end

 return domainerror
end
