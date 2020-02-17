"""
Type: GenericState
Methods: update!, reinit!

A generic State to describe the state of a problem at a point x.

Tracked data include:
 - x : our current iterate
 - current_time  : time
 - current_score : score

Note: by default, unknown entries are set to nothing.
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

The function compare the kwargs and the entries of the State.
If the type of the kwargs is the same as the entry or the entry was nothing, then
it is updated.

Set convert to true, to update even incompatible types.
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
reinit!: function that set all the entries at void except the mandatory x

Note: If x is given as a keyword argument it will be prioritized over
the argument x.
"""
function reinit!(stateatx :: AbstractState, x :: Iterate; kwargs...)

 for k ∈ fieldnames(typeof(stateatx))
   if k != :x setfield!(stateatx, k, nothing) end
 end

 return update!(stateatx; x=x, kwargs...)
end

"""
reinit!: shorter version of reinit! reusing the x in the state

Note: If x is given as a keyword argument it will be prioritized over
the argument x.
"""
function reinit!(stateatx :: AbstractState; kwargs...)
 return reinit!(stateatx, stateatx.x; kwargs...)
end

"""
_domain_check: verifies is there is a NaN in the State entries

return true if a NaN has been found
"""
function _domain_check(stateatx :: AbstractState)
 domainerror = false

 for k ∈ fieldnames(typeof(stateatx))
   try domainerror = domainerror || (true in isnan.(getfield(stateatx, k))) catch end
 end

 return domainerror
end
