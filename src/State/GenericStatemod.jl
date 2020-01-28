################################################################################
# This is the Generic implementation of an AbstractState. More documentation
# can be found on the specific types and the README.
################################################################################
abstract type AbstractState end

mutable struct GenericState <: AbstractState

    x :: Vector

    #Starting time
    current_time :: FloatVoid

    function GenericState(x            :: Vector;
                          current_time :: FloatVoid = nothing)

      return new(x, current_time)
   end
end

"""
Generic update function for the State
The function compare the kwargs and the entries of the State.
If the type of the kwargs is the same as the entry or the entry was void, then
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
reinit!: short version of reinit! reusing the x in the state

Note: If x is given as a keyword argument it will be prioritized over
the argument x.
"""
function reinit!(stateatx :: AbstractState; kwargs...)
 return reinit!(stateatx, stateatx.x; kwargs...)
end
