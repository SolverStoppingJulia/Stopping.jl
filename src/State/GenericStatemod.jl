################################################################################
# This is the Generic implementation of an AbstractState. More documentation
# can be found on the specific types and the README.
################################################################################
abstract type AbstractState end

mutable struct GenericState <: AbstractState

    x :: Iterate

    #Starting time
    current_time :: FloatVoid

    function GenericState(x            :: Iterate;
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
  if (k ∈ keys(kwargs)) && (typeof(getfield(stateatx, k)) ∈ [typeof(kwargs[k]), Nothing] || convert)
   setfield!(stateatx, k, kwargs[k])
  end
 end

 return stateatx
end
