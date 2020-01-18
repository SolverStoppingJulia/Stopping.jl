################################################################################
# This is the Generic implementation of an AbstractState. More documentation
# can be found on the specific types and the README.
################################################################################
abstract type AbstractState end

mutable struct GenericState <: AbstractState

    x :: Iterate

    #Starting time
    start_time :: FloatVoid

    function GenericState(x          :: Iterate;
                          start_time :: FloatVoid = nothing)

      return new(x, start_time)
   end
end

function update!(stateatx :: AbstractState;
                 x        :: Iterate    = nothing,
                 tmps     :: FloatVoid  = nothing)

 stateatx.x          = x      == nothing ? stateatx.x : x
 stateatx.start_time = tmps   == nothing ? stateatx.start_time : tmps

 return stateatx
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
