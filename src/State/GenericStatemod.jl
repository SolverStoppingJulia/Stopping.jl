_init_field(t :: Type) = _init_field(Val{t}())
_init_field(::Val{T}) where T <: AbstractMatrix = zeros(eltype(T),0,0)
_init_field(::Val{T}) where T <: SparseMatrixCSC = sparse(zeros(eltype(T),0,0))
_init_field(::Val{T}) where T <: AbstractVector = zeros(eltype(T),0)
_init_field(::Val{T}) where T <: SparseVector = sparse(zeros(eltype(T),0))
_init_field(::Val{BigFloat}) = BigFloat(NaN)
_init_field(::Val{Float64})  = NaN
_init_field(::Val{Float32})  = NaN32
_init_field(::Val{Float16})  = NaN16
_init_field(::Val{Missing})  = missing
_init_field(::Val{Nothing})  = nothing
_init_field(::Val{Bool})     = false #unfortunately no way to differentiate
_init_field(::Val{T}) where T<:Number = typemin(T)
_init_field(::Val{Counters}) = Counters()

"""
Type: GenericState

Methods: update!, reinit!

A generic State to describe the state of a problem at a point x.

Tracked data include:
 - x             : current iterate
 - d [opt]       : search direction
 - res [opt]     : residual
 - current_time  : time
 - current_score : score

Constructors:
 `GenericState(:: T, :: S; d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64 = NaN) where {S, T <:AbstractVector}`

 `GenericState(:: T; d :: T = _init_field(T), res :: T = _init_field(T), current_time :: Float64 = NaN, current_score :: Union{T,eltype(T)} = _init_field(eltype(T))) where T <:AbstractVector`

Note: By default, unknown entries are set using *\\_init\\_field*.

Examples:
GenericState(x)
GenericState(x, current\\_time = 1.0)
GenericState(x, current\\_score = 1.0)

See also: Stopping, NLPAtX
"""
mutable struct GenericState{S,T<:Union{AbstractFloat,AbstractVector}} <: AbstractState{S,T}

    x   :: T

    d   :: T
    res :: T

    #Current time
    current_time  :: Float64
    #Current score
    current_score :: S

    function GenericState(x             :: T,
                          current_score :: S;
                          d             :: T = _init_field(T),
                          res           :: T = _init_field(T),
                          current_time  :: Float64    = NaN) where {S, T <:AbstractVector}

      return new{S,T}(x, d, res, current_time, current_score)
   end
end

function GenericState(x             :: T;
                      d             :: T = _init_field(T),
                      res           :: T = _init_field(T),
                      current_time  :: Float64    = NaN,
                      current_score :: Union{T,eltype(T)} = _init_field(eltype(T))) where T <:AbstractVector

  return GenericState(x, current_score, d = d, res = res, current_time = current_time)
end

scoretype(typestate :: AbstractState{S,T}) where {S, T} = S
xtype(typestate :: AbstractState{S,T}) where {S, T} = T
"""
update!: generic update function for the State

`update!(:: AbstractState; convert = false, kwargs...)`

The function compares the kwargs and the entries of the State.
If the type of the kwargs is the same as the entry, then
it is updated.

Set kargs *convert* to true to update even incompatible types.

Examples:
update!(state1)
update!(state1, current\\_time = 2.0)
update!(state1, convert = true, current\\_time = 2.0)

See also: GenericState, reinit!, update\\_and\\_start!, update\\_and\\_stop!
"""
function update!(stateatx :: T; convert :: Bool = false, kwargs...) where T <: AbstractState

 fnames = fieldnames(T)
 for k ∈ keys(kwargs)
  #check if k is in fieldnames and type compatibility
  #two calls to getfield...
  if (k ∈ fnames) && typeof(kwargs[k]) <: typeof(getfield(stateatx, k))
   setfield!(stateatx, k, kwargs[k])
  end
 end

 return stateatx
end

#Ca serait cool d'avoir un shortcut en repérant certains keywords
#ou si il n'y a aucun keyword!
#function update!(stateatx :: T; x :: TT = stateatx.x) where {TS, TT, T <: AbstractState{TS,TT}}
# setfield!(stateatx, :x, x)
# return stateatx
#end

"""
\\_smart\\_update!: generic update function for the State without Type verification.

`_smart_update!(:: AbstractState; kwargs...)`

The function works exactly as update! without type and field verifications.
So, affecting a value to nothing or a different type will return an error.

See also: update!, GenericState, reinit!, update\\_and\\_start!, update\\_and\\_stop!
"""
function _smart_update!(stateatx :: T; kwargs...) where T <: AbstractState

 for k ∈ keys(kwargs)
   setfield!(stateatx, k, kwargs[k])
 end

 return stateatx
end
#https://github.com/JuliaLang/julia/blob/f3252bf50599ba16640ef08eb1e43c632eacf264/base/Base.jl#L34

function _update_time!(stateatx :: T, current_time :: Float64) where T <: AbstractState

 setfield!(stateatx, :current_time, current_time)

 return stateatx
end
"""
reinit!: function that set all the entries at *\\_init\\_field* except the mandatory *x*.

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
function reinit!(stateatx :: T, x :: Any; kwargs...)  where T <: AbstractState

#for k not in the kwargs
 for k ∈ setdiff(fieldnames(T), keys(kwargs))
   if k != :x setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k)))) end
 end

 return update!(stateatx; x = x, kwargs...)
end

function reinit!(stateatx :: AbstractState; kwargs...)
 return reinit!(stateatx, stateatx.x; kwargs...)
end

"""
\\_domain\\_check: returns true if there is a *NaN* or a *Missing* in the state entries (short-circuiting), false otherwise.

`_domain_check(:: AbstractState; kwargs...)`

Note:
- The fields given as keys in kwargs are not checked.

Examples:
\\_domain\\_check(state1)
\\_domain\\_check(state1, x = true)
"""
function _domain_check(stateatx :: T; kwargs...) where T <: AbstractState

 for k ∈ setdiff(fieldnames(T), keys(kwargs))

     gf = getfield(stateatx, k)
     if _check_nan_miss(gf)
         return true
     end
 end

 return false
end

_check_nan_miss(field :: Any) = false #Nothing or Counters
_check_nan_miss(field :: SparseMatrixCSC) = any(isnan, field.nzval) #because checking in sparse matrices is too slow
_check_nan_miss(field :: Union{AbstractVector,AbstractMatrix}) = any(isnan, field)
#We don't check for NaN's in Float as they are the _init_field
_check_nan_miss(field :: AbstractFloat) = ismissing(field)

import Base.copy
ex=:(_genobj(typ)=$(Expr(:new, :typ))); eval(ex)
function copy(state :: T) where T <: AbstractState

 #ex=:(_genobj(typ)=$(Expr(:new, :typ))); eval(ex)
 cstate = _genobj(T)
 #cstate = $(Expr(:new, typeof(state)))

 for k ∈ fieldnames(T)
  setfield!(cstate, k, deepcopy(getfield(state, k)))
 end

 return cstate
end

"""
compress_state!: compress State with the following rules.
- If it contains matrices and save_matrix is false, then the corresponding entries
are set to _init_field(typeof(getfield(stateatx, k)).
- If it contains vectors with length greater than max_vector_size, then the
corresponding entries are replaced by a vector of size 1 containing its pnorm-norm.
- If keep is true, then only the entries given in kwargs will be saved (the others are set to _init_field(typeof(getfield(stateatx, k))).
- If keep is false and an entry in the State is in the kwargs list, then it is put as _init_field(typeof(getfield(stateatx, k)) if possible.


`compress_state!(:: AbstractState; save_matrix :: Bool = false, max_vector_size :: Int = length(stateatx.x), pnorm :: Float64 = Inf, keep :: Bool = false, kwargs...)`

see also: copy, copy\\_compress\\_state, ListStates
"""
function compress_state!(stateatx        :: T;
                         save_matrix     :: Bool    = false,
                         max_vector_size :: Int     = length(stateatx.x),
                         pnorm           :: Float64 = Inf,
                         keep            :: Bool    = false,
                         kwargs...)  where T <: AbstractState

  for k ∈ fieldnames(T)
   if k ∈ keys(kwargs) && !keep
    try
        setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
    catch
        #nothing happens
    end
   end
   if k ∉ keys(kwargs) && keep
    try
        setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
    catch
        #nothing happens
    end
   end
   if typeof(getfield(stateatx, k)) <: AbstractVector
       katt = getfield(stateatx, k)
       if (length(katt) > max_vector_size)  setfield!(stateatx, k, [norm(katt, pnorm)]) end
   elseif typeof(getfield(stateatx, k)) <: Union{AbstractArray, AbstractMatrix}
       if save_matrix
        katt = getfield(stateatx, k)
        if maximum(size(katt)) > max_vector_size setfield!(stateatx, k, norm(getfield(stateatx, k))*ones(1,1)) end
       else #save_matrix is false
        setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
       end
   else
       #nothing happens
   end

  end

 return stateatx
end

"""
copy\\_compress\\_state: copy the State and then compress it.

`copy_compress_state(:: AbstractState; save_matrix :: Bool = false, max_vector_size :: Int = length(stateatx.x), pnorm :: Float64 = Inf, kwargs...)`

see also: copy, compress_state!, ListStates
"""
function copy_compress_state(stateatx        :: AbstractState;
                             save_matrix     :: Bool    = false,
                             max_vector_size :: Int     = length(stateatx.x),
                             pnorm           :: Float64 = Inf,
                             kwargs...)
 cstate = copy(stateatx)
 return compress_state!(cstate; save_matrix = save_matrix, max_vector_size = max_vector_size, pnorm = pnorm, kwargs...)
end
