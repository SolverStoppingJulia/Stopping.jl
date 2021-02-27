abstract type AbstractListofStates end

struct VoidListofStates <: AbstractListofStates end

"""
Type: list of States

Constructor:

`ListofStates(:: AbstractState)`

Note:
- If n != -1, then it stores at most n AbstractState.
- add additional methods following https://docs.julialang.org/en/v1/base/collections/
- ListofStates recursively handles sub-list of states as the attribute list is
an array of pair whose first component is a, AbstractState and the second
component is a ListofStates (or nothing).

Examples:
ListofStates(state)
ListofStates(state, n = 2)
ListofStates(-1)
ListofStates(-1, [(state1, VoidListofStates), (state2, VoidListofStates)], 2)
ListofStates(-1, [(state1, another_list)], 1)
"""
mutable struct ListofStates{T <: Array} <: AbstractListofStates

  n     :: Int #If length of the list is knwon, -1 if unknown
  i     :: Int #current index in the list/length
  list  :: T #Array{Tuple{AbstractState, AbstractListofStates},1}
                 #Tanj: \TODO Tuple instead of an Array would be better, I think

end

function ListofStates(n :: T) where T <: Int
  list = Array{Any}(nothing, max(n, zero(T)))
  i = 0
  return ListofStates(n, i, list)
end

function ListofStates(n :: T, list :: Array) where T <: Int
  i = length(list)
  return ListofStates(n, i, list)
end

function ListofStates(state :: AbstractState; n :: Int = -1, kwargs...)
  i =  1
  list = [(copy_compress_state(state; kwargs...), VoidListofStates())]
  return ListofStates(n, i, list)
end

"""
add\\_to\\_list!: add a State to the list of maximal size n.
If a n+1-th State is added, the first one in the list is removed.
The given is State is compressed before being added in the list (via State.copy\\_compress\\_state).

`add_to_list!(:: AbstractListofStates, :: AbstractState; kwargs...)`

Note: 
 -  kwargs are passed to the compress_state call.
 -  does nothing for `VoidListofStates`

see also: ListofStates, State.compress\\_state, State.copy\\_compress\\_state
"""
function add_to_list!(list :: AbstractListofStates, state :: AbstractState; kwargs...)

  if typeof(list.n) <: Int && list.n > 0 #If n is a natural number
    if list.i + 1 > list.n
      popfirst!(list.list) #remove the first item
      list.i = list.n
    else
      list.i += 1
    end
    cstate = copy_compress_state(state; kwargs...)
    push!(list.list, (cstate, VoidListofStates()))
  else
    push!(list.list, (copy_compress_state(state; kwargs...), VoidListofStates()))
    list.i += 1
  end

  return list
end

function add_to_list!(list :: VoidListofStates, state :: AbstractState; kwargs...)
  return list
end

import Base.length
"""
length: return the number of States in the list.

`length(:: ListofStates)`

see also: print, add_to_list!, ListofStates
"""
function length(list :: AbstractListofStates)
  return list.i
end

import Base.print
"""
print: output formatting. return a DataFrame.

`print(:: ListofStates; verbose :: Bool = true, print_sym :: Union{Nothing,Array{Symbol,1}})`

Note:
- set *verbose* to false to avoid printing.
- if *print_sym* is an Array of Symbol, only those symbols are printed. Note that
the returned DataFrame still contains all the columns.
- More information about DataFrame: http://juliadata.github.io/DataFrames.jl

see also: add\\_to\\_list!, length, ListofStates
"""
function print(list :: AbstractListofStates; 
               verbose :: Bool = true, 
               print_sym :: Union{Nothing,Array{Symbol,1}} = nothing)
   
  tab = zeros(0, length(list.list))#Array{Any,2}(undef, length(fieldnames(typeof(list.list[1,1]))))
  for k in fieldnames(typeof(list.list[1,1]))
    tab = vcat(tab, [getfield(i[1], k) for i in list.list]');
  end
  df = DataFrame(tab, :auto)

  if isnothing(print_sym)
    verbose && print(df)
  else
    verbose && print(df[!, print_sym])
  end

  return df
end

import Base.getindex
"""
`getindex(:: ListofStates, :: Int)`
`getindex(:: ListofStates, :: Int, :: Int)`

Example:
stop_lstt.listofstates.list[3]
stop_lstt.listofstates.list[3,1]
"""
function getindex(list :: AbstractListofStates, i :: Int)
  return list.list[i]
end

function getindex(list :: AbstractListofStates, i :: Int, j :: Int)
  return list.list[i][j]
end
