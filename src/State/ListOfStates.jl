"""
Type: list of States

Constructor:

`ListStates(:: AbstractState)`

Note:
- If n != -1, then it stores at most n AbstractState.
- add additional methods following https://docs.julialang.org/en/v1/base/collections/

Examples:
ListStates(state)
ListStates(state, n = 2)
ListStates(n = -1, list = [state1, state2], i = 2)
"""
mutable struct ListStates

  n     :: Int #If length of the list is knwon, -1 if unknown
  i     :: Int #current index in the list/length
  list :: Array #list of States

end

function ListStates(n :: Int; list :: Array = Array{Any}(nothing, n), i :: Int = 0)
  return ListStates(n, i, list)
end

function ListStates(state :: AbstractState; n :: Int = -1, kwargs...)
    i =  1
    list = [copy_compress_state(state; kwargs...)]
  return ListStates(n, i, list)
end

"""
add\\_to\\_list!: add a State to the list of maximal size n.
If a n+1-th State is added, the first one in the list is removed.
The given is State is compressed before being added in the list (via State.copy\\_compress\\_state).

`add_to_list!(:: ListStates, :: AbstractState; kwargs...)`

Note: kwargs are passed to the compress_state call.

see also: ListStates, State.compress\\_state, State.copy\\_compress\\_state
"""
function add_to_list!(list :: ListStates, state :: AbstractState; kwargs...)

 if typeof(list.n) <: Int && list.n > 0 #If n is a natural number
  if list.i + 1 > list.n
      popfirst!(list.list) #remove the first item
      list.i = list.n
  else
      list.i += 1
  end
  cstate = copy_compress_state(state; kwargs...)
  push!(list.list, cstate)
 else
  push!(list.list, copy_compress_state(state; kwargs...))
  list.i += 1
 end

 return list
end

import Base.length
"""
length: return the number of States in the list.

`length(:: ListStates)`

see also: print, add_to_list!, ListStates
"""
function length(list :: ListStates)
 return list.i
end

import Base.print
"""
print: output formatting. return a DataFrame.

`print(:: ListStates; verbose :: Bool = true, print_sym :: Union{Nothing,Array{Symbol,1}})`

Note:
- set *verbose* to false to avoid printing.
- if *print_sym* is an Array of Symbol, only those symbols are printed. Note that
the returned DataFrame still contains all the columns.
- More information about DataFrame: http://juliadata.github.io/DataFrames.jl

see also: add\\_to\\_list!, length, ListStates
"""
function print(list :: ListStates; verbose :: Bool = true, print_sym :: Union{Nothing,Array{Symbol,1}} = nothing)

   df = DataFrame()

   for k in fieldnames(typeof(list.list[1]))
       df[!,k] = [getfield(i, k) for i in list.list]
   end

   if print_sym == nothing
    verbose && print(df)
   else
    verbose && print(df[!, print_sym])
   end

 return df
end

import Base.getindex
"""
`getindex(:: ListStates, :: Int)`

Example:
stop_lstt.listofstates.list[3]
"""
function getindex(list :: ListStates, i :: Int)
    return list.list[i]
end
