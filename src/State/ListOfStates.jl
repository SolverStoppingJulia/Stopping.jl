"""
Type: list of States

Constructor:

`ListStates(:: AbstractState)`

Note:
- add additional methods following https://docs.julialang.org/en/v1/base/collections/
- If n != -1, then store at most n AbstractState.
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
add_to_list!: add a State to the list. In the case, where n != -1, the first one is deleted before update.

`add_to_list!(: ListStates, :: AbstractState; kwargs...)`

Note: kwargs are passed to the compress_state call.
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
  #list.list[list.i] = cstate
  push!(list.list, cstate)
 else
  push!(list.list, copy_compress_state(state; kwargs...))
  list.i += 1
 end

 return list
end

import Base.length
"""
length: return the number of States in the list

`length(: ListStates)`
"""
function length(list :: ListStates)
 return list.i
end

import Base.print
"""
print: output formatting. return a DataFrame

`print(: ListStates)`

Note: set verbose to false to avoid printing
"""
function print(list :: ListStates; verbose :: Bool = true)

   #foreach(x -> println([norm(x.x), x.current_time]), list.list)
   df = DataFrame()
   for k in fieldnames(typeof(list.list[1]))
       df[!,k] = [getfield(i, k) for i in list.list]
   end
 verbose && print(df)
 return df
end
