## Do you speak Stopping?

When using a Stopping-compatible algorithm, a.k.a an algorithm that takes a Stopping as an input and return it,
the user is free to explore the results and influence the execution of the algorithm.

First, we need to create a Stopping.
```julia
x = ones(10)
problem = nothing #or your instance
stp = GenericStopping(pb, x, max_time = 10.) #short-cut initializing a `GenericState` and a `StoppingMeta`
@show stp.meta.max_time == 10. #by default the `kwargs...` are passed to the meta.
```
One can also creates separately a state and a meta to form a Stopping:
```julia
state = GenericState(x)
meta  = StoppingMeta(max_time = 10.)
stp = GenericStopping(pb, meta, state)
```
Once the `Stopping` has been initialized, we can call the algorithm and exploit the output.
```julia
stp = rand_solver(stp, x) #call your favorite solver
```
To get the reason why the algorithm stopped we use `status`.
```julia
status(stp) #or `status(stp, rlist = true)` to have the complete list.
```
The solution as well as problem-related information can be accessed from the state.
```julia
sol = stp.current_state.x
```

### FAQ: How do I know the entries in the Stopping, State or the Meta?
You can use Julia's build-in `fieldnames` function.
```julia
fieldnames(stp)
fieldnames(stp.current_state)
fieldnames(stp.meta)
```
