## How to Stop 2

The Stopping structure eases the implementation of algorithms and the
stopping criterion.
We illustrate here the features of Stopping when the algorithm is used a
subStopping.

The Julia file corresponding to this tutorial can be found [here](https://github.com/Goysa2/Stopping.jl/tree/master/test/examples/howtostop-2.jl).

```
using Test, Stopping
```

Assume we want to solve "pb" starting from "x0" and solving at each step
of the algorithm the subproblem "subpb".
 We can use this additional info to improve the stopping criterion.
```
x0 = ones(2)
pb = nothing
subpb = nothing
subsubpb = nothing
```

Initialize a Stopping for the main pb
```
main_stop = GenericStopping(pb, x0)
```
We can then, initialize another stopping to the subproblem, and providing
the main_stop as a keyword argument:
```
sub_stop = GenericStopping(subpb, x0, main_stp = main_stop, tol_check = (atol, rtol, opt0) -> atol)
```
Note that by default main_stp is void
```
@test main_stop.main_stp == nothing
```

The only difference appears in the event of a call to stop!, which now also
check the time and resources of the main_pb.
```
OK = start!(sub_stop)
@test OK == false #no reason to stop just yet.
```

Assume time is exhausted for the main_stop
```
main_stop.meta.start_time = 0.0 #force a timing failure in the main problem
stop!(sub_stop)

@test status(sub_stop, list = true) == [:ResourcesOfMainProblemExhausted]
@test sub_stop.meta.tired == false
@test sub_stop.meta.main_pb == true
```

The same applies if there is now a third subproblem
```
reinit!(main_stop)
reinit!(sub_stop)
subsub_stop = GenericStopping(subsubpb, x0, main_stp = sub_stop, tol_check = (atol, rtol, opt0) -> atol)
main_stop.meta.start_time = 0.0 #force a timing failure in the main problem
stop!(subsub_stop)

@test status(subsub_stop, list = true) == [:ResourcesOfMainProblemExhausted]
@test subsub_stop.meta.tired   == false
@test subsub_stop.meta.main_pb == true
@test status(sub_stop, list = true) == [:ResourcesOfMainProblemExhausted]
@test sub_stop.meta.tired   == false
@test sub_stop.meta.main_pb == true
@test status(main_stop, list = true) == [:Tired]
@test main_stop.meta.tired   == true
@test main_stop.meta.main_pb == false
```
