## Stopping's attributes ID: StopRemoteControl

Usual instances of `AbstractStopping` contains a `StopRemoteControl <: AbstractStopRemoteControl` (`stp.stop_remote`), which controls the various checks run by the functions `start!` and `stop!`. An instance of `StopRemoteControl` contains:
- `unbounded_and_domain_x_check :: Bool`
- `domain_check                 :: Bool`
- `optimality_check             :: Bool`
- `infeasibility_check          :: Bool`
- `unbounded_problem_check      :: Bool`
- `tired_check                  :: Bool`
- `resources_check              :: Bool`
- `stalled_check                :: Bool`
- `iteration_check              :: Bool`
- `main_pb_check                :: Bool`
- `user_check                   :: Bool`
- `user_start_check             :: Bool`
- `cheap_check                  :: Bool`
Only the last attributes, `cheap_check`, is not related with a specific check. Set as `true`, it stopped whenever one of the checks is successful and the algorithm needs to stop. It is `false` by default. All the other entries are set as `true` by default, i.e.
```julia
src = StopRemoteControl() #initializes a remote control with all the checks on.
```
In order to remove some checks, it suffices to use keywords:
```julia
src = StopRemoteControl(tired_check = false, iteration_check = false) #remove time and iteration checks.
```

### FAQ: Is there performance issues with all these checks?
Assuming that `x` is a vector of length `n`, some of these checks are indeed in O(n), which can be undesirable for some applications. In this case, you can either initialize a "cheap" remote control as follows
```julia
src = cheap_stop_remote_control() #initialize a StopRemoteControl with 0(n) checks set as false
```
or deactivate the tests by hand as shown previously.

### FAQ: How can I fine-tune these checks?
All these checks can be fine-tuned by selecting entries in the `StoppingMeta`.
