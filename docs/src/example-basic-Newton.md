## Example I: Stopping in the flow

We present here a typical iterative algorithm to illustrate how to use Stopping.

```julia
function rand_solver(stp :: AbstractStopping, x0 :: AbstractVector)

    x = x0
    #First, call update and start! to check optimality and set an initial configuration
    update(stp, x = x)
    OK = start!(stp)
    # which may be combined in a single call equivalent
    #OK = update_and_start!(stp, x = x)

    while !OK
        #Run some computations and update the iterate
        d = rand(length(x))
        x += d

        #Update the State and call the Stopping with stop!
        OK = update_and_stop!(stp, x = x, d = d)
    end

    return stp
end
```
This example shows the most basic features of Stopping. It does many checks for you. In this innocent-looking algorithm, the call to `update_and_start!` and `update_and_stop!` will verifies unboundedness of `x`, the time spent in the algorithm, the number of iterations (= number of call to `stop!`), and the domain of `x` (in case some of its components become `NaN` for instance).

### FAQ: How can I disable some checks done by Stopping?
The native instances of `AbstractStopping` available in Stopping.jl all contain an attribute `stop_remote`.
This is a remote control for Stopping's checks.
```julia
typeof(stp.stop_remote) <: StopRemoteControl #stop_remote is an instance of StopRemoteControl
```
This attributes contains boolean values for each check done by Stopping, see
```julia
fieldnames(stp.stop_remote) #get all the attributes of the remote control
```
For instance, we can remove the unboundedness and domain check done on `x` by setting:
```julia
stp.stop_remote = StopRemoteControl(unbounded_and_domain_x_check = false)
```
