## Stopping-work flow

The table below depict the various checks done by the function `stop!` and their connection with the `meta`, `current_state` and `current_state`. The *function* entry correspond to the function used internally by `stop!`, they can be imported and redifined to be adapted for a specific problem, for instance `NLPStopping` for `NLPModels`. The *remote\_control* entry corresponds to the attribute in the `remote_control` that could be set as true/false to activate/deactivate this check. The *meta\_status* gives the attribute in the `meta` with the check's answer. Finally the last column corresponds to entries in the `meta` parametrizing this check.

| Check description                         | Function                          | remote control  | meta statuses | meta tolerances |
| ----------------------------------------- | ---------------------------------:| ---------------:| -----------:| ----:|
| Check unboundedness and the domain of `x` | _unbounded_and_domain_x_check!    | unbounded_and_domain_x_check | domainerror and unbounded_problem_x | stp.meta.unbounded_x|
| Check the domain in state (NaN's ...)     | _domain_check                     | domain_check                 | domainerror | |
| Check optimality                          | _optimality_check! and _null_test | optimality_check             | optimal | See *how to check optimality with Stopping* |
| Check for infeasibility                   | _infeasibility_check!             | infeasibility_check          | infeasible | |
| Check for unboundedness in problem values | _unbounded_problem_check!         | unbounded_problem_check      | unbounded_problem | |
| Check time-limit                          | _tired_check!                     | tired_check                  | tired | start_time, max_time |
| Check for limits in resources             | _resources_check!                 | resources_check              | resources ||
| Check if algo is stalling                 | _stalled_check!                   | stalled_check                | stalled ||
| Count the number of stop! and limits      | _iteration_check!                 | iteration_check              | iteration_limit | max_iter |
| Check if the main_stp stops               | _main_pb_check!                   | main_pb_check                | main_pb ||
| Callback user check                       | _user_check!                      | user_check                   | stopbyuser | user_check_func! |

### FAQ: Is Stopping initializing `meta.start_time` on its own?

Yes, it does when you call `start!` as well as `optimality0` if `start!` check the optimality.

### FAQ: How to set-up the `user_check`?

Stopping call the `user_check_func!` defined in the `meta`.

### FAQ: How does Stopping check the optimality?

See the tutorial on this topic.
