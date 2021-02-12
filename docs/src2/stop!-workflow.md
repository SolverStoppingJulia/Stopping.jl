## Stopping-work flow

The table below depict the various checks done by the function `stop!` and their connection with the `meta`, `current_state` and `current_state`.

| Check description                         | Function                          | remote control  | meta status | meta |
| ----------------------------------------- | ---------------------------------:| ---------------:| -----------:| ----:|
| Check unboundedness and the domain of `x` | _unbounded_and_domain_x_check!    | unbounded_and_domain_x_check | domainerror and unbounded_problem_x | |
| Check the domain in state (NaN's ...)     | _domain_check                     | domain_check                 | domainerror | |
| Check optimality                          | _optimality_check! and _null_test | optimality_check             | optimal | |
| Check for infeasibility                   | _infeasibility_check!             | infeasibility_check          | infeasible | |
| Check for unboundedness in problem values | _unbounded_problem_check!         | unbounded_problem_check      | unbounded_problem | |
| Check time-limit                          | _tired_check!                     | tired_check                  | tired | |
| Check for limits in resources             | _resources_check!                 | resources_check              | resources ||
| Check if algo is stalling                 | _stalled_check!                   | stalled_check                | stalled ||
| Count the number of stop! and limits      | _iteration_check!                 | iteration_check              | iteration_limit ||
| Check if the main_stp stops               | _main_pb_check!                   | main_pb_check                | main_pb ||
| Callback user check                       | _user_check!                      | user_check                   | stopbyuser ||
