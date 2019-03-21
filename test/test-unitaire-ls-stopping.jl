using LinearAlgebra


h = nothing
lsatx = LSAtT(0.0)

# Create the stopping object to test
stop = LS_Stopping(h, (x,y)-> armijo(x,y), lsatx);

# We tests different functions of stopping
OK = update_and_start!(stop, x = 1.0)
@test OK == false
@test stop.current_state.x == 1.0


@test start!(stop) == false
OK2 = update_and_stop!(stop, ht = 10.0)
@test OK2 == false
@test stop.current_state.ht == 10.0
@test stop!(stop) == false

# Check if stalled checked works
update!(stop.current_state, dx = 0.0,  x = 1.0)
@test stop!(stop)
update!(stop.current_state, dx = 1.0, x = 0.0)
@test !(stop!(stop))
update!(stop.current_state, df = 0.0)
@test stop!(stop)
update!(stop.current_state, df = 1.0, x = 1.0)
@test !(stop!(stop))

# Check if _tired_check works
update!(stop.current_state, tmps = 0.0)
@test stop!(stop)
update!(stop.current_state, tmps = NaN)
@test !(stop!(stop))

# we can check _tired_check for max evals with NLPStopping

# Check if _unbounded_check! works
update!(stop.current_state, x = 1e100)
@test stop!(stop)


## _optimality_check and _null_test are tested with NLP
