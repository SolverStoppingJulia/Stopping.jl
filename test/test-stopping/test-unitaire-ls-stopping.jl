h = nothing
lsatx = LSAtT(0.0)

# Create the stopping object to test
stop = LS_Stopping(h, (x,y)-> armijo(x,y), lsatx, max_iter = 10)

# We tests different functions of stopping
OK = update_and_start!(stop, x = 1.0, g₀ = NaN, h₀ = NaN, ht = NaN)
@test OK == true
@test status(stop) == :DomainError
@test stop.current_state.x == 1.0

reinit!(stop)
@test stop.meta.domainerror == false
update!(lsatx, g₀ = 0.0, h₀ = 0.0, ht = 10.0)
@test (stop.current_state.g₀ == lsatx.g₀) && (lsatx.g₀ == 0.0)
@test (stop.current_state.h₀ == lsatx.h₀) && (lsatx.h₀ == 0.0)
@test (stop.current_state.ht == lsatx.ht) && (lsatx.ht == 10.0)
@test start!(stop) == false
@test status(stop) == :Unknown
#
@test update_and_stop!(stop, ht = -10.0) == true
@test status(stop) == :Optimal

reinit!(stop)
@test stop.meta.optimal == false
update!(stop.current_state, ht = 10.0)
# Check if _tired_check works
@test isnan(stop.meta.start_time) #default time value is NaN
start!(stop) #start initializes the start_time
@test !isnan(stop.meta.start_time)
@test !stop!(stop) #shall we stop?Not yet
#If we check stop! without a given start_time:
stop.meta.start_time = NaN
@test !(stop!(stop))
@test stop.meta.tired == false
@test status(stop) == :Unknown

# we can check _tired_check for max evals with NLPStopping

# Check if _unbounded_check! works
update!(stop.current_state, x = 1e100)
@test stop!(stop)

reinit!(stop, rstate = true, x = 1.0)
@test stop.current_state.x == 1.0
@test stop.current_state.ht == nothing

## _optimality_check and _null_test are tested with NLP
try
armijo(stop.pb, stop.current_state)
@test false #nothing entries in the stop
catch
@test true
end
try
wolfe(stop.pb, stop.current_state)
@test false #nothing entries in the stop
catch
@test true
end
try
armijo_wolfe(stop.pb, stop.current_state)
@test false #nothing entries in the stop
catch
@test true
end
update!(stop.current_state, h₀ = 1.0, ht = 0.0, g₀ = 1.0, gt = 0.0)
@test wolfe(stop.pb, stop.current_state) == 0.0
@test armijo_wolfe(stop.pb, stop.current_state) == 0.0
mutable struct Tpb
    d :: Number
end
stop.pb = Tpb(0.0)
@test shamanskii_stop(stop.pb, stop.current_state) == 0.0 #specific LineModel
@test goldstein(stop.pb, stop.current_state) >= 0.0

stop.optimality_check = (x,y) -> 0.0
stop.pb = ADNLPModel(x -> 0.0, [1.0])
stop.meta.max_f = -1
reinit!(stop.current_state, 0.0)
@test stop.current_state.ht == nothing
@test stop!(stop) == true
@test stop.current_state.ht == nothing
@test stop.meta.resources == true
