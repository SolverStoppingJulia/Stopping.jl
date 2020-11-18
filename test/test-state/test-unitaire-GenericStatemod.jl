#unitary test GenericStatemod
x0 = ones(6)
state0 = GenericState(x0)

@test state0.current_time == nothing #Default value of start_time is void
@test state0.current_score == nothing
x1 = [1.0]
update!(state0, x = x1) #Check the update of state0
@test state0.x == x1
@test state0.current_time == nothing #start_time must be unchanged
@test state0.current_score == nothing

update!(state0, current_time = 1.0)
@test state0.x == x1 #must be unchanged
@test state0.current_time == 1.0
@test state0.current_score == nothing

reinit!(state0, x0)
@test state0.x == x0
@test state0.current_time == nothing
@test state0.current_score == nothing

update!(state0, x = x1)
reinit!(state0, current_time = 0.5)
@test state0.x == x1
@test state0.current_time == 0.5
@test state0.current_score == nothing
