#unitary test GenericStatemod
x0 = ones(6)
state0 = GenericState(x0)

@test state0.start_time == nothing #Default value of start_time is void
x1 = [1.0]
update!(state0, x = x1) #Check the update of state0
@test state0.x == x1
@test state0.start_time == nothing #start_time must be unchanged

update!(state0, tmps = 1.0)
@test state0.x == x1 #must be unchanged
@test state0.start_time == 1.0
