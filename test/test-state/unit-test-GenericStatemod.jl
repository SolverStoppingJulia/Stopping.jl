#unitary test GenericStatemod
x0 = ones(6)
state0 = GenericState(x0)

show(state0)

@test scoretype(state0) == Float64
@test xtype(state0) == Array{Float64,1}

@test isnan(state0.current_time) #Default value of start_time is void
@test isnan(state0.current_score)
x1 = [1.0]
update!(state0, x = x1) #Check the update of state0
@test state0.x == x1
@test isnan(state0.current_time) #start_time must be unchanged
@test isnan(state0.current_score)

update!(state0, current_time = 1.0)
@test state0.x == x1 #must be unchanged
@test state0.current_time == 1.0
@test isnan(state0.current_score)

reinit!(state0, x0)
@test state0.x == x0
@test isnan(state0.current_time)
@test isnan(state0.current_score)

update!(state0, x = x1)
reinit!(state0, current_time = 0.5)
@test state0.x == x1
@test state0.current_time == 0.5
@test isnan(state0.current_score)

#Test _init_field
@test _init_field(typeof(zeros(2,2))) == zeros(0,0)
@test _init_field(SparseMatrixCSC{Float64,Int64}) == spzeros(0,0)
@test _init_field(typeof(zeros(2))) == zeros(0)
@test _init_field(typeof(sparse(zeros(2)))) == spzeros(0)
@test isnan(_init_field(BigFloat))
@test isnan(_init_field(typeof(1.)))
@test isnan(_init_field(Float32))
@test isnan(_init_field(Float16))
@test _init_field(Nothing) == nothing
@test ismissing(Main.Stopping._init_field(Missing))
@test !_init_field(typeof(true))
@test _init_field(typeof(1)) == -9223372036854775808

#_check_nan_miss
@test !Stopping._check_nan_miss(nothing)
@test !Stopping._check_nan_miss(Counters())
@test !Stopping._check_nan_miss(spzeros(0))
@test !Stopping._check_nan_miss(zeros(0))
@test !Stopping._check_nan_miss(missing)
@test !Stopping._check_nan_miss(spzeros(0,0))
