function rosenbrock(x)

    n = 6;

    # Initializations
    f = 0

    evenIdx = 2:2:n
    oddIdx  = 1:2:(n-1)

    f1  = x[evenIdx] .- x[oddIdx].^2
    f2  = 1 .- x[oddIdx]

    # Function
    f   = sum( f1.^2 .+ f2.^2 )

    return f
end

#We build a first stopping:
x0 = ones(6)
state = GenericState(x0)
stop = GenericStopping(rosenbrock, state, max_time = 2.0)

@test start!(stop) == false
@test stop.meta.start_time != NaN
@test stop!(stop) == false

#We build a substopping:
x1 = zeros(6)
state1 = GenericState(x1)
ABigInt = 100000000000000000 #to avoid the stop by counting stop calls
substop = GenericStopping(rosenbrock, state1, main_stp = stop, max_iter = ABigInt )

@test start!(substop) == false
@test stop!(substop) == false

function infinite_algorithm(stp :: AbstractStopping)

 x0 = stp.current_state.x
 smallest_f = stp.pb(x0) #stp.pb is a function here

 if stp.main_stp != nothing
  start!(stp.main_stp) #start the time counter of the upper loop
 end
 OK = start!(stp)

 while !OK

     x = 10*rand(length(x0))
     smallest_f = min(smallest_f, stp.pb(x))

     OK = stop!(stp) #check the resources of stp AND stp.main_stp

 end
 return smallest_f
end

res = infinite_algorithm(substop)

@test status(substop) == :ResourcesOfMainProblemExhausted
@test status(stop)    == :Tired
