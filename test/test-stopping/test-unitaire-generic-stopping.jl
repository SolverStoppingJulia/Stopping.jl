@testset "Test GenericStopping" begin
    
    include("rosenbrock.jl")
    #We build a first stopping: to test the change of tol_check function
    x0 = ones(6)
    state0 = GenericState(x0)
    stop0 = GenericStopping(rosenbrock, state0, tol_check = (atol,rtol,opt0) -> atol + rtol * opt0, list = ListStates(state0) )

    show(stop0)

    meta = StoppingMeta(tol_check = (atol,rtol,opt0) -> atol + rtol * opt0, list = ListStates(state0))
    stop0_meta = GenericStopping(rosenbrock, meta, state0)
    stop0_src = GenericStopping(rosenbrock, meta, cheap_stop_remote_control(), state0)
    stop0_src_without_meta = GenericStopping(rosenbrock, StopRemoteControl(), state0, 
                                             tol_check = (atol,rtol,opt0) -> atol + rtol * opt0, 
                                             list = ListStates(state0) )

    @test start!(stop0) == true #opt0 = Inf as default meta.optimality_check returns Inf, so any point is optimal.
    @test status(stop0) == :Optimal
    #We now illustrate the impact of the choice of the norm for the unboundedness of the iterate
    stop0.meta.unbounded_x = sqrt(6)
    stop!(stop0)
    @test status(stop0, list = true) == [:Optimal] #ok as ||x||_\infty = 1 < sqrt(6)

    #We now test that stop! verifies that:
    #- there are no NaN in the score
    #- if the listofstates != nothing, stop! increases the list of states with the current_state.
    stop0.meta.optimality_check = (a,b) -> NaN
    #stop0.listofstates = ListStates(state0)
    stop!(stop0)
    @test :DomainError in status(stop0, list = true)
    @test length(stop0.listofstates) == 3

    #Initialize a GenericStopping by default
    stop_def = GenericStopping(rosenbrock, x0, atol = 1.0)
    @test stop_def.current_state.x == x0
    @test stop_def.meta.atol == 1.0
    @test start!(stop_def) == true

    #We build a first stopping:
    x0 = ones(6)
    state = GenericState(x0)
    stop = GenericStopping(rosenbrock, state, max_time = 2.0, rtol = 0.0)
    #If rtol != 0, any point is a solution as optimality0 = Inf.

    @show(stop)

    @test start!(stop) == false
    @test stop.meta.start_time != NaN
    @test stop!(stop) == false

    #We build a substopping:
    x1 = zeros(6)
    state1 = GenericState(x1)
    ABigInt = 100000000000000000 #to avoid the stop by counting stop calls
    substop = GenericStopping(rosenbrock, state1, main_stp = stop, max_iter = ABigInt, rtol = 0.0 )
    substop.stop_remote = StopRemoteControl()
    #If rtol != 0, any point is a solution as optimality0 = Inf.

    show(substop)

    @test start!(substop) == false
    @test stop!(substop) == false

    function infinite_algorithm(stp :: AbstractStopping)

        x0 = stp.current_state.x
        smallest_f = stp.pb(x0) #stp.pb is a function here

        if !(typeof(stp.main_stp) <: VoidStopping)
        start!(stp.main_stp) #start the time counter of the upper loop
        end
        OK = start!(stp)

        while !OK

            x = 10*rand(length(x0))
            smallest_f = min(smallest_f, stp.pb(x))

            OK = update_and_stop!(stp, x = x) #check the resources of stp AND stp.main_stp

        end
        return smallest_f
    end

    #Solve the problem with substopping
    res = infinite_algorithm(substop)

    #The algorithm stopped because the time limit of stop is attained
    @test status(substop) == :ResourcesOfMainProblemExhausted
    @test status(stop)    == :TimeLimit
    @test substop.current_state.x != x1

    #Reinitialize the stop
    #test the reinit!
    timesave, opt0save = stop.meta.start_time, stop.meta.optimality0
    reinit!(stop)
    @test timesave != stop.meta.start_time
    @test opt0save != stop.meta.optimality0

    #Solve again the problem
    res2 = infinite_algorithm(stop)

    #The algorithm stopped as it attained the iteration limit (stop! call)
    @test status(stop) == :IterationLimit

    reinit!(stop)
    reinit!(substop)

    #
    # Test the triple sub-Stopping now:
    #
    subsubstop = GenericStopping(rosenbrock, state1, main_stp = substop, max_iter = ABigInt, rtol = 0.0 )
    #If rtol != 0, any point is a solution as optimality0 = Inf.

    #Solve again the problem
    start!(stop) #initialize here as infinite_algorithm has 2 "loops" only
    res3 = infinite_algorithm(subsubstop)

    @test status(stop) == :TimeLimit #stop because of the main main stopping.
    @test status(substop) == :ResourcesOfMainProblemExhausted
    @test status(subsubstop) == :ResourcesOfMainProblemExhausted

    stop.meta.infeasible = true
    @test status(stop, list = true) == [:TimeLimit, :Infeasible]

    try
        fill_in!(stop, zeros(5))
        @test false
    catch
        @test true
    end

end