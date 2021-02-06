@testset "Test Line-Search Stopping" begin

    lsatx = LSAtT(0.0)

    pb = ADNLPModel(x -> 0., ones(1)) #NLPModels of size 1

    @test pb.meta.nvar == 1
    @test unconstrained(pb)

    # Create the stopping object to test
    stop = NLPStopping(pb, lsatx, max_iter = 10, optimality_check = armijo)
    stop_without_check = NLPStopping(pb, lsatx, max_iter = 10)
    @test stop_without_check.meta.optimality_check == stop.meta.optimality_check

    meta = StoppingMeta(max_iter = 10, optimality_check = (x,y)-> armijo(x,y))
    stop_meta = NLPStopping(pb, meta, lsatx)
    stop_meta_src = NLPStopping(pb, meta, StopRemoteControl(), lsatx)

    @test stop_meta_src.stop_remote.cheap_check == stop.stop_remote.cheap_check
    @test stop_meta.stop_remote.cheap_check == stop.stop_remote.cheap_check

    # We tests different functions of stopping
    #This is no longer true as NaN is the _init_field for the quantities
    #OK = update_and_start!(stop, x = 1.0, g₀ = NaN, h₀ = NaN, ht = NaN)
    #@test OK == true
    #@test status(stop) == :DomainError
    #@test stop.current_state.x == 1.0

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
    @test isnan(stop.current_state.ht)

    ## _optimality_check! and _null_test are tested with NLP
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

    @test shamanskii_stop(stop.pb, stop.current_state) == 0.0 #specific LineModel
    @test goldstein(stop.pb, stop.current_state) >= 0.0

    stop.meta.optimality_check = (x,y) -> 0.0


    #stop.pb = ADNLPModel(x -> 0.0, [1.0]) #Can't do that
    #stop.meta.max_f = -1
    #reinit!(stop.current_state, 0.0)
    #@test stop.current_state.ht == nothing
    #@test stop!(stop) == true
    #@test stop.current_state.ht == nothing
    #@test stop.meta.resources == true

    #We now check the tol_check_neg function
    #stop.meta.optimality_check = (x,y) -> y.ht
    #stop.current_state.ht = 0.0
    #@test stop.current_state.ht == 0.0
    #stop.meta.tol_check = (a,b,c) -> 1.0
    #@test Stopping._null_test(stop, Stopping._optimality_check!(stop))
    #stop.meta.tol_check_neg = (a,b,c) -> 0.5
    #@test !(Stopping._null_test(stop, Stopping._optimality_check!(stop)))

end
