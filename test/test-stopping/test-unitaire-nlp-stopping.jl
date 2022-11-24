@testset "Test ListofStates in NLPStopping constructors" begin
  nlp = ADNLPModel(x -> sum(x), zeros(5))
  n = 7
  stp = NLPStopping(nlp, n_listofstates = n)
  @test get_list_of_states(stp).n == 7

  nlp_at_x = NLPAtX(zeros(5))
  stp = NLPStopping(nlp, nlp_at_x, n_listofstates = n)
  @test get_list_of_states(stp).n == 7

  meta = StoppingMeta()
  stp = NLPStopping(nlp, meta, nlp_at_x, n_listofstates = n)
  @test get_list_of_states(stp).n == 7

  stop_remote = StopRemoteControl()
  stp = NLPStopping(nlp, meta, stop_remote, nlp_at_x, n_listofstates = n)
  @test get_list_of_states(stp).n == 7
end

@testset "Test NLP Stopping unconstrained" begin

  # We create a simple function to test
  A = rand(5, 5)
  Q = A' * A

  f(x) = x' * Q * x
  nlp = ADNLPModel(f, zeros(5))

  stp_error = NLPStopping(nlp)
  @test stop!(stp_error) # returns a warning "KKT needs stp.current_state.gx to be filled-in"
  @test status(stp_error) == :Infeasible

  nlp_at_x = NLPAtX(zeros(5))
  meta = StoppingMeta(
    optimality0 = 0.0,
    max_cntrs = init_max_counters(),
    optimality_check = (x, y) -> unconstrained_check(x, y),
  )
  stop_nlp = NLPStopping(nlp, meta, nlp_at_x)

  @test get_pb(stop_nlp) == stop_nlp.pb
  @test get_meta(stop_nlp) == stop_nlp.meta
  @test get_remote(stop_nlp) == stop_nlp.stop_remote
  @test get_state(stop_nlp) == stop_nlp.current_state
  @test get_main_stp(stop_nlp) == stop_nlp.main_stp
  @test get_list_of_states(stop_nlp) == stop_nlp.listofstates
  @test get_user_struct(stop_nlp) == stop_nlp.stopping_user_struct

  src = StopRemoteControl()
  stop_nlp_src = NLPStopping(nlp, meta, src, nlp_at_x)

  a = zeros(5)
  fill_in!(stop_nlp, a)

  # we make sure that the fill_in! function works properly
  @test obj(nlp, a) == stop_nlp.current_state.fx
  @test grad(nlp, a) == stop_nlp.current_state.gx
  @test stop_nlp.meta.optimality0 == 0.0

  # we make sure the optimality check works properly
  @test stop!(stop_nlp)
  @test stop_nlp.current_state.current_score ==
        unconstrained_check(stop_nlp.pb, stop_nlp.current_state)
  # we make sure the counter of stop works properly
  @test stop_nlp.meta.nb_of_stop == 1

  reinit!(stop_nlp, rstate = true, x = ones(5))
  @test stop_nlp.current_state.x == ones(5)
  @test isnan(stop_nlp.current_state.fx)
  @test stop_nlp.meta.nb_of_stop == 0

  #We know test how to initialize the counter:
  test_max_cntrs = init_max_counters(obj = 2)
  stop_nlp_cntrs = NLPStopping(nlp, max_cntrs = test_max_cntrs)
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_obj] == 2
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_grad] == typemax(Int)
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_sum] == typemax(Int)

  reinit!(stop_nlp.current_state)
  @test unconstrained_check(stop_nlp.pb, stop_nlp.current_state) >= 0.0
  reinit!(stop_nlp.current_state)

  #We now test the _unbounded_problem_check:
  @test stop_nlp.pb.meta.minimize #we are minimizing
  stop_nlp.current_state.fx = -1.0e50 #default meta.unbounded_threshold
  stop!(stop_nlp)
  @test :UnboundedPb in status(stop_nlp, list = true) # the problem is unbounded as fx <= - 1.0e50
  stop_nlp.meta.unbounded_pb = false #reinitialize
  #Let us now consider a maximization problem:
  nlp_max = ADNLPModel(f, zeros(5), minimize = false)
  stop_nlp.pb = nlp_max
  @test !stop_nlp.pb.meta.minimize
  stop!(stop_nlp)
  @test !(:UnboundedPb in status(stop_nlp, list = true)) # the problem is NOT unbounded as fx <= 1.0e50

  nlp_bnd = ADNLPModel(f, zeros(5), lvar = zeros(5), uvar = zeros(5))

  stp_error = NLPStopping(nlp_bnd)
  @test stop!(stp_error) # returns a warning "KKT needs stp.current_state.mu to be filled-in"
  @test status(stp_error) == :Infeasible

  stop_bnd = NLPStopping(nlp_bnd)
  fill_in!(stop_bnd, zeros(5))
  @test KKT(stop_bnd.pb, stop_bnd.current_state) == 0.0
  reinit!(stop_bnd.current_state)
  @test optim_check_bounded(stop_bnd.pb, stop_bnd.current_state) == 0.0

  stop_bnd = NLPStopping(nlp_bnd, optimality_check = (x, y) -> NaN)
  start!(stop_bnd)
  @test stop_bnd.meta.domainerror == true
  reinit!(stop_bnd, rcounters = true)
  @test neval_grad(stop_bnd.pb) == 0
  @test stop_bnd.meta.domainerror == false
  stop!(stop_bnd)
  @test stop_bnd.meta.domainerror == true

  stop_bnd = NLPStopping(nlp_bnd, optimality_check = (x, y) -> 0.0)
  reinit!(stop_bnd, rcounters = true)
  @test neval_grad(stop_bnd.pb) == 0
  fill_in!(stop_bnd, zeros(5), mu = ones(5), lambda = zeros(0))
  @test stop_bnd.current_state.mu == ones(5)
  @test stop_bnd.current_state.lambda == zeros(0)

  update!(stop_bnd.current_state, fx = NaN, current_score = 0.0)
  stop!(stop_bnd)
  @test !isnan(stop_bnd.current_state.fx)

  # Test with a different type
  T = Float16
  pb16 = ADNLPModel(x -> zero(T), ones(T, 5), zeros(T, 5), zeros(T, 5))
  stp16 = NLPStopping(pb16)
  update!(stp16, mu = zeros(T, 5), gx = zeros(T, 5))
  @test typeof(unconstrained_check(stp16.pb, stp16.current_state)) == T
  @test typeof(optim_check_bounded(stp16.pb, stp16.current_state)) == T
  @test typeof(KKT(stp16.pb, stp16.current_state)) == T
end
