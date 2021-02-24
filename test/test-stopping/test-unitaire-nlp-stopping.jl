@testset "Test NLP Stopping unconstrained" begin

  # We create a simple function to test
  A = rand(5, 5);
  Q = A' * A

  f(x) = x' * Q * x
  nlp = ADNLPModel(f, zeros(5))
  nlp_at_x = NLPAtX(zeros(5))
  meta = StoppingMeta(optimality0 = 0.0, 
                      max_cntrs   = Stopping._init_max_counters(),
                      optimality_check = (x,y) -> unconstrained_check(x,y))
  stop_nlp = NLPStopping(nlp, meta, nlp_at_x)

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
  @test stop_nlp.current_state.current_score == unconstrained_check(stop_nlp.pb, stop_nlp.current_state)
  # we make sure the counter of stop works properly
  @test stop_nlp.meta.nb_of_stop == 1

  reinit!(stop_nlp, rstate = true, x = ones(5))
  @test stop_nlp.current_state.x == ones(5)
  @test isnan(stop_nlp.current_state.fx)
  @test stop_nlp.meta.nb_of_stop == 0

  #We know test how to initialize the counter:
  test_max_cntrs = Stopping._init_max_counters(obj = 2)
  stop_nlp_cntrs = NLPStopping(nlp, max_cntrs = test_max_cntrs)
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_obj] == 2
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_grad] == 20000
  @test stop_nlp_cntrs.meta.max_cntrs[:neval_sum] == 20000*11

  reinit!(stop_nlp.current_state)
  @test unconstrained_check(stop_nlp.pb, stop_nlp.current_state) >= 0.0
  reinit!(stop_nlp.current_state)
  @test unconstrained2nd_check(stop_nlp.pb, stop_nlp.current_state) >= 0.0
  @test stop_nlp.current_state.Hx != nothing

  #We now test the _unbounded_problem_check:
  @test stop_nlp.pb.meta.minimize #we are minimizing
  stop_nlp.current_state.fx = - 1.0e50 #default meta.unbounded_threshold
  stop!(stop_nlp)
  @test :UnboundedPb in status(stop_nlp, list = true) # the problem is unbounded as fx <= - 1.0e50
  stop_nlp.meta.unbounded_pb = false #reinitialize
  #Let us now consider a maximization problem:
  nlp_max = ADNLPModel(NLPModelMeta(5, minimize = false), Counters(), f, x->[])
  stop_nlp.pb = nlp_max
  @test !stop_nlp.pb.meta.minimize
  stop!(stop_nlp)
  @test !(:UnboundedPb in status(stop_nlp, list = true)) # the problem is NOT unbounded as fx <= 1.0e50

  #Warning: see https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/autodiff_model.jl
  #for the proper way of defining an ADNLPModel
  nlp_bnd = ADNLPModel(NLPModelMeta(5, x0=zeros(5), lvar=zeros(5), uvar=zeros(5)),
                       Counters(), f, x->[])
  stop_bnd = NLPStopping(nlp_bnd)
  fill_in!(stop_bnd, zeros(5))
  @test KKT(stop_bnd.pb, stop_bnd.current_state) == 0.0
  reinit!(stop_bnd.current_state)
  @test optim_check_bounded(stop_bnd.pb, stop_bnd.current_state) == 0.0

  stop_bnd.meta.optimality_check = (x,y) -> NaN
  start!(stop_bnd)
  @test stop_bnd.meta.domainerror == true
  reinit!(stop_bnd, rcounters = true)
  @test neval_grad(stop_bnd.pb) == 0
  @test stop_bnd.meta.domainerror == false
  stop!(stop_bnd)
  @test stop_bnd.meta.domainerror == true

  stop_bnd.meta.optimality_check = (x,y) -> 0.0
  reinit!(stop_bnd, rcounters = true)
  @test neval_grad(stop_bnd.pb) == 0
  fill_in!(stop_bnd, zeros(5), mu = ones(5), lambda = zeros(0))
  @test stop_bnd.current_state.mu == ones(5)
  @test stop_bnd.current_state.lambda == zeros(0)

  update!(stop_bnd.current_state, fx = NaN, current_score = 0.0)
  stop!(stop_bnd)
  @test !isnan(stop_bnd.current_state.fx)

end
