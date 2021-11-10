@testset "Test Line-Search Stopping with LineModel" begin
  include("rosenbrock.jl")
  nlp = ADNLPModel(rosenbrock, 0.5ones(6))
  #We create a LineModel from nlp.meta.x0
  #The pb can be a LineModel defined in [SolverTools.jl](https://github.com/JuliaSmoothOptimizers/SolverTools.jl)
  d = -grad(nlp, nlp.meta.x0)
  pb = LineModel(nlp, nlp.meta.x0, d)
  state = OneDAtX(0.0, f₀ = obj(pb, 0.0), g₀ = grad(pb, 0.0))
  @test state.x == 0.0
  @test !isnan(state.f₀)
  @test !isnan(state.g₀)
  @test isnan(state.fx)
  @test isnan(state.gx)

  τ₀ = 1e-4 #Armijo parameter
  meta = StoppingMeta(
    optimality_check = (pb, lsatt) -> lsatt.fx - lsatt.f₀ - lsatt.g₀ * lsatt.x * τ₀,
    tol_check = (a, b, c) -> 0.0,
    tol_check_neg = (atol, rtol, opt0) -> -Inf,
  )
  src = StopRemoteControl(resources_check = false)

  stp = NLPStopping(pb, meta, src, state)

  #Independent of x
  @test !Stopping._infeasibility_check!(stp, 1.0)
  @test !Stopping._resources_check!(stp, 1.0)
  #Depend on x
  #Stopping._unbounded_problem_check!(stp, 1.) #doesn't work if there is no state.fx

  @test start!(stp) #NaN is optimal
  @test isnan(stp.current_state.current_score)
  @test stop!(stp)
  @test !stp.meta.infeasible

  #Simulate a backtracking
  reinit!(stp)
  reset!(pb)
  t = 1.0
  i = 0
  OK = update_and_start!(stp, x = t, fx = obj(stp.pb, t), gx = grad(stp.pb, t))
  while !OK
    t /= 2
    i += 1
    OK = update_and_stop!(stp, x = t, fx = obj(stp.pb, t), gx = grad(stp.pb, t))
  end
  @test i == stp.meta.nb_of_stop
  @test neval_obj(stp.pb) == i + 1
  @test neval_grad(stp.pb) == i + 1
  @test stp.meta.optimal
  @test :Optimal in status(stp, list = true)

  #With the nlp as a main_stp
  #The issue is that pb.meta.x0 is a vector of size 1!, so NLPStopping(pb) is not possible
  stp2 = NLPStopping(
    pb,
    OneDAtX(0.0, f₀ = obj(pb, 0.0), g₀ = grad(pb, 0.0)),
    main_stp = NLPStopping(nlp),
    optimality_check = armijo,
  )

  @test update_and_start!(stp2, fx = 0.0, gx = 0.0)
  @test update_and_stop!(stp2, fx = 0.0, gx = 0.0)

  #Check an OR condition
  lsatt = OneDAtX(0.0, f₀ = obj(pb, 0.0), g₀ = grad(pb, 0.0), (NaN, NaN))
  @test isnan.(lsatt.current_score) == (true, true)
  @test lsatt.x == 0.0

  #Check an AND condition
  lsatt = OneDAtX(0.0, f₀ = obj(pb, 0.0), g₀ = grad(pb, 0.0), [Inf, Inf])
  τ₀ = 1e-4 #Armijo parameter
  τ₁ = 0.99 #Wolfe parameter
  meta = StoppingMeta(
    optimality_check = (pb, lsatt) ->
      vcat(lsatt.fx - lsatt.f₀ - lsatt.g₀ * lsatt.x * τ₀, abs(lsatt.gx) - τ₁ * abs(lsatt.g₀)),
    tol_check = (a, b, c) -> 0.0,
    tol_check_neg = (atol, rtol, opt0) -> -Inf,
  )
  src = StopRemoteControl(resources_check = false)

  stp = NLPStopping(pb, meta, src, lsatt)

  #Simulate a backtracking
  reinit!(stp)
  reset!(pb)
  t = 1.0
  i = 0
  OK = update_and_start!(stp, x = t, fx = obj(stp.pb, t), gx = grad(stp.pb, t))
  while !OK
    t /= 2
    i += 1
    OK = update_and_stop!(stp, x = t, fx = obj(stp.pb, t), gx = grad(stp.pb, t))
  end
  @test i == stp.meta.nb_of_stop
  @test neval_obj(stp.pb) == i + 1
  @test neval_grad(stp.pb) == i + 1
  @test stp.meta.optimal
  @test :Optimal in status(stp, list = true)

  reinit!(stp, rcounters = true, rstate = true)
  fill_in!(stp, 1.0)
  @test !isnan(stp.current_state.fx)
  @test !isnan(stp.current_state.gx)
  @test !isnan(stp.current_state.f₀)
  @test !isnan(stp.current_state.g₀)
end
