@testset "StoppingMeta" begin
  #Checked the default constructor
  test_meta = StoppingMeta()

  io = IOBuffer()
  show(io, test_meta) #improve coverage

  @test Stopping.checktype(test_meta) == Float64
  @test Stopping.toltype(test_meta) == Float64
  @test Stopping.metausertype(test_meta) == Nothing
  @test Stopping.inttype(test_meta) == Int64

  @test test_meta.optimality0 == 1.0
  @test test_meta.optimality_check(1, 1) == Inf
  @test test_meta.unbounded_threshold == 1.0e50
  @test test_meta.unbounded_x == 1.0e50
  @test test_meta.max_f == 9223372036854775807
  @test test_meta.max_cntrs == Dict()
  @test test_meta.max_eval == 20_000
  @test test_meta.max_iter == 5_000
  @test test_meta.max_time == 300.0
  @test isnan(test_meta.start_time)
  @test test_meta.fail_sub_pb == false
  @test test_meta.unbounded == false
  @test test_meta.unbounded_pb == false
  @test test_meta.tired == false
  @test test_meta.stalled == false
  @test test_meta.iteration_limit == false
  @test test_meta.optimal == false
  @test test_meta.suboptimal == false
  @test test_meta.main_pb == false
  @test test_meta.stopbyuser == false
  @test test_meta.exception == false
  @test test_meta.infeasible == false
  @test test_meta.nb_of_stop == 0
  @test isnothing(test_meta.meta_user_struct)
  @test test_meta.recomp_tol

  @test_throws MethodError StoppingMeta(tol_check = x -> x)
  @test_throws MethodError StoppingMeta(tol_check_neg = x -> x)

  @test tol_check(test_meta) == (1.0e-6, -1.0e-6)

  test_meta.recomp_tol = false #if recomp_tol is false, tol_check don't reevaluate the functions
  test_meta.atol = 1e-2
  @test tol_check(test_meta) == (1.0e-6, -1.0e-6)

  update_tol!(test_meta, atol = 1e-1)
  @test test_meta.recomp_tol == true
  @test test_meta.atol == 1e-1
  @test tol_check(test_meta) == (0.1, -0.1)

  @test !OK_check(test_meta)
  test_meta.recomp_tol = false
  test_meta.suboptimal = true
  @test OK_check(test_meta)

  io = IOBuffer()
  show(io, test_meta)

  reinit!(test_meta)
  @test !test_meta.suboptimal
  @test !OK_check(test_meta)

  #@test_throws ErrorException("StoppingMeta: tol_check should be greater than tol_check_neg.") StoppingMeta(tol_check_neg = (a,b,c) -> Inf)
end

@testset "StoppingMeta - 2nd constructor" begin
  #Checked the default constructor
  test_meta = StoppingMeta(ones(2), -ones(2))

  @test Stopping.checktype(test_meta) == Array{Float64, 1}
  @test Stopping.toltype(test_meta) == Float64
  @test Stopping.metausertype(test_meta) == Nothing
  @test Stopping.inttype(test_meta) == Int64

  @test test_meta.optimality0 == 1.0
  @test test_meta.optimality_check(1, 1) == Inf
  @test test_meta.unbounded_threshold == 1.0e50
  @test test_meta.unbounded_x == 1.0e50
  @test test_meta.max_f == 9223372036854775807
  @test test_meta.max_cntrs == Dict()
  @test test_meta.max_eval == 20_000
  @test test_meta.max_iter == 5_000
  @test test_meta.max_time == 300.0
  @test isnan(test_meta.start_time)
  @test test_meta.fail_sub_pb == false
  @test test_meta.unbounded == false
  @test test_meta.unbounded_pb == false
  @test test_meta.tired == false
  @test test_meta.stalled == false
  @test test_meta.iteration_limit == false
  @test test_meta.optimal == false
  @test test_meta.suboptimal == false
  @test test_meta.main_pb == false
  @test test_meta.stopbyuser == false
  @test test_meta.exception == false
  @test test_meta.infeasible == false
  @test test_meta.nb_of_stop == 0
  @test isnothing(test_meta.meta_user_struct)
  @test !test_meta.recomp_tol
end
