# on vérifie simplement que le constructeur par défaut fait son travail
test_meta = StoppingMeta()

@test test_meta.optimality0           == 1.0
@test test_meta.optimality_check(1,1) == Inf
@test test_meta.unbounded_threshold   == 1.0e50
@test test_meta.unbounded_x           == 1.0e50
@test test_meta.max_f                 == 9223372036854775807
@test test_meta.max_cntrs             == Dict()
@test test_meta.max_eval              == 20_000
@test test_meta.max_iter              == 5_000
@test test_meta.max_time              == 300.0
@test isnan(test_meta.start_time)
@test test_meta.fail_sub_pb           == false
@test test_meta.unbounded             == false
@test test_meta.unbounded_pb          == false
@test test_meta.tired                 == false
@test test_meta.stalled               == false
@test test_meta.iteration_limit       == false
@test test_meta.optimal               == false
@test test_meta.suboptimal            == false
@test test_meta.main_pb               == false
@test test_meta.nb_of_stop            == 0

try
 StoppingMeta(tol_check = x -> x)
 @test false #tol_check must have 3 arguments and not one
catch
 @test true
end

try
 StoppingMeta(tol_check_neg = x -> x)
 @test false #tol_check must have 3 arguments and not one
catch
 @test true
end
