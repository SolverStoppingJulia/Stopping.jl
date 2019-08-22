# on vérifie simplement que le constructeur par défaut fait son travail
test_meta = StoppingMeta()

@test test_meta.atol == 1.0e-6
@test test_meta.rtol == 1.0e-15
@test test_meta.optimality0 == 1.0
@test test_meta.unbounded_threshold == -1.0e50
@test test_meta.unbounded_x == 1.0e50
@test test_meta.max_f == 9223372036854775807
@test test_meta.max_eval == 20_000
@test test_meta.max_iter == 5_000
@test test_meta.max_time == 300.0
@test test_meta.optimal_sub_pb == false
@test test_meta.unbounded == false
@test test_meta.tired == false
@test test_meta.stalled == false
@test test_meta.optimal == false
@test test_meta.nb_of_stop == 0
add_stop!(test_meta)
@test test_meta.nb_of_stop == 1
