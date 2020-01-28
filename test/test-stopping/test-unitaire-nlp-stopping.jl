# We create a simple function to test
A = rand(5, 5);
Q = A' * A

f(x) = x' * Q * x
nlp = ADNLPModel(f, zeros(5))
nlp_at_x = NLPAtX(zeros(5))
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), nlp_at_x, optimality0 = 0.0)


a = zeros(5)
fill_in!(stop_nlp, a)

# we make sure that the fill_in! function works properly
@test obj(nlp, a) == stop_nlp.current_state.fx
@test grad(nlp, a) == stop_nlp.current_state.gx
@test stop_nlp.meta.optimality0 == 0.0

# we make sure the optimality check works properly
@test stop!(stop_nlp)
# we make sure the counter of stop works properly
@test stop_nlp.meta.nb_of_stop == 1

reinit!(stop_nlp, rstate = true, x = ones(5))
@test stop_nlp.current_state.x == ones(5)
@test stop_nlp.current_state.fx == nothing
@test stop_nlp.meta.nb_of_stop == 0

#We know test how to initialize the counter:
test_max_cntrs = Stopping._init_max_counters(obj = 2)
stop_nlp_cntrs = NLPStopping(nlp, max_cntrs = test_max_cntrs)
@test stop_nlp_cntrs.max_cntrs[:neval_obj] == 2
@test stop_nlp_cntrs.max_cntrs[:neval_grad] == 20000
@test stop_nlp_cntrs.max_cntrs[:neval_sum] == 20000*11
