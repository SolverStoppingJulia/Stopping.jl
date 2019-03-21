# We create a simple function to test
A = rand(5, 5);
Q = A' * A

f(x) = x' * Q * x
nlp = ADNLPModel(f, zeros(5))
nlp_at_x = NLPAtX(zeros(5))
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), nlp_at_x)


a = zeros(5)
fill_in!(stop_nlp, a)

# we make sure that the fill_in! function works properly
@test obj(nlp, a) == stop_nlp.current_state.fx
@test grad(nlp, a) == stop_nlp.current_state.gx

# we make sure the optimality check works properly
@test stop!(stop_nlp)
