include("rosenbrock.jl")

x0 = ones(6)
c(x) = [sum(x)]
nlp2 = ADNLPModel(rosenbrock,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.])

nlp_at_x_c = NLPAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_c = NLPStopping(nlp2, KKT, nlp_at_x_c)

a = zeros(6)
fill_in!(stop_nlp_c, a)

@test cons(nlp2, a) == stop_nlp_c.current_state.cx
@test jac(nlp2, a) == stop_nlp_c.current_state.Jx

@test stop!(stop_nlp_c) == false
# we make sure the counter of stop works properly
@test stop_nlp_c.meta.nb_of_stop == 1

sol = ones(6)
fill_in!(stop_nlp_c, sol)

@test stop!(stop_nlp_c) == true
@test stop_nlp_c.current_state.current_score == KKT(stop_nlp_c.pb, stop_nlp_c.current_state)

stop_nlp_default = NLPStopping(nlp2, atol = 1.0)
fill_in!(stop_nlp_default, sol)
@test stop_nlp_default.meta.atol == 1.0
@test stop!(stop_nlp_default) == true

stop_nlp_default.optimality_check = unconstrained_check
update_and_stop!(stop_nlp_default, cx = nothing)
@test stop_nlp_default.current_state.cx != nothing

#Keywords in the stop! call
nlp_at_x_kargs = NLPAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_kargs = NLPStopping(nlp2, (x,y; test = 1.0, kwargs...) -> Stopping.KKT(x,y,kwargs...) + test, nlp_at_x_c)
fill_in!(stop_nlp_kargs, sol)
@test stop!(stop_nlp_kargs) == false
@test stop!(stop_nlp_kargs, test = 0.0) == true
