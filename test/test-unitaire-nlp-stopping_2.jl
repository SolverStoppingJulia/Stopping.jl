function rosenbrock(x)

    n = 6;

    # Initializations
    f = 0

    evenIdx = 2:2:n
    oddIdx  = 1:2:(n-1)

    f1  = x[evenIdx] .- x[oddIdx].^2
    f2  = 1 .- x[oddIdx]

    # Function
    f   = sum( f1.^2 .+ f2.^2 )

    return f
end

using LinearAlgebra
x0 = ones(6)
c(x) = [sum(x)]
nlp2 = ADNLPModel(rosenbrock,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.])

nlp_at_x_c = NLPAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_c = NLPStopping(nlp2, (x,y) -> Stopping.KKT(x,y), nlp_at_x_c)

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
