using NLPModels
using Main.State
using Main.Stopping

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

function newton(stp :: NLPStopping)
    state = stp.current_state; xt = state.x;
    update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))
    OK = start!(stp)

    while !OK
        d = -inv(state.Hx) * state.gx

        xt = xt + d

        update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))

        OK = stop!(stp)
    end

    return stp
end

using LinearAlgebra
x0 = 1.5*ones(6)
c(x) = [sum(x)]
nlp2 = ADNLPModel(rosenbrock,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.])

nlp_at_x_c = NLPAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_c = NLPStopping(nlp2, (x,y) -> Stopping.KKT(x,y), nlp_at_x_c)

function penalty(stp :: NLPStopping)

 #algorithm parameters
 rho, rho_min = 100.0, 1e-10

 start!(stp)

 #prepare the subproblem stopping:
 sub_nlp_at_x = NLPAtX(stp.current_state.x)
 sub_stp = NLPStopping(stp.pb, (x,y) -> Stopping.unconstrained(x,y), sub_nlp_at_x)
 OK = false
 #main loop
 while !OK

  #solve the subproblem
  sub_stp.meta.atol = min(rho, sub_stp.meta.atol)
  newton(sub_stp)

  #update!(stp)
  fill_in!(stp, sub_stp.current_state.x)
  #Either stop! is true OR the penalty parameter is too small
  OK = stop!(stp) || rho < rho_min

@show stp.meta.nb_of_stop, OK, rho

  #update the penalty parameter if necessary
  rho = rho / 2
 end

 return stp
end

penalty(stop_nlp_c)
status(stop_nlp_c)
