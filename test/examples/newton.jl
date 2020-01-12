using NLPModels
using Stopping

# We create a simple function to test
A = rand(5, 5);
Q = A' * A

f(x) = 0.5 * x' * Q * x
nlp = ADNLPModel(f,  ones(5))
nlp_at_x = NLPAtX(ones(5))
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), nlp_at_x)


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

stop_nlp = newton(stop_nlp)

# We can look at the meta to know what happened
stop_nlp.meta.tired
stop_nlp.meta.unbounded
stop_nlp.meta.optimal
