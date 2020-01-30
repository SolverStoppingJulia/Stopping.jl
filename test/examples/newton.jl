###############################################################################
#
# We illustrate here the use of Stopping in a classical algorithm
# the Newton method for unconstrained optimization.
#
###############################################################################

using NLPModels, Stopping, Test

# We create a simple function to test
A = rand(5, 5);
Q = A' * A
f(x) = 0.5 * x' * Q * x
nlp = ADNLPModel(f,  ones(5))

#We now initialize the NLPStopping
nlp_at_x = NLPAtX(ones(5))
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), nlp_at_x)
#Note that in this case an alternative is:
#stop_nlp = NLPStopping(nlp)


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
@test stop_nlp.meta.tired == false
@test stop_nlp.meta.unbounded == false
@test stop_nlp.meta.optimal == true
