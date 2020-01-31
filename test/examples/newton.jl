###############################################################################
#
# We illustrate here the use of Stopping in a classical algorithm,
# the Newton method for unconstrained optimization.
#
###############################################################################

using NLPModels, Stopping, Test

# We create a quadratic test function, and create an NLPModels
A = rand(5, 5);
Q = A' * A
f(x) = 0.5 * x' * Q * x
nlp = ADNLPModel(f,  ones(5))

#We now initialize the NLPStopping:
nlp_at_x = NLPAtX(ones(5)) #First create a State
#We use unconstrained_check as an optimality function (src/Stopping/nlp_admissible_functions.jl)
stop_nlp = NLPStopping(nlp, (x,y) -> unconstrained_check(x,y), nlp_at_x)
#Note that in this case an alternative is:
#stop_nlp = NLPStopping(nlp)

function newton(stp :: NLPStopping)

    #Notations
    pb = stp.pb; state = stp.current_state;
    #Initialization
    xt = state.x

    #First, call start! to check optimality and set an initial configuration
    #(start the time counter, set relative error ...)
    OK = update_and_start!(stp, x = xt, gx = grad(pb, xt), Hx = hess(pb, xt))

    while !OK
        #Compute the Newton direction
        d = -inv(state.Hx) * state.gx
        #Update the iterate
        xt = xt + d
        #Update the State and call the Stopping with stop!
        OK = update_and_stop!(stp, x = xt, gx = grad(pb, xt), Hx = hess(pb, xt))
    end

    return stp
end #end of function newton

stop_nlp = newton(stop_nlp)
#We can then ask stop_nlp the final status
@test :Optimal in status(stop_nlp, list = true)
#Explore the final values in stop_nlp.current_state
printstyled("Final solution is $(stop_nlp.current_state.x)", color = :green)
