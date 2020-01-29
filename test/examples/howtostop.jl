###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
# We illustrate here the basic features of Stopping.
#
# -> the case where a Stopping is a sub-Stopping is treated in the next tuto.
#
###############################################################################
using Test, Main.Stopping

x0 = ones(2)
pb = nothing

###############################################################################
#I. Initialize a Stopping
#The lazy way to initialize the stopping is to provide an initial point:
stop1 = GenericStopping(pb, x0, rtol = 1e-1)

#The more sophisticated way is to first build a State:
state1 = GenericState(ones(2))
#then, use it to create a Stopping:
stop2 = GenericStopping(pb, state1, rtol = 1e-1)

#Both ways give the same result:
@test stop1.current_state.x == stop2.current_state.x
@test stop1.current_state.current_time == stop2.current_state.current_time
#Keywords given in the Stopping creator are forwarded to the StoppingMeta.
@test stop1.meta.rtol == 1e-1

###############################################################################
#II. Check the status
#To ask the Stopping what is the current situation, we have the status function:
@test status(stop1) == :Unknown #nothing happened yet.
#The status function check the boolean values in the Meta:
#unbounded, unbounded_pb, tired, stalled, iteration_limit, resources, optimal,
#infeasible, main_pb, domainerror, suboptimal
stop1.meta.unbounded  = true
stop1.meta.suboptimal = true
#By default the status function prioritizes a status:
@test status(stop1) == :SubOptimal
#while you can access the list of status by turning the keyword list as true:
@test status(stop1, list =true) == [:SubOptimal, :Unbounded]

###############################################################################
#III. Analyze the situation: start!
#Two functions are designed to ask Stopping to analyze the current situation
#mainly described by the State: start!, stop!
#start! is designed to be used right at the beginning of the algorithm:
start!(stop1) #we will compare with stop2

#this call initializes a few entries:
#a) start_time in the META
@test isnan(stop2.meta.start_time)
@test !isnan(stop1.meta.start_time)
#b) optimality0 in the META (used to check the relative error)
@test stop2.meta.optimality0 == 1.0 #default value was 1.0
@test stop1.meta.optimality0 == Inf #GenericStopping has no specified measure
#c) the time measured is also updated in the State (if void)
@test stop1.current_state.current_time != nothing
#d) in the case where optimality0 is NaN, meta.domainerror becomes true
@test stop1.meta.domainerror == false
#e) the problem would be already solved if optimality0 pass a _null_test
#Since optimality0 is Inf, any value would pass the relative error check:
@test Stopping._null_test(stop1, Inf) == true
@test stop1.meta.optimal == true
@test :Optimal in status(stop1, list = true)
#The Stopping determines the optimality by testing a score at zero.
#The test at zero is controlled by the function meta.tol_check which
#takes 3 arguments: atol, rtol, optimality0. By default it check if the score
#is less than: max(atol, rtol * opt0)
stop1.meta.tol_check = (atol, rtol, opt0) -> atol
@test Stopping._null_test(stop1, Inf) == false
#This can be determined in the initialization of the Stopping
stop3 = GenericStopping(pb, state1, tol_check = (atol, rtol, opt0) -> atol)
@test Stopping._null_test(stop3, Inf) == false

#The function _optimality_check providing the score returns Inf by default
#and must be specialized for specialized Stopping.
#If State entries have to be specified before the start!, you can use the
#function update_and_start! instead of a update! and then a start!
update_and_start!(stop3, x = zeros(2), current_time = -1.0)
@test stop3.meta.optimal == false
@test stop3.current_state.current_time == -1.0
@test stop3.meta.start_time != nothing
@test stop3.current_state.x == zeros(2)

###############################################################################
#Once the iterations begins #stop! is the main function.
#if needed an update is needed first, we can use update_and_stop!
OK = stop!(stop3) #update the Stopping and return  a boolean
@test OK == false #no reason to stop just yet!
#The stop! call check the following:
#1) meta.domainerror: check if the score is NaN
#2) meta.optimal: the score passes the _null_test
#3) meta.unbounded: check if state.x is too large
#4) meta.unbounded_pb: false by default
#5) meta.tired: check if time is exhausted
#6) meta.resources: false by default
#7) meta.iteration_limit: check the number of iterations
#8) meta.stalled: false by default
#9) meta.main_pb: false by default -> see Stopping as a subproblem tutorial
# Note that 1 and 2 are also done by start!.
#1) check unboundedness of x:
@test update_and_stop!(stop3, x = (stop3.meta.unbounded_x + 1.0) * x0 )
@test stop3.meta.unbounded == true
#5) check time
stop3.meta.start_time = 0.0 #too  force the time limit.
stop!(stop3)
@test stop3.meta.tired == true
#7) Stopping the number of iterations by the number of calls to stop!
@test stop3.meta.nb_of_stop == 3 #We called stop3 3 times already
stop3.meta.max_iter = 3
stop!(stop3)
@test stop3.meta.iteration_limit == true #as stop3.meta.nb_of_stop > 3.
#Overall we activated three flags:
@test status(stop3, list = true) == [:Unbounded, :IterationLimit, :Tired]

###############################################################################
#Once we are done with an algorithm and want to reuse a stopping, we need to
#reinitialize all the entries.
reinit!(stop3)
#the status boolean are back to false
@test !stop3.meta.iteration_limit && !stop3.meta.tired && !stop3.meta.unbounded
#reinitialize also the entries updated by the start!
@test isnan(stop3.meta.start_time) && (stop3.meta.optimality0 == 1.0)
@test stop3.meta.nb_of_stop == 0 #and the counter of stop
#Note that by default reinit! does not reinitialize the current_state.
#This can be done by switching the keyword rstate to true.
#In this case, keywords are forwarded to the reinit! of current_state.
reinit!(stop3, rstate =  true, x = zeros(2))
@test stop3.current_state.current_time == nothing
@test stop3.current_state.x == zeros(2)
