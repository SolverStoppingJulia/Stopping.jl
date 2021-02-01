@testset "How to stop NLP" begin
###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# We illustrate here the basic features of NLPStopping, which is a
# specialized version of the Stopping to the case where:
# pb is an AbstractNLPModel
# state shares the structure of the NLPAtX.
#
# NLPModels is a package to handle non-linear (constrained) optimization.
# NLPStopping is following this approach.
#
###############################################################################
#using Test, NLPModels, Stopping

#We first create a toy problem
f(x) = sum(x.^2)
x0 = zeros(5)
nlp = ADNLPModel(f, x0)
#Warning: see https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/autodiff_model.jl
#for the proper way of defining an ADNLPModel
meta = NLPModelMeta(5, x0=x0, lvar=zeros(5), uvar = Inf * ones(5))
nlp2 = ADNLPModel(meta, Counters(), f,  x->[])
nlp_at_x = NLPAtX(x0)
x1 = ones(5)

###############################################################################
#1) Initialize the NLPStopping.
#The specificity here is that the NLPStopping requires another mandatory input:
#optimality_check which is the function later used to compute the score.
#Recall that the score is then tested at 0, to declare optimality.
#Stopping provides a default KKT function.
stop_nlp = NLPStopping(nlp, nlp_at_x, optimality_check = (x,y) -> KKT(x,y))

#Another approach is to use the lazy way:
stop_nlp_lazy = NLPStopping(nlp2) #use nlp.meta.x0 as initial point
@test stop_nlp_lazy.current_state.x == nlp2.meta.x0

###############################################################################
#2) Fill in
#Before calling start! and stop! one should fill in current information in the
#State. -> the optimality_check then exploits this knowledge.
#As seen before, we by hand use update_and_start and update_and_stop.
#Another way is to call the fill_in! function:
fill_in!(stop_nlp, x1, matrix_info = false)
@test stop_nlp.current_state.x  == x1
@test stop_nlp.current_state.fx == 5.
@test stop_nlp.current_state.Hx == zeros(0,0)
#Note that since there are no constraints, c(x) and J(x) are not called:
@test stop_nlp.current_state.Jx == zeros(0,0)
@test stop_nlp.current_state.cx == zeros(0)
#Since there are no bounds on x, the Lagrange multiplier is not updated:
@test stop_nlp.current_state.mu == zeros(0)

#would give Hx if matrix_info = true
fill_in!(stop_nlp_lazy, x1)
@test stop_nlp_lazy.current_state.Hx != zeros(0,0)
#stop_nlp_lazy.pb has bounds, so mu is a vector of size x
@test size(x0) == size(stop_nlp_lazy.current_state.mu)

###############################################################################
#3) Evaluations
#Another particularity is that the NLPModels has a counter keeping track of
#the evaluations of each function.
#Similarly the NLPStopping has a dictionary keeping all the maximum number of
#evaluations:
@test typeof(stop_nlp.meta.max_cntrs) <: Dict
#For instance the limit in evaluations of objective and gradient:
@test stop_nlp.meta.max_cntrs[:neval_obj] == 20000
@test stop_nlp.meta.max_cntrs[:neval_grad] == 20000

#Limit can be set using _init_max_counters function:
stop_nlp.meta.max_cntrs = Stopping._init_max_counters(obj = 3, grad = 0, hess = 0)
@test stop_nlp.meta.max_cntrs[:neval_obj] == 3
@test stop_nlp.meta.max_cntrs[:neval_grad] == 0

OK = update_and_stop!(stop_nlp, evals = stop_nlp.pb.counters)
@test OK == true
@test stop_nlp.meta.resources == true
@test status(stop_nlp) == :EvaluationLimit

###############################################################################
#4) Unbounded problem
#An additional feature of the NLPStopping is to provide an _unbounded_problem_check
#whenever \|c(x)\| or -f(x) become too large.
stop_nlp.meta.unbounded_threshold = -6.0 #by default 1.0e50
stop!(stop_nlp)
@test stop_nlp.meta.unbounded_pb == true
@test stop_nlp.current_state.fx > stop_nlp.meta.unbounded_threshold
@test stop_nlp.meta.resources == true #still true as the state has not changed

###############################################################################
#An advanced feature is the possibility to send keywords to optimality_check:
optimality_fct_test = (x,y;a = 1.0) -> a

#In this case, the optimality_check function used to compute the score may
#depend on a parameter (algorithm-dependent for instance)
stop_nlp_2 = NLPStopping(nlp, nlp_at_x, optimality_check = optimality_fct_test)
fill_in!(stop_nlp_2, x0)

OK = stop!(stop_nlp_2, a = 0.0)
@test OK == true
@test stop_nlp_2.meta.optimal == true

#However, note that the same cannot be achieved with update_and_stop!:
reinit!(stop_nlp_2)
OK = update_and_stop!(stop_nlp_2, a = 0.0)
@test OK == false

end