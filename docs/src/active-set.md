## Active set algorithm

In this test problem we consider an active-set method.

Note that there is no optimization of the evaluations here.

Note the use of a structure for the algorithmic parameters which is
forwarded to all the 3 steps. If a parameter is not mentioned, then the default
entry in the algorithm will be taken.

```
include("penalty.jl")
```

First, we create a subtype of AbstractNLPModel to represent the unconstrained
subproblem we "solve" at each iteration of the activeset.
```
import NLPModels: obj, grad, hess, hprod

mutable struct ActifNLP <: AbstractNLPModel
 nlp :: AbstractNLPModel
 x0  :: Vector #reference vector
 I   :: Vector #set of active indices
 Ic  :: Vector #set of inactive indices
 meta :: AbstractNLPModelMeta
 counters :: Counters
end

function obj(anlp :: ActifNLP, x :: Vector) t=anlp.x0;t[anlp.Ic]=x; return obj(anlp.nlp, t) end
function grad(anlp :: ActifNLP, x :: Vector) t=anlp.x0;t[anlp.Ic]=x; return grad(anlp.nlp, t)[anlp.Ic] end
function hess(anlp :: ActifNLP, x :: Vector) t=anlp.x0;t[anlp.Ic]=x; return hess(anlp.nlp, t)[anlp.Ic,anlp.Ic] end
function hprod(anlp :: ActifNLP, x :: Vector, v :: Vector, y :: Vector) return hess(anlp, x) * v end
```

### Active-set algorithm for bound constraints optimization
fill_in! used instead of update! (works but usually more costly in evaluations)
subproblems are solved via Newton method
```
function activeset(stp :: NLPStopping;
                   active :: Float64 = stp.meta.tol_check(stp.meta.atol,stp.meta.rtol,stp.meta.optimality0),
                   prms = nothing)

 xt = stp.current_state.x; n = length(xt); all = findall(xt .== xt)

 if maximum(vcat(max.(xt  - stp.pb.meta.uvar,0.0),max.(- xt  + stp.pb.meta.lvar,0.0))) > 0.0
  #OK = true; stp.meta.fail_sub_pb = true
  #xt is not feasible
  xt = max.(min.(stp.current_state.x,  stp.pb.meta.uvar),  stp.pb.meta.lvar)
 end

 fill_in!(stp, xt)
 OK = start!(stp)

 Il = findall(abs.(- xt  + stp.pb.meta.lvar).<= active)
 Iu = findall(abs.(  xt  - stp.pb.meta.uvar).<= active)
 I = union(Il, Iu); Ic = setdiff(all, I)
 nI = max(0, length(xt) - length(Il) - length(Iu)) #lvar_i != uvar_i
@show xt, I
while !OK

   #prepare the subproblem stopping:
   subpb = ActifNLP(nlp, xt, I, Ic, NLPModelMeta(nI), Counters())
   #the subproblem stops if he solved the unconstrained nlp or iterate is infeasible
   feas(x,y) = maximum(vcat(max.(y.x  - stp.pb.meta.uvar[Ic],0.0),max.(- y.x  + stp.pb.meta.lvar[Ic],0.0)))
   check_func(x,y) = feas(x,y) > 0.0 ? 0.0 : unconstrained_check(x,y)
   substp = NLPStopping(subpb, check_func, NLPAtX(xt[Ic]), main_stp = stp)

   #we solve the unconstrained subproblem:
   global_newton(substp, prms)
   @show status(substp, list = true)

   if feas(substp.pb, substp.current_state) > 0.0 #new iterate is infeasible
     #then we need to project
     xt[Ic] = max.(min.(substp.current_state.x,  stp.pb.meta.uvar[Ic]),  stp.pb.meta.lvar[Ic])
     #we keep track of the new active indices
     Inew = setdiff(union(findall(abs.(- xt  + stp.pb.meta.lvar).<= active), findall(abs.(  x0  - stp.pb.meta.uvar).<= active)), I)
   else
     Inew = []
   end

   fill_in!(stp, xt) #the lazy update

   OK = update_and_stop!(stp, evals = stp.pb.counters)

   if !OK #we use a relaxation rule based on an approx. of Lagrange multipliers
     Irmv = findall(stp.current_state.mu .<0.0)
     I = union(setdiff(I, Irmv), Inew)
     Ic = setdiff(all, I)
   end
 @show xt, I
 end #end of main loop

 return stp
end
```
