###############################################################################
#
# # ListofStates tutorial
#
# We illustrate here the use of ListofStates in dealing with a warm start
# procedure.
#
# ListofStates can also prove the user history over the iteration process.
#
# We compare the resolution of a convex unconstrained problem with 3 variants:
# - a steepest descent method
# - an inverse-BFGS method
# - a mix with 5 steps of steepest descent and then switching to BFGS with
#the history (using the strength of the ListofStates).
#
###############################################################################
using Stopping, NLPModels, LinearAlgebra, Test, Printf

import Stopping.armijo
function armijo(xk, dk, fk, slope, f)
  t = 1.0
  fk_new = f(xk + dk)
  while f(xk + t * dk) > fk + 1.0e-4 * t * slope
    t /= 1.5
    fk_new = f(xk + t * dk)
  end
  return t, fk_new
end

#Newton's method for optimization:
function steepest_descent(stp :: NLPStopping)

  xk = stp.current_state.x
  fk, gk = objgrad(stp.pb, xk)

  OK = update_and_start!(stp, fx = fk, gx = gk)

  @printf "%2s %9s %7s %7s %7s\n" "k" "fk" "||∇f(x)||" "t" "λ"
  @printf "%2d %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score)
  while !OK
    dk = - gk
    slope = dot(dk, gk)
    t, fk = armijo(xk, dk, fk, slope, x->obj(stp.pb, x))
    xk += t * dk
    fk, gk = objgrad(stp.pb, xk)
    
    OK = update_and_stop!(stp, x = xk, fx = fk, gx = gk)

    @printf "%2d %9.2e %7.1e %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score) t slope
  end
  return stp
end

function bfgs_quasi_newton_armijo(stp :: NLPStopping; Hk = nothing)

  xk = stp.current_state.x
  fk, gk = objgrad(stp.pb, xk)
  gm = gk

  dk, t = similar(gk), 1.
  if isnothing(Hk)
    Hk = I #start from identity matrix
  end

  OK = update_and_start!(stp, fx = fk, gx = gk)

  @printf "%2s %7s %7s %7s %7s\n" "k" "fk" "||∇f(x)||" "t" "cos"
  @printf "%2d %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score)

  while !OK
    if stp.meta.nb_of_stop != 0
      sk = t * dk
      yk = gk - gm
      ρk = 1/dot(yk, sk)
      #we need yk'*sk > 0 for instance yk'*sk ≥ 1.0e-2 * sk' * Hk * sk
      Hk = ρk ≤ 0.0 ? Hk : (I - ρk * sk * yk') * Hk * (I - ρk * yk * sk') + ρk * sk * sk'
      if norm(sk) ≤ 1e-14
        break
      end
      #H2 = Hk + sk * sk' * (dot(sk,yk) + yk'*Hk*yk )*ρk^2 - ρk*(Hk * yk * sk' + sk * yk'*Hk)
    end
    dk = - Hk * gk
    slope = dot(dk, gk) # ≤ -1.0e-4 * norm(dk) * gnorm
    t, fk = armijo(xk, dk, fk, slope, x->obj(stp.pb, x))

    xk = xk + t * dk
    gm = copy(gk)
    gk = grad(stp.pb, xk)

    OK = update_and_stop!(stp, x = xk, fx = fk, gx = gk)
    @printf "%2d %7.1e %7.1e %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score) t slope
  end
  return stp
end
using Test

############ PROBLEM TEST #############################################
fH(x) = (x[2]+x[1].^2-11).^2+(x[1]+x[2].^2-7).^2
nlp = ADNLPModel(fH, [10., 20.])
stp = NLPStopping(nlp, optimality_check = unconstrained_check, 
                 atol = 1e-6, rtol = 0.0, max_iter = 100)

reinit!(stp, rstate = true, x = nlp.meta.x0)
steepest_descent(stp)

@test status(stp) == :Optimal
@test stp.listofstates == VoidListofStates()

@show elapsed_time(stp)
@show nlp.counters

reinit!(stp, rstate = true, x = nlp.meta.x0, rcounters = true)
bfgs_quasi_newton_armijo(stp)

@test status(stp) == :Optimal
@test stp.listofstates == VoidListofStates()

@show elapsed_time(stp)
@show nlp.counters

NLPModels.reset!(nlp)
stp_warm = NLPStopping(nlp, optimality_check = unconstrained_check, 
                      atol = 1e-6, rtol = 0.0, max_iter = 5, 
                      n_listofstates = 5) #shortcut for list = ListofStates(5, Val{NLPAtX{Float64,Array{Float64,1},Array{Float64,2}}}()))
steepest_descent(stp_warm)
@test status(stp_warm) == :IterationLimit
@test length(stp_warm.listofstates) == 5

Hwarm = I
for i=2:5
  sk = stp_warm.listofstates.list[i][1].x - stp_warm.listofstates.list[i-1][1].x 
  yk = stp_warm.listofstates.list[i][1].gx - stp_warm.listofstates.list[i-1][1].gx 
  ρk = 1/dot(yk, sk)
  if ρk > 0.0
    global Hwarm = (I - ρk * sk * yk') * Hwarm * (I - ρk * yk * sk') + ρk * sk * sk'
  end
end

reinit!(stp_warm)
stp_warm.meta.max_iter = 100
bfgs_quasi_newton_armijo(stp_warm, Hk = Hwarm)
status(stp_warm)

@show elapsed_time(stp_warm)
@show nlp.counters

#=
 k        fk ||∇f(x)||       t       λ
 0 1.7e+05 3.2e+04
 1  2.73e+04 8.6e+03 1.0e-03 -1.1e+09
 2  1.80e+03 1.1e+03 2.3e-03 -7.3e+07
 3  1.24e+03 7.9e+02 1.2e-02 -1.3e+06
 4  6.37e+01 2.4e+01 1.2e-02 -6.3e+05
 5  1.34e+01 5.8e+01 2.0e-01 -8.3e+02
 6  5.87e+00 2.5e+01 1.3e-01 -3.5e+03
 7  2.88e+00 2.4e+01 2.6e-02 -6.7e+02
 8  2.42e+00 1.8e+01 1.7e-02 -6.1e+02
 9  6.58e-01 1.2e+01 1.2e-02 -6.1e+02
10  1.64e-01 5.3e+00 1.2e-02 -1.7e+02
11  4.96e-02 3.2e+00 1.2e-02 -4.4e+01
12  1.44e-02 1.6e+00 1.2e-02 -1.3e+01
13  4.35e-03 9.2e-01 1.2e-02 -3.9e+00
14  1.29e-03 5.0e-01 1.2e-02 -1.2e+00
15  3.87e-04 2.7e-01 1.2e-02 -3.5e-01
16  1.15e-04 1.5e-01 1.2e-02 -1.0e-01
17  3.45e-05 8.2e-02 1.2e-02 -3.1e-02
18  1.03e-05 4.5e-02 1.2e-02 -9.2e-03
19  3.08e-06 2.4e-02 1.2e-02 -2.8e-03
20  9.21e-07 1.3e-02 1.2e-02 -8.2e-04
21  2.75e-07 7.3e-03 1.2e-02 -2.5e-04
22  8.23e-08 4.0e-03 1.2e-02 -7.4e-05
23  2.46e-08 2.2e-03 1.2e-02 -2.2e-05
24  7.35e-09 1.2e-03 1.2e-02 -6.6e-06
25  2.20e-09 6.5e-04 1.2e-02 -2.0e-06
26  6.57e-10 3.6e-04 1.2e-02 -5.9e-07
27  1.96e-10 1.9e-04 1.2e-02 -1.8e-07
28  5.87e-11 1.1e-04 1.2e-02 -5.3e-08
29  1.75e-11 5.8e-05 1.2e-02 -1.6e-08
30  5.24e-12 3.2e-05 1.2e-02 -4.7e-09
31  1.57e-12 1.7e-05 1.2e-02 -1.4e-09
32  4.68e-13 9.5e-06 1.2e-02 -4.2e-10
33  1.40e-13 5.2e-06 1.2e-02 -1.3e-10
34  4.18e-14 2.8e-06 1.2e-02 -3.7e-11
35  1.25e-14 1.6e-06 1.2e-02 -1.1e-11
36  3.74e-15 8.5e-07 1.2e-02 -3.3e-12
elapsed_time(stp) = 0.7508440017700195
nlp.counters =   Counters:
             obj: ████████████████████ 889               grad: █⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 37                cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

 k      fk ||∇f(x)||       t     cos
 0 1.7e+05 3.2e+04
 1 2.7e+04 8.6e+03 1.0e-03 -1.1e+09
 2 1.8e+04 4.5e+03 1.2e-02 -1.8e+06
 3 2.5e+03 1.3e+03 1.0e+00 -7.1e+04
 4 1.2e+03 8.5e+02 1.0e+00 -1.7e+03
 5 3.2e+02 3.3e+02 1.0e+00 -1.4e+03
 6 9.8e+01 1.4e+02 1.0e+00 -3.2e+02
 7 2.7e+01 6.0e+01 1.0e+00 -1.1e+02
 8 6.4e+00 2.4e+01 1.0e+00 -3.0e+01
 9 9.9e-01 7.9e+00 1.0e+00 -8.2e+00
10 6.3e-02 1.9e+00 1.0e+00 -1.5e+00
11 8.7e-04 3.2e-01 1.0e+00 -1.1e-01
12 3.6e-05 7.9e-02 1.0e+00 -1.6e-03
13 1.4e-05 4.2e-02 1.0e+00 -2.9e-05
14 2.0e-07 3.4e-03 1.0e+00 -2.6e-05
15 4.1e-09 4.9e-04 1.0e+00 -3.6e-07
16 2.9e-12 2.5e-05 1.0e+00 -8.1e-09
17 2.5e-15 6.3e-07 1.0e+00 -5.6e-12
elapsed_time(stp) = 0.017869949340820312
nlp.counters =   Counters:
             obj: ████████████████████ 91                grad: ████⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 18                cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

 k        fk ||∇f(x)||       t       λ
 0 1.7e+05 3.2e+04
 1  2.73e+04 8.6e+03 1.0e-03 -1.1e+09
 2  1.80e+03 1.1e+03 2.3e-03 -7.3e+07
 3  1.24e+03 7.9e+02 1.2e-02 -1.3e+06
 4  6.37e+01 2.4e+01 1.2e-02 -6.3e+05
 5  1.34e+01 5.8e+01 2.0e-01 -8.3e+02
 6  5.87e+00 2.5e+01 1.3e-01 -3.5e+03
 k      fk ||∇f(x)||       t     cos
 0 5.9e+00 2.5e+01
 1 3.8e+00 2.7e+01 1.7e-02 -1.1e+03
 2 2.8e+00 2.4e+01 4.4e-01 -1.1e+01
 3 1.4e+00 1.2e+01 3.0e-01 -3.0e+01
 4 1.1e-02 1.3e+00 1.0e+00 -2.5e+00
 5 9.0e-05 9.2e-02 1.0e+00 -2.5e-02
 6 7.9e-08 3.9e-03 1.0e+00 -1.8e-04
 7 7.7e-10 3.8e-04 1.0e+00 -1.4e-07
 8 1.3e-19 4.2e-09 1.0e+00 -1.5e-09
elapsed_time(stp_warm) = 0.01520395278930664
nlp.counters =   Counters:
             obj: ████████████████████ 192               grad: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16                cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

  Counters:
             obj: ████████████████████ 192               grad: ██⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 16                cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     


=#
