#mutable struct Counters
#  neval_obj    :: Int  # Number of objective evaluations.
#  neval_grad   :: Int  # Number of objective gradient evaluations.
#  neval_cons   :: Int  # Number of constraint vector evaluations.
#  neval_jcon   :: Int  # Number of individual constraint evaluations.
#  neval_jgrad  :: Int  # Number of individual constraint gradient evaluations.
#  neval_jac    :: Int  # Number of constraint Jacobian evaluations.
#  neval_jprod  :: Int  # Number of Jacobian-vector products.
#  neval_jtprod :: Int  # Number of transposed Jacobian-vector products.
#  neval_hess   :: Int  # Number of Lagrangian/objective Hessian evaluations.
#  neval_hprod  :: Int  # Number of Lagrangian/objective Hessian-vector products.
#  neval_jhprod :: Int  # Number of individual constraint Hessian-vector products.
#
#  function Counters()
#    return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
#  end
#end

function cvec(c::NLPModels.Counters)
    vc = Array{Int}(11)
    vc[1]  = c.neval_obj
    vc[2]  = c.neval_grad
    vc[3]  = c.neval_cons
    vc[4]  = c.neval_jcon
    vc[5]  = c.neval_jgrad
    vc[6]  = c.neval_jac
    vc[7]  = c.neval_jprod
    vc[8]  = c.neval_jtprod
    vc[9]  = c.neval_hess
    vc[10] = c.neval_hprod
    vc[11] = c.neval_jhprod

    return vc
end

import Base.<=
function <=(c1::NLPModels.Counters,c2::NLPModels.Counters)
    return all(cvec(c1) .<= cvec(c2))
end

function tired_check(s::TStoppingB)
    max_each = !(s.nlp.counters <= s.max_counters)
    max_total = sum_counters(s.nlp.counters) > s.max_eval

    max_calls = max_each | max_total
    s.elapsed_time = time() - s.start_time
    max_time = s.elapsed_time > s.max_time
    max_iter = s.iter >= s.max_iter

    return (max_iter) | (max_calls) | (max_time)
end

