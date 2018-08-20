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

function put_at_inf!(c::NLPModels.Counters)
        c.neval_obj    = typemax(Int)
        c.neval_grad   = typemax(Int)
        c.neval_cons   = typemax(Int)
        c.neval_jcon   = typemax(Int)
        c.neval_jgrad  = typemax(Int)
        c.neval_jac    = typemax(Int)
        c.neval_jprod  = typemax(Int)
        c.neval_jtprod = typemax(Int)
        c.neval_hess   = typemax(Int)
        c.neval_hprod  = typemax(Int)
        c.neval_jhprod = typemax(Int)
end
