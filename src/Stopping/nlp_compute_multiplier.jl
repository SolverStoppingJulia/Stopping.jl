"""
_compute_mutliplier: Additional function to estimate Lagrange multiplier of the problems
    (guarantee if LICQ holds)

`_compute_mutliplier(pb :: AbstractNLPModel, x :: T, gx :: T, cx :: T, Jx :: MT; active_prec_c :: Real = 1e-6, active_prec_b :: Real = 1e-6)`
"""
function _compute_mutliplier(
  pb::AbstractNLPModel,
  x::T,
  gx::T,
  cx::T,
  Jx::MT;
  active_prec_c::Real = 1e-6,
  active_prec_b::Real = 1e-6,
) where {MT, T}
  n = length(x)
  nc = length(cx)

  #active res_bounds
  Ib = findall(x -> (norm(x) <= active_prec_b), min(abs.(x - pb.meta.lvar), abs.(x - pb.meta.uvar)))
  if nc != 0
    #active constraints
    Ic = findall(
      x -> (norm(x) <= active_prec_c),
      min(abs.(cx - pb.meta.ucon), abs.(cx - pb.meta.lcon)),
    )

    Jc = hcat(Matrix(1.0I, n, n)[:, Ib], Jx'[:, Ic])
  else
    Ic = []
    Jc = hcat(Matrix(1.0I, n, n)[:, Ib])
  end

  mu, lambda = zeros(eltype(T), n), zeros(eltype(T), nc)
  if (Ib != []) || (Ic != [])
    l = Jc \ (-gx)
    mu[Ib], lambda[Ic] = l[1:length(Ib)], l[(length(Ib) + 1):length(l)]
  end

  return mu, lambda
end
