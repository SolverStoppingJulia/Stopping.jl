"""
_compute_mutliplier: Additional function to estimate Lagrange multiplier of the problems
    (guarantee if LICQ holds)
"""
function _compute_mutliplier(pb    :: AbstractNLPModel,
                             x     :: Iterate,
                             gx    :: Iterate,
                             cx    :: Iterate,
                             Jx    :: Any;
                             active_prec_c :: Float64 = 1e-6,
                			 active_prec_b :: Float64 = 1e-6)

 n  = length(x)
 nc = cx == nothing ? 0 : length(cx)

 #active res_bounds
 Ib = findall(x->(norm(x) <= active_prec_b),
			      min(abs.(x - pb.meta.lvar),
				      abs.(x - pb.meta.uvar)))
 if nc != 0
  #active constraints
  Ic = findall(x->(norm(x) <= active_prec_c),
                   min(abs.(cx-pb.meta.ucon),
                   abs.(cx-pb.meta.lcon)))

  Jc = hcat(Matrix(1.0I, n, n)[:,Ib], Jx'[:,Ic])
 else
  Ic = []
  Jc = hcat(Matrix(1.0I, n, n)[:,Ib])
 end

 mu, lambda = zeros(n), zeros(nc)
 if (Ib != []) || (Ic  != [])
  l = Jc \ (- gx)
  mu[Ib], lambda[Ic] = l[1:length(Ib)], l[length(Ib)+1:length(l)]
 end

 return mu, lambda
end
