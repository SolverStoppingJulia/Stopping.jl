import LinearAlgebra.dot
function dot(u::AbstractLinearOperator, v::Union{AbstractVector, AbstractLinearOperator})
  if size(u, 1) == 1
    vu = Matrix(u')[:]
  elseif size(u, 2) == 1
    vu = Matrix(u)[:]
  else
    throw("Wrong dimension")
  end
  if size(v, 1) == 1
    vv = Matrix(v')[:]
  elseif size(v, 2) == 1
    vv = typeof(v) <: AbstractVector ? v : Matrix(v)[:]
  else
    throw("Wrong dimension")
  end
  if length(vu) != length(vv)
    throw("Wrong dimension")
  end
  return dot(vu, vv)
end

import Base.-
function -(u::AbstractVector, v::AbstractLinearOperator)
  vv = zeros(length(u))
  if size(v, 1) == 1
    vv = Matrix(v')[:]
  elseif size(v, 2) == 1
    vv = Matrix(v)[:]
  else
    throw("Wrong dimension")
  end
  if length(u) != length(vv)
    throw("Wrong dimension")
  end

  return u - vv
end
