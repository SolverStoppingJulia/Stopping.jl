function Armijo(nlp, f, g, x, d, τ₀)
    # Simple Armijo backtracking
    hp0 = g'*d
    t   = 1.0
    ft  = obj(nlp, x + t*d)
    while ft > ( f + τ₀*t*hp0)
        t /= 2.0
        ft = obj(nlp, x + t*d)
    end
end
