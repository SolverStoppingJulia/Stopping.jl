## State

As discussed before each Stopping contains a `current_state :: AbstractState` attribute containing the current information/state of the problem. When running the iterative loop, the `State` is updated and the `Stopping` make a decision based on this information.

The `current_state` contains all the information relative to a problem. We implemented three instances as an illustration:
- `GenericState` ;
-  `NLPAtX` representing the state of an `NLPModel`;
- `OneDAtX` for 1D optimization problems.

`GenericState` is an illustration of the behavior of such object that minimally contains:
- `x` the current iterate;
- `d` the current direction;
- `res` the current residual;
- `current_time` the current time;
- `current_score` the current optimality score.

By convention, `x` and `current_score` are mandatory information, and the other attribute are initialized with keywords arguments:
```julia
GenericState(zeros(n), 0.0, d = zeros(n), current_time = NaN)
```
the alternative would be
```julia
GenericState(zeros(n), d = zeros(n), current_time = NaN)
```

Beyond the use inside Stopping, returning the State also provides the user the opportunity to use some of the information computed by the algorithm.

### FAQ: Are there Type constraints when initializing a State?

Yes, an AbstractState{S,T} is actually a paramtric type where `S` is the type of the `current_score` and `T` is the type of `x`.
```julia
x0, score0 = rand(n), Array{Float64,1}(undef, n)
GenericState(x0, score0) #is an AbstractState{Array{Float64,1}, Array{Float64,1}}
```
By default, the `current_score` is a real number, hence
```julia
x0 = rand(n)
GenericState(x0) #is an AbstractState{Float64, Array{Float64,1}}
```
These types can be obtained with the functions `xtype` and `scoretype`:
```julia
scoretype(stp.current_state)
xtype(stp.current_state)
```

### FAQ: Can I design a tailored State for my problem?

`NLPAtX` is an illustration of a more evolved instance associated to `NLPModels` for nonlinear optimization models. It contains:
```julia
mutable struct 	NLPAtX{S, T <: AbstractVector, MT <: AbstractMatrix}  <: AbstractState{S, T}
#Unconstrained State
    x            :: T     # current point
    fx           :: eltype(T) # objective function
    gx           :: T  # gradient size: x
    Hx           :: MT  # hessian size: |x| x |x|
#Bounds State
    mu           :: T # Lagrange multipliers with bounds size of |x|
#Constrained State
    cx           :: T # vector of constraints lc <= c(x) <= uc
    Jx           :: MT  # jacobian matrix, size: |lambda| x |x|
    lambda       :: T    # Lagrange multipliers

    d            :: T #search direction
    res          :: T #residual
 #Resources State
    current_time   :: Float64
    current_score  :: S
    evals          :: Counters

 function NLPAtX(x             :: T,
                 lambda        :: T,
                 current_score :: S;
                 fx            :: eltype(T) = _init_field(eltype(T)),
                 gx            :: T = _init_field(T),
                 Hx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                 mu            :: T = _init_field(T),
                 cx            :: T = _init_field(T),
                 Jx            :: AbstractMatrix = _init_field(Matrix{eltype(T)}),
                 d             :: T = _init_field(T),
                 res           :: T = _init_field(T),
                 current_time  :: Float64 = NaN,
                 evals         :: Counters = Counters()
                 ) where {S, T <: AbstractVector}

  _size_check(x, lambda, fx, gx, Hx, mu, cx, Jx)

  return new{S, T, Matrix{eltype(T)}}(x, fx, gx, Hx, mu, cx, Jx, lambda, d, 
                                      res, current_time, current_score, evals)
 end
end
```
`_init_field(T)` initializes a value for a given type guaranteing type stability and minimal storage.
