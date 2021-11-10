@testset "Test @instate" begin
  function algo(stp::AbstractStopping, n::Int)
    OK = start!(stp)
    x = stp.current_state.x

    while !OK
      x = x .+ 1

      OK = update_and_stop!(stp, x = x)
    end

    return stp
  end

  function instatealgo(stp::AbstractStopping, n::Int)
    state = stp.current_state
    x = state.x
    OK = start!(stp)

    @instate state while !OK
      x = x .+ 1

      OK = stop!(stp)
    end

    return stp
  end

  n = 10
  stp1 = GenericStopping(x -> 0.0, zeros(n), max_iter = n, rtol = 0.0)
  stp2 = GenericStopping(x -> 0.0, zeros(n), max_iter = n, rtol = 0.0)

  stp1 = algo(stp1, n)
  stp2 = algo(stp2, n)

  @test stp1.current_state.x == stp2.current_state.x

  #=
    Suggestions:
     - It would be better to say stp.current_state instead of state. 
  =#
end
