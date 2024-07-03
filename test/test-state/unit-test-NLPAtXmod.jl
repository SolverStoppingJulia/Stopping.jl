
list_vectors = CUDA.functional() ? [Vector{Float64}, CuVector{Float64}] : [Vector{Float64}]
@testset "NLPAtX $T" for T in list_vectors
  #Test unconstrained NLPAtX
  x10 = fill!(T(undef, 10), 0)
  x10_1 = fill!(T(undef, 10), 1)
  x00 = fill!(T(undef, 0), 0)
  uncons_nlp_at_x = NLPAtX(x10)

  @test uncons_nlp_at_x.x == zeros(10)
  @test isnan(uncons_nlp_at_x.fx)
  @test uncons_nlp_at_x.gx == zeros(0)
  @test uncons_nlp_at_x.Hx == zeros(0, 0)
  @test uncons_nlp_at_x.mu == zeros(0)
  @test uncons_nlp_at_x.cx == zeros(0)
  @test uncons_nlp_at_x.Jx == zeros(0, 0)

  @test uncons_nlp_at_x.lambda == zeros(0)
  @test isnan(uncons_nlp_at_x.current_time)
  @test isnan(uncons_nlp_at_x.current_score)

  #check constrained NLPAtX
  cons_nlp_at_x = NLPAtX(x10, x10)

  @test cons_nlp_at_x.x == zeros(10)
  @test isnan(cons_nlp_at_x.fx)
  @test cons_nlp_at_x.gx == zeros(0)
  @test cons_nlp_at_x.Hx == zeros(0, 0)
  @test cons_nlp_at_x.mu == zeros(0)
  @test cons_nlp_at_x.cx == zeros(0)
  @test cons_nlp_at_x.Jx == zeros(0, 0)
  @test (false in (cons_nlp_at_x.lambda .== 0.0)) == false
  @test isnan(cons_nlp_at_x.current_time)
  @test isnan(cons_nlp_at_x.current_score)

  update!(cons_nlp_at_x, Hx = ones(20, 20), gx = fill!(T(undef, 2), 1), lambda = fill!(T(undef, 2), 0))
  compress_state!(cons_nlp_at_x, max_vector_size = 5, lambda = x0, gx = true)
  @test cons_nlp_at_x.Hx == zeros(0, 0)
  @test cons_nlp_at_x.x == [0.0]
  @test cons_nlp_at_x.lambda == zeros(0)
  @test cons_nlp_at_x.gx == zeros(0)

  # On vérifie que la fonction update! fonctionne
  update!(uncons_nlp_at_x, x = x10_1, fx = 1.0, gx = x10_1)
  update!(uncons_nlp_at_x, lambda = x10_1, current_time = 1.0)
  update!(uncons_nlp_at_x, Hx = ones(10, 10), mu = x10_1, cx = x10_1, Jx = ones(10, 10))

  @test (false in (uncons_nlp_at_x.x .== 1.0)) == false #assez bizarre comme test...
  @test uncons_nlp_at_x.fx == 1.0
  @test (false in (uncons_nlp_at_x.gx .== 1.0)) == false
  @test (false in (uncons_nlp_at_x.Hx .== 1.0)) == false
  @test uncons_nlp_at_x.mu == ones(10)
  @test uncons_nlp_at_x.cx == ones(10)
  @test (false in (uncons_nlp_at_x.Jx .== 1.0)) == false
  @test (false in (uncons_nlp_at_x.lambda .== 1.0)) == false
  @test uncons_nlp_at_x.current_time == 1.0
  @test isnan(uncons_nlp_at_x.current_score)

  reinit!(uncons_nlp_at_x)
  @test uncons_nlp_at_x.x == ones(10)
  @test isnan(uncons_nlp_at_x.fx)
  reinit!(uncons_nlp_at_x, x = x10)
  @test uncons_nlp_at_x.x == zeros(10)
  @test isnan(uncons_nlp_at_x.fx)
  reinit!(uncons_nlp_at_x, x10)
  @test uncons_nlp_at_x.x == zeros(10)
  @test isnan(uncons_nlp_at_x.fx)
  reinit!(uncons_nlp_at_x, x10, l = x00)
  @test uncons_nlp_at_x.x == zeros(10)
  @test isnan(uncons_nlp_at_x.fx)

  c_uncons_nlp_at_x = copy_compress_state(uncons_nlp_at_x, max_vector_size = 5)

  @test c_uncons_nlp_at_x != uncons_nlp_at_x
  @test c_uncons_nlp_at_x.x == [0.0]
  @test c_uncons_nlp_at_x.lambda == [1.0]

  uncons_nlp_at_x.Hx = zeros(10, 10)
  zip_uncons_nlp_at_x =
    compress_state!(uncons_nlp_at_x, keep = true, save_matrix = true, max_vector_size = 5, Hx = 1)
  zip_uncons_nlp_at_x.Hx == 0.0

  nlp_64 = NLPAtX(x10)
  nlp_64.x = x10
  nlp_64.fx = 1.0
  nlp_64.gx = x10

  # nlp_32 = convert_nlp(Float32, nlp_64)
  # @test typeof(nlp_32.x[1]) == Float32
  # @test typeof(nlp_32.fx[1]) == Float32
  # @test typeof(nlp_32.gx[1]) == Float32
  # @test isnan(nlp_32.mu[1])
  # @test isnan(nlp_32.current_time)
  #
  # @test typeof(nlp_64.x[1]) == Float64
  # @test typeof(nlp_64.fx[1]) == Float64
  # @test typeof(nlp_64.gx[1]) == Float64
  # @test isnan(nlp_64.mu[1])
  # @test isnan(nlp_64.current_time)

  #Test the _size_check:
  try
    NLPAtX(fill!(T(undef, 5), 1), gx = fill!(T(undef, 4), 0))
    @test false
  catch
    @test true
  end
  try
    NLPAtX(fill!(T(undef, 5), 1), mu = fill!(T(undef, 4), 0))
    @test false
  catch
    @test true
  end
  try
    NLPAtX(fill!(T(undef, 5), 1), Hx = zeros(4, 4))
    @test false
  catch
    @test true
  end
  try
    NLPAtX(fill!(T(undef, 5), 1), fill!(T(undef, 1), 0), cx = fill!(T(undef, 2), 0))
    @test false
  catch
    @test true
  end

  # Test matrix types
  state = NLPAtX(fill!(T(undef, 5), 0), fill!(T(undef, 2), 0), Jx = spzeros(2, 5), Hx = spzeros(5, 5))
  @test typeof(spzeros(2, 5)) == typeof(state.Jx)
  @test typeof(spzeros(5, 5)) == typeof(state.Hx)
  state = NLPAtX(fill!(T(undef, 5), 0), fill!(T(undef, 2), 0), Jx = spzeros(2, 5))
  @test typeof(spzeros(2, 5)) == typeof(state.Jx)
  @test typeof(zeros(5, 5)) == typeof(state.Hx)
end
