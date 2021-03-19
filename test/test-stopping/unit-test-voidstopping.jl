@testset "Test VoidStopping" begin
  stp = VoidStopping()

  @test typeof(stp) <: AbstractStopping

  io = IOBuffer()
  show(io, stp)
end