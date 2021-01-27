@testset "Test VoidStopping" begin
  stp = VoidStopping()

  @test typeof(stp) <: AbstractStopping

  show(stp)
end