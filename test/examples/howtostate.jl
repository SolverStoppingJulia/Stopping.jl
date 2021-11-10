@testset "Test How to State" begin
  ###############################################################################
  #
  # The data used through the algorithmic process in the Stopping framework
  # are stored in a State.
  # We illustrate here the GenericState and its features
  #
  ###############################################################################
  #using Test, Stopping

  ###############################################################################
  #The GenericState contains only two entries:
  # a vector x, and a Float current_time
  state1 = GenericState(ones(2)) #takes a Vector as a mandatory input
  state2 = GenericState(ones(2), current_time = 1.0)

  #By default if a non-mandatory entry is not specified it is void:
  @test isnan(state1.current_time)
  @test state2.current_time == 1.0

  ###############################################################################
  #The GenericState has two functions: update! and reinit!
  #update! is used to update entries of the State:
  update!(state1, current_time = 1.0)
  @test state1.current_time == 1.0
  #Note that the update select the relevant entries
  update!(state1, fx = 1.0) #does nothing as there are no fx entry
  @test state1.current_time == 1.0 && state1.x == ones(2)

  #The update! can be done only if the new entry is void or has the same type
  #as the existing one.
  update!(state1, current_time = 2) #does nothing as it is the wrong type
  @test state1.current_time == 1.0
  #An advanced user can force the update even if the type is not the same by
  #turning the keyword convert as true (it is false by default).
  #update!(state1, convert = true, current_time = 2) NON!!!
  #@test state1.current_time == 2
  #Non-required entry in the State can always be set as void without convert
  update!(state1, current_time = NaN)
  @test isnan(state1.current_time)

  #A shorter way to empty the State is to use the reinit! function.
  #This function is particularly useful, when there are many entries.
  reinit!(state2)
  @test state2.x == ones(2) && isnan(state2.current_time)
  #If we want to reinit! with a different value of the mandatory entry:
  reinit!(state2, zeros(2))
  @test state2.x == zeros(2) && isnan(state2.current_time)
  #After reinitializing the State reinit! can update entries passed as keywords.
  #either in the default call:
  reinit!(state2, current_time = 1.0)
  @test state2.x == zeros(2) && state2.current_time == 1.0
  #or in the one changing x:
  reinit!(state2, ones(2), current_time = 1.0)
  @test state2.x == ones(2) && state2.current_time == 1.0

  ###############################################################################
  #The State has also a private function guaranteeing there are no NaN
  OK = Stopping._domain_check(state1) #function returns a boolean
  @test OK == false #no NaN

  @test Stopping._domain_check(state2) == false
  update!(state2, x = [NaN, 0.0])
  @test Stopping._domain_check(state2) == true
end
