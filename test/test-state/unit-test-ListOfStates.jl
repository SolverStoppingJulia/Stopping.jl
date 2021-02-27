@testset "List of States" begin

    s0 = GenericState(zeros(50))
    s1 = GenericState(ones(10))
    s2 = GenericState(NaN*ones(10), current_time = 1.0, current_score = 0.0)
    
    @test typeof(ListofStates(s0)) <: AbstractListofStates
    @test typeof(ListofStates(-1)) <: AbstractListofStates
    @test typeof(ListofStates(1)) <: AbstractListofStates
    @test typeof(ListofStates(-1, 3, [])) <: AbstractListofStates
    @test typeof(ListofStates(-1, [])) <: AbstractListofStates

    stest = ListofStates(s0, max_vector_size = 2, pnorm = Inf)

    add_to_list!(stest, s1, max_vector_size = 2, pnorm = Inf)
    add_to_list!(stest, s2, max_vector_size = 2, pnorm = Inf)

    @test length(stest) == 3

    stest2 = ListofStates(s0, n = 2, max_vector_size = 2, pnorm = Inf)

    add_to_list!(stest2, s1, max_vector_size = 2, pnorm = Inf)
    add_to_list!(stest2, s2, max_vector_size = 2, pnorm = Inf)

    @test length(stest2) == 2

    df1 = print(stest, verbose = false)

    df2 = print(stest2, verbose = false)

    df3 = print(stest2, verbose = false, print_sym = [:x])

    @test typeof(df2) <: DataFrame

    stest3 = ListofStates(-1, 3, [(s0, VoidListofStates()), (s1, VoidListofStates()), (s2, VoidListofStates())])

    @test stest3[2,1] == s1
    
    stest4 = ListofStates(-1, [(s0, VoidListofStates()), (s1, VoidListofStates()), (s2, VoidListofStates())])
    
    @test length(stest4) == 3

    #nested lists

    stest5 = ListofStates(-1, [(s0, stest3)])

    df5 = print(stest5[1,2], verbose = false)

end
