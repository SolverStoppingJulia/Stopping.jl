s0 = GenericState(zeros(50))
s1 = GenericState(ones(10))
s2 = GenericState(NaN*ones(10), current_time = 1.0, current_score = 0.0)

stest = ListStates(s0, max_vector_size = 2, pnorm = Inf)

add_to_list!(stest, s1, max_vector_size = 2, pnorm = Inf)
add_to_list!(stest, s2, max_vector_size = 2, pnorm = Inf)

@test length(stest) == 3

stest2 = ListStates(s0, n = 2, max_vector_size = 2, pnorm = Inf)

add_to_list!(stest2, s1, max_vector_size = 2, pnorm = Inf)
add_to_list!(stest2, s2, max_vector_size = 2, pnorm = Inf)

@test length(stest2) == 2

df1 = print(stest, verbose = false)

df2 = print(stest2, verbose = false)

df3 = print(stest2, verbose = false, print_sym = [:x])

@test typeof(df2) <: DataFrame

stest3 = ListStates(-1, list = [[s0,nothing], [s1,nothing], [s2,nothing]], i = 3)

@test stest3[2,1] == s1

#nested lists

stest4 = ListStates(-1, list = [[s0, stest3]])

df4 = print(stest4[1,2], verbose = false)
