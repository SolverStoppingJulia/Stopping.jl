# Fonction qu'on veut tester
# include("../NLPAtXmod.jl")
# importall NLPAtXmod

# On vérifie que le constructeur pour problème sans contrainte fonctionne
uncons_nlp_at_x = NLPAtX(zeros(10))

@test (false in (uncons_nlp_at_x.x .== 0.0))  == false #assez bizarre comme test...
@test (false in (isnan.(uncons_nlp_at_x.fx))) == false
@test (false in (isnan.(uncons_nlp_at_x.gx))) == false
@test (false in (isnan.(uncons_nlp_at_x.Hx))) == false
@test (false in (isnan.(uncons_nlp_at_x.mu))) == false
@test (false in (isnan.(uncons_nlp_at_x.cx))) == false
@test (false in (isnan.(uncons_nlp_at_x.Jx))) == false

@test (false in (isnan.(uncons_nlp_at_x.lambda))) == false
@test (false in (isnan.(uncons_nlp_at_x.start_time))) == false
@test (!(uncons_nlp_at_x.evals == nothing))

# On vérifie que le constucteur pour problème avec contrainte fonctionne
cons_nlp_at_x = NLPAtX(zeros(10), zeros(10))

@test (false in (uncons_nlp_at_x.x .== 0.0)) == false #assez bizarre comme test...
@test (false in (isnan.(uncons_nlp_at_x.fx))) == false
@test (false in (isnan.(uncons_nlp_at_x.gx))) == false
@test (false in (isnan.(uncons_nlp_at_x.Hx))) == false
@test (false in (isnan.(uncons_nlp_at_x.mu))) == false
@test (false in (isnan.(uncons_nlp_at_x.cx))) == false
@test (false in (isnan.(uncons_nlp_at_x.Jx))) == false
@test (false in (uncons_nlp_at_x.lambda .== 0.0)) == false
@test (false in (isnan.(uncons_nlp_at_x.start_time))) == false


# On vérifie que la fonction update! fonctionne
update!(uncons_nlp_at_x, x = ones(10), fx = 1.0, gx = ones(10))
update!(uncons_nlp_at_x, Hx = ones(10,10), mu = 1.0, cx = 1.0, Jx = ones(10,10))
update!(uncons_nlp_at_x, lambda = ones(10), tmps = 1.0)

@test (false in (uncons_nlp_at_x.x .== 1.0)) == false #assez bizarre comme test...
@test uncons_nlp_at_x.fx == 1.0
@test (false in (uncons_nlp_at_x.gx .== 1.0)) == false
@test (false in (uncons_nlp_at_x.Hx .== 1.0)) == false
@test uncons_nlp_at_x.mu == 1.0
@test uncons_nlp_at_x.cx == 1.0
@test (false in (uncons_nlp_at_x.Jx .== 1.0)) == false
@test (false in (uncons_nlp_at_x.lambda .== 1.0)) == false
@test uncons_nlp_at_x.start_time == 1.0

nlp_64 = NLPAtX(ones(10))
nlp_64.x = ones(10)
nlp_64.fx = 1.0
nlp_64.gx = ones(10)

# nlp_32 = convert_nlp(Float32, nlp_64)
# @test typeof(nlp_32.x[1]) == Float32
# @test typeof(nlp_32.fx[1]) == Float32
# @test typeof(nlp_32.gx[1]) == Float32
# @test isnan(nlp_32.mu[1])
# @test isnan(nlp_32.start_time)
#
# @test typeof(nlp_64.x[1]) == Float64
# @test typeof(nlp_64.fx[1]) == Float64
# @test typeof(nlp_64.gx[1]) == Float64
# @test isnan(nlp_64.mu[1])
# @test isnan(nlp_64.start_time)
