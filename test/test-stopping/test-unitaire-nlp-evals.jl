@testset "Test NLP Evals" begin

    x0 = zeros(2)

    nlp = ADNLPModel(x -> norm(x.^2), x0)
    max_nlp = init_max_counters(obj = 2)
    nlp_stop_evals = NLPStopping(nlp, max_cntrs = max_nlp)

    nls = ADNLSModel(x -> x.^2, x0, 2)
    max_nls = init_max_counters_NLS(obj = 2, residual = 1)
    nls_stop_evals = NLPStopping(nls, max_cntrs = max_nls)

    @test typeof(nlp_stop_evals.pb.counters) == Counters
    @test nlp_stop_evals.meta.max_cntrs[:neval_obj] == 2
    @test typeof(nls_stop_evals.pb.counters) == NLSCounters
    @test nls_stop_evals.meta.max_cntrs[:neval_obj] == 2
    @test nls_stop_evals.meta.max_cntrs[:neval_residual] == 1

    Stopping._resources_check!(nlp_stop_evals, x0)
    @test nlp_stop_evals.meta.resources == false
    Stopping._resources_check!(nls_stop_evals, x0)
    @test nls_stop_evals.meta.resources == false

    nlp_stop_evals.meta.max_cntrs[:neval_obj] = 0
    Stopping._resources_check!(nlp_stop_evals, x0)
    @test nlp_stop_evals.meta.resources == false
    nls_stop_evals.meta.max_cntrs[:neval_residual] = -1
    Stopping._resources_check!(nls_stop_evals, x0)
    @test nls_stop_evals.meta.resources == true

    max_evals!(nlp_stop_evals, 10)
    @test nlp_stop_evals.meta.max_cntrs[:neval_obj] == 10
    @test nlp_stop_evals.meta.max_cntrs[:neval_sum] == 10 * length(fieldnames(Counters))

    max_evals!(nlp_stop_evals, allevals = 10, obj = 2)
    @test nlp_stop_evals.meta.max_cntrs[:neval_cons] == 10
    @test nlp_stop_evals.meta.max_cntrs[:neval_sum]  == 10 * length(fieldnames(Counters))
    @test nlp_stop_evals.meta.max_cntrs[:neval_obj] == 2

    #Test the case with a counters different from Counters and NLSCounters in NLPStopping
    mutable struct Test_cntrs
        neval     :: Int
    end
    mutable struct Test_pb <: AbstractNLPModel
        counters :: Test_cntrs
        meta     :: AbstractNLPModelMeta
    end
    NLPModels.sum_counters(pb :: Test_cntrs) = pb.neval
    NLPModels.sum_counters(pb :: Test_pb) = sum_counters(pb.counters)
    maxcount = Dict([(:neval, 1),(:neval_sum, -1)])

    pb = Test_pb(Test_cntrs(0.0), NLPModelMeta(5, x0 = zeros(5)))
    nls_stop_evals = NLPStopping(pb, max_cntrs = maxcount)
    Stopping._resources_check!(nls_stop_evals, x0)
    @test nls_stop_evals.meta.resources == true

end