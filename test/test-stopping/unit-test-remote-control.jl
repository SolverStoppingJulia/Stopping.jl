@testset "Test Stopping Remote Control" begin
    src = StopRemoteControl()
    src2 = StopRemoteControl(user_check = false)
    cheap_src = cheap_stop_remote_control()
    cheap_src2 = cheap_stop_remote_control(user_check = false)
    
    for k in fieldnames(StopRemoteControl)
        if k == :cheap_check 
            @test !getfield(src, k)
        else
            @test getfield(src, k)
        end
    end
    for k in fieldnames(StopRemoteControl)
        if k ∈ (:cheap_check, :user_check)
            @test !getfield(src2, k)
        else
            @test getfield(src2, k)
        end
    end
    for k in fieldnames(StopRemoteControl)
        if k ∈ (:unbounded_and_domain_x_check, :domain_check, :unbounded_problem_check, :main_pb_check)
            @test !getfield(cheap_src, k)
        else
            @test getfield(cheap_src, k)
        end
    end
    for k in fieldnames(StopRemoteControl)
        if k ∈ (:unbounded_and_domain_x_check, :domain_check, :unbounded_problem_check, :main_pb_check, :user_check)
            @test !getfield(cheap_src2, k)
        else
            @test getfield(cheap_src2, k)
        end
    end

  show(src)

end
