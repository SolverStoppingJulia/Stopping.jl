abstract type AbstractStopRemoteControl end

"""
Turn a boolean to false to cancel this check in the functions stop! and start!.
"""
struct StopRemoteControl <: AbstractStopRemoteControl
    
    unbounded_and_domain_x_check :: Bool
    domain_check                 :: Bool
    optimality                   :: Bool
    infeasibility_check          :: Bool
    unbounded_problem_check      :: Bool
    tired_check                  :: Bool
    resources_check              :: Bool
    stalled_check                :: Bool
    iteration_check              :: Bool
    main_pb_check                :: Bool
    user_check                   :: Bool
    
    cheap_check                  :: Bool #`stop!` and `start!` stop whenever one check worked 
                                         #not used now
    
end

function StopRemoteControl(;unbounded_and_domain_x_check :: Bool = true, #O(n)
                            domain_check                 :: Bool = true, #O(n)
                            optimality                   :: Bool = true,
                            infeasibility_check          :: Bool = true,
                            unbounded_problem_check      :: Bool = true, #O(n)
                            tired_check                  :: Bool = true,
                            resources_check              :: Bool = true,
                            stalled_check                :: Bool = true,
                            iteration_check              :: Bool = true,
                            main_pb_check                :: Bool = true, #O(n)
                            user_check                   :: Bool = true,
                            cheap_check                  :: Bool = false)
                            
 return StopRemoteControl(unbounded_and_domain_x_check, domain_check, 
                          optimality, infeasibility_check, 
                          unbounded_problem_check, tired_check, 
                          resources_check, stalled_check,
                          iteration_check, main_pb_check, 
                          user_check, cheap_check)
end

"""
Return a StopRemoteControl with the most expansive checks at false (the O(n))
by default in Stopping when it has a main_stp.
"""
function cheap_stop_remote_control(;unbounded_and_domain_x_check :: Bool = false,
                                    domain_check                 :: Bool = false,
                                    optimality                   :: Bool = true,
                                    infeasibility_check          :: Bool = true,
                                    unbounded_problem_check      :: Bool = false,
                                    tired_check                  :: Bool = true,
                                    resources_check              :: Bool = true,
                                    stalled_check                :: Bool = true,
                                    iteration_check              :: Bool = true,
                                    main_pb_check                :: Bool = false,
                                    user_check                   :: Bool = true,
                                    cheap_check                  :: Bool = true)
                            
 return StopRemoteControl(unbounded_and_domain_x_check, domain_check, 
                          optimality, infeasibility_check, 
                          unbounded_problem_check, tired_check, 
                          resources_check, stalled_check,
                          iteration_check, main_pb_check, 
                          user_check, cheap_check)                        
end
