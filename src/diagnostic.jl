# A stopping manager for iterative solvers
#
# the diagnostic structure

export TDiagnostic


type TDiagnostic
    # global diagnostic
    optimal :: Bool      # dual feasibility
    unbounded :: Bool    # 
    tired :: Bool        # max ressourses exhausted
    feasible :: Bool     # primal unfeasibility
    # fine grain diagnostic (if fine grain limits were specified)
    max_obj_f:: Bool
    max_obj_g :: Bool
    max_obj_H :: Bool
    max_obj_Hv :: Bool
    max_total :: Bool
    max_iter :: Bool
    max_time :: Bool

end

