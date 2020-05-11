# On vérifie que le constructeur par défaut fonctionne
ls_at_t = LSAtT(0.0)

@test ls_at_t.x   == 0.0
@test ls_at_t.ht  == nothing
@test ls_at_t.gt  == nothing
@test ls_at_t.h₀  == nothing
@test ls_at_t.g₀  == nothing

@test ls_at_t.current_time == nothing
@test ls_at_t.current_score == nothing

# On test la fonction update!(...)
update!(ls_at_t, x = 1.0, ht = 1.0, gt = 1.0, h₀ = 1.0)
update!(ls_at_t, g₀ = 1.0, current_time = 0.0, current_score = 0.0)

@test ls_at_t.x == 1.0
@test ls_at_t.ht == 1.0
@test ls_at_t.gt == 1.0
@test ls_at_t.h₀ == 1.0
@test ls_at_t.g₀ == 1.0
@test ls_at_t.current_time == 0.0
@test ls_at_t.current_score == 0.0

# on vérifie que la fonction copy fonctionne
ls_at_t_2 = copy(ls_at_t)

@test ls_at_t_2.x == 1.0
@test ls_at_t_2.ht == 1.0
@test ls_at_t_2.gt == 1.0
@test ls_at_t_2.h₀ == 1.0
@test ls_at_t_2.g₀ == 1.0
@test ls_at_t_2.current_time == 0.0
@test ls_at_t_2.current_score == 0.0

ls_64 = LSAtT(0.0)
update!(ls_64, x = 1.0, ht = 1.0, gt = 1.0, h₀ = 1.0)

# ls_32 = convert_ls(Float32, ls_64)
# @test typeof(ls_32.x) == Float32

reinit!(ls_64)
@test ls_64.x == 1.0
@test ls_64.ht == nothing
@test ls_64.current_time == nothing
