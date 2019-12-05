# On vérifie que le constructeur par défaut fonctionne
ls_at_t = LSAtT(0.0)

@test ls_at_t.x == 0.0
@test isnan(ls_at_t.ht)
@test isnan(ls_at_t.gt)
@test isnan(ls_at_t.h₀)
@test isnan(ls_at_t.g₀)
@test isnan(ls_at_t.start_time)

# On test la fonction update!(...)
update!(ls_at_t, x = 1.0, ht = 1.0, gt = 1.0, h₀ = 1.0)
update!(ls_at_t, g₀ = 1.0, tmps = 0.0)

@test ls_at_t.x == 1.0
@test ls_at_t.ht == 1.0
@test ls_at_t.gt == 1.0
@test ls_at_t.h₀ == 1.0
@test ls_at_t.g₀ == 1.0
@test ls_at_t.start_time == 0.0

# on vérifie que la fonction copy fonctionne
ls_at_t_2 = copy(ls_at_t)

@test ls_at_t_2.x == 1.0
@test ls_at_t_2.ht == 1.0
@test ls_at_t_2.gt == 1.0
@test ls_at_t_2.h₀ == 1.0
@test ls_at_t_2.g₀ == 1.0
@test ls_at_t_2.start_time == 0.0

ls_64 = LSAtT(0.0)
update!(ls_64, x = 1.0, ht = 1.0, gt = 1.0, h₀ = 1.0)

# ls_32 = convert_ls(Float32, ls_64)
# @test typeof(ls_32.x) == Float32
