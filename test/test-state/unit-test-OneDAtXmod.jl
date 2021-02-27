@testset "OneDAtX" begin
    
# On vérifie que le constructeur par défaut fonctionne
ls_at_t = OneDAtX(0.0)

@test scoretype(ls_at_t) == Float64
@test xtype(ls_at_t) == Float64

@test ls_at_t.x == 0.0
@test isnan(ls_at_t.fx)
@test isnan(ls_at_t.gx)
@test isnan(ls_at_t.f₀)
@test isnan(ls_at_t.g₀)

@test isnan(ls_at_t.current_time)
@test isnan(ls_at_t.current_score)

# On test la fonction update!(...)
update!(ls_at_t, x = 1.0, fx = 1.0, gx = 1.0, f₀ = 1.0)
update!(ls_at_t, g₀ = 1.0, current_time = 0.0, current_score = 0.0)

@test ls_at_t.x == 1.0
@test ls_at_t.fx == 1.0
@test ls_at_t.gx == 1.0
@test ls_at_t.f₀ == 1.0
@test ls_at_t.g₀ == 1.0
@test ls_at_t.current_time == 0.0
@test ls_at_t.current_score == 0.0

# on vérifie que la fonction copy fonctionne
ls_at_t_2 = copy(ls_at_t)

@test scoretype(ls_at_t_2) == Float64
@test xtype(ls_at_t_2) == Float64

@test ls_at_t_2.x == 1.0
@test ls_at_t_2.fx == 1.0
@test ls_at_t_2.gx == 1.0
@test ls_at_t_2.f₀ == 1.0
@test ls_at_t_2.g₀ == 1.0
@test ls_at_t_2.current_time == 0.0
@test ls_at_t_2.current_score == 0.0

ls_64 = OneDAtX(0.0)

@test scoretype(ls_64) == Float64
@test xtype(ls_64) == Float64

update!(ls_64, x = 1.0, fx = 1.0, gx = 1.0, f₀ = 1.0)

reinit!(ls_64)
@test ls_64.x == 1.0
@test isnan(ls_64.fx)
@test isnan(ls_64.current_time)

end
