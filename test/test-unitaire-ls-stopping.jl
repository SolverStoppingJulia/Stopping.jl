using LineSearch
using LinearAlgebra

# On crée notre problème
nlp = CUTEstModel("ARWHEAD")
h = LineModel(nlp, nlp.meta.x0, -grad(nlp,nlp.meta.x0))
lsatx = LSAtT(0.0)

# On crée notre objet stopping qu'on va passer au Line Search
stop = LS_Stopping(h, (x,y)-> armijo(x,y), lsatx);

# On vérifie que les fonctions ont le bon comportement sur un problème
OK = update_and_start!(stop, x = 1.0)
@test OK == false
@test stop.current_state.x == 1.0
# @test_throws ErrorException fill_in!(stop, 1.0) # on ne test pas, car
                                                  # la fonction fill_in!
                                                  # n'est pas exportée
@test start!(stop) == false
OK2 = update_and_stop!(stop, ht = 10.0)
@test OK2 == false
@test stop.current_state.ht == 10.0
@test stop!(stop) == false

# on vérifie que _stalled_check! fonctionne
update!(stop.current_state, dx = 0.0,  x = 1.0)
@test stop!(stop)
update!(stop.current_state, dx = 1.0, x = 0.0)
@test !(stop!(stop))
update!(stop.current_state, df = 0.0)
@test stop!(stop)
update!(stop.current_state, df = 1.0, x = 1.0)
@test !(stop!(stop))

# On vérifie que _tired_check fonctionne
update!(stop.current_state, tmps = 0.0)
@test stop!(stop)
update!(stop.current_state, tmps = NaN)
@test !(stop!(stop))
# on crée un deuxième stopping pour vérifier le nombre d'évaluations maximal
meta_ls = StoppingMeta(max_eval = 1)
stop2 = LS_Stopping(h, (x,y)-> armijo(x,y), lsatx, meta = meta_ls)
@test !(stop!(stop2))
obj(h, 1.0), obj(h, 0.0)
@test stop!(stop2)

# on vérifie que _unbounded_check! fonctionne
update!(stop.current_state, x = 1e100)
@test stop!(stop)


## vérifier les fonctions _optimality_check et _null_test
