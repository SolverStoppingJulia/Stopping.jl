# on rétulise le nlp des tests unitaire de linesearch
nlp_at_x = NLPAtX(0.0)
stop_nlp = NLPStopping(nlp, (x,y) -> unconstrained(x,y), nlp_at_x)

# comme on a pas mal testé toutes les fonctions dans les tests unitaires de
# line search on regarde seulement la fonction fill_in qui fonctionne seulement
# pour les NLPStoppings

a = ones(5000)
fill_in!(stop_nlp, a)
@test obj(nlp, a) == stop_nlp.current_state.fx
@test grad(nlp, a) == stop_nlp.current_state.gx
