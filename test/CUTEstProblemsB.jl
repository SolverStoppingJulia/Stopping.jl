using CUTEst

probs = open(readlines,"CUTEstBound.list")
#probs = filter(
#    x -> x != "MSQRTALS\n"  # very long to solve
#    && x != "MSQRTBLS\n",   # very long to solve
#    probs)

#probs = ["PENALTY3","PENALTY3"]
cute_probs = (CUTEstModel(p)  for p in probs)

