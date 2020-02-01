###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# The following examples illustre the various possibilities offered by Stopping
#
###############################################################################

using Test, NLPModels, Stopping

printstyled("How to State ")
include("howtostate.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to State for NLP ")
include("howtostate-nlp.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to Stop ")
include("howtostop.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to Stop II ")
include("howtostop-2.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to Stop for NLP ")
include("howtostop-nlp.jl")
printstyled("passed ✓ \n", color = :green)
