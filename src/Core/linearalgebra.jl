*(c::Cholesky,a::Real) = Cholesky(sqrt(a)*getfield(c,:factors),getfield(c,:uplo),getfield(c,:info))
*(a::Real,c::Cholesky) =  c * a
