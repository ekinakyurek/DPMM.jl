function readNYTimes(file, entry::Int=69679427)
    J=Array{Int}(undef,entry);
    Ir= Array{Int}(undef,entry);
    V = Array{Int}(undef,entry);
    f = open(file,"r")
    for i=1:3; readline(f); end
    for i=1:entry
        tokens    = split(readline(f),' ')
        J[i]  = parse(Int,tokens[1])
        Ir[i] = parse(Int,tokens[2])
        V[i]  = parse(Int,tokens[3])
    end
    return DPSparseMatrix(sparse(Ir,J,V))
end
