"""

    readNYTimes(file::AbstractString)

    Read NYTimes dataset from given data file. It returns DPSparseMatrix
"""
function readNYTimes(file::AbstractString=dir("data/docword.nytimes.txt"),
                     entry::Int=69679427)
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

function printNYClusters(X, labels; modelType=DPMNMM)
    model = modelType(X)
    uniquez  = unique(labels)
    stats = SuffStats(model, X, labels)
    words = readlines(dir("data/vocab.nytimes.txt"))
    for (k,stat) in stats
          fname = dir("clusters/",string(k,".txt"))
          println("writing to $fname")
          f = open(fname,"w")
          println(f,"word\tfreq")
          α = stat.s
          v = sortperm(α; rev=true)
          for i in v
               freq = α[i]
               word = words[i]
        	   if freq != 0
                   	println(f,"$(freq)\t$(word)")
        	   end
          end
          close(f)
    end
end
