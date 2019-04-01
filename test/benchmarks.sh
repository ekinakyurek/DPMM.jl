for N in 100 1000 10000 100000 1000000
do
   for K in 2 5 10 20
   do
	for alpha in 1.0 5.0 10.0
	do
	  for D in 2 10 50
		do
	    	    julia benchmark.jl --N $N --K $K --Kinit $K --alpha $alpha --D $D
	            sleep 1
         	done
	done
  done
done
