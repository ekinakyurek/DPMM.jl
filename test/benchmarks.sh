for N in 10000 100000 1000000
do
for ncpu in 2 4 8
do
   echo $ncpu
   for K in 6
   do
	for alpha in 1.0
	do
	  for D in 2 5 10
		do
	    	    julia benchmark.jl --N $N --K $K --Kinit $K --alpha $alpha --D $D --ncpu $ncpu
	            sleep 1
         	done
	  done
  done
done
done
