# DPMM.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ekinakyurek.github.io/DPMM.jl/latest)
[![](https://travis-ci.org/ekinakyurek/DPMM.jl.svg?branch=master)](https://travis-ci.org/ekinakyurek/DPMM.jl)

This repository is a research work on parallel dirichlet process mixture models and clustering on Julia by Ekin Akyürek with supervision of John W. Fischer III.

## Getting Started

Demo:
```julia
  gm = GridMixture(2)
  X, clabels = rand_with_label(gm,100000)
  fit(X; ncpu=3) # runs parallel split-merge algorithm
```

Visual Demo (requires OpenGL) :
```julia
  gm = GridMixture(2)
  X, clabels = rand_with_label(gm,100000)
  scene = setup_scene(X)
  fit(X; ncpu=3, scene=scene) # visualize parallel split-merge algorithm
```
For details please see the [function documentation](https://ekinakyurek.github.io/DPMM.jl/latest)

## [Technical Report](./docs/main.tex)

## Algorithms

1. Collapsed Gibbs Sampler
```julia
labels = fit(X; algorithm=CollapsedAlgorithm) # serial collapsed
```
2. Quasi-Collapsed Gibbs Sampler
```julia
labels = fit(X; algorithm=CollapsedAlgorithm, quasi=true) # quasi & serial collapsed
labels = fit(X; algorithm=CollapsedAlgorithm, quasi=true, ncpu=4) # quasi & parallel collapsed
```
3. Direct Gibbs Sampler
```julia
labels = fit(X; algorithm=DirectAlgorithm) # direct
labels = fit(X; algorithm=DirectAlgorithm ncpu=4) # parallel direct
```
4. Quasi-Direct Gibbs Sampler
```julia
labels = fit(X; algorithm=DirectAlgorithm, quasi=true) # quasi direct gibbs algorithm
labels = fit(X; algorithm=DirectAlgorithm, quasi=true, ncpu=4) # quasi & parallel direct gibbs direct gibbs
```
5. Split-Merge Gibbs Sampler
```julia
labels = fit(X; algorithm=SplitMergeAlgorithm) # split-merge
labels = fit(X; algorithm=SplitMergeAlgorithm, ncpu=4) # parallel split-merge
```

##  Parallel Benchmarking

Run below command:
```SHELL
julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --Kinit 1 --ncpu 4
```

* Results-I: Time (sec) to run 100 DP-GMM iterations for d=2, N=1e6, K=6.


Code        |   ncpu=1  |   ncpu=2  | ncpu=4 | ncpu=8 |
----------- | --------- | --------- | ------ | ------ |
C++         | 76.94     |   40.57   |  22.23 | 13.01      
DPMM.jl     | 103.90    |   53.11   |  26.45 | 16.42         
Julia-BNP   | 1101.97   |   572.50  | 345.58 | 172.30  
