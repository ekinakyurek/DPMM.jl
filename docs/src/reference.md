# Reference

**Contents**

```@contents
Pages = ["reference.md"]
```

## Algorithms

```@docs
DPMM.fit
DPMM.DPMMAlgorithm
DPMM.CollapsedAlgorithm
DPMM.DirectAlgorithm
DPMM.SplitMergeAlgorithm
DPMM.run!
DPMM.setup_workers
DPMM.initialize_clusters
```

## Algorithms (Internal)
```@docs
DPMM.random_labels
DPMM.create_clusters
DPMM.empty_cluster
DPMM.RestrictedClusterProbs
DPMM.CRPprobs
DPMM.SampleSubCluster
DPMM.ClusterProbs
DPMM.place_x!
DPMM.label_x
DPMM.logmixture_πs

```

## Distributions
```@docs
DPMM.NormalWishart
DPMM.MvNormalFast
DPMM.DirichletFast
DPMM.MultinomialFast
```

## Clusters
```@docs
DPMM.AbstractCluster
DPMM.lognαpdf
DPMM.logαpdf
DPMM.population
DPMM.CollapsedCluster
DPMM.DirectCluster
DPMM.SplitMergeCluster
```

## Models

```@docs
DPMM.AbstractDPModel
DPMM.DPGMM
DPMM.DPMNMM
DPMM.DPGMMStats
DPMM.DPMNMMStats
```

## Data
```@docs
DPMM.setup_scene
DPMM.readNYTimes
DPMM.GridMixture
DPMM.RandMixture
DPMM.DPSparseMatrix
DPMM.DPSparseVector
```

## Function Index

```@index
Pages = ["reference.md"]
```
