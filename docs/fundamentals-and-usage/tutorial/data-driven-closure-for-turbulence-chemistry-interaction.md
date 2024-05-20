# Data-Driven Closure for Turbulence-Chemistry interaction

## Prerequisites

***

Before reading this tutorial, you should know a bit of Python. If you would like to refresh your memory, take a look at the [Python tutorial](https://docs.python.org/3/tutorial/).

We are going to work on the reduced [dataset](https://github.com/LorenzoPiu/aPriori/tree/main/data) that is available on the [project's GitHub page](https://github.com/LorenzoPiu/aPriori/tree/main). Make sure to download the data folder before starting.

## Learner Profile

This tutorial is aimed at a user who has a solid understanding of Computational Fluid Dynamics (CFD) for Combustion. A basic understanding of Machine learning is not required.&#x20;

The tutorial could also be beneficial for those who do not have strong competencies in CFD but know Data Science; in fact, apart from some computations to compute interesting quantities in reacting flows, the basic operations that we are going to perform on the data are based on filtering of 3D fields.

## Dataset Description

***

The dataset is extracted from a DNS simulation of a [lifted non-premixed hydrogen flame](https://blastnet.github.io/index.html). The variables saved comprise Temperature, Species Mass Fractions, 3 Velocity components, and Pressure.\
The chemical mechanism used comprises 9 species.

<figure><img src="../../.gitbook/assets/Dataset Description.png" alt=""><figcaption></figcaption></figure>

## What you will learn

You will learn:&#x20;

* how to use aPriori to compute the chemical source terms,
* how to filter the results obtained to resemble an LES field,
* and build a data-driven closure based on Neural Networks to model the subgrid interactions between Turbulence and Chemistry.

## A-priori methodology

<figure><img src="../../.gitbook/assets/A priori validation methodology.png" alt=""><figcaption></figcaption></figure>
