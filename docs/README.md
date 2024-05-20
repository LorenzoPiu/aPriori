---
description: A Python package to process combustion DNS.
cover: .gitbook/assets/Logo-0.0.9.png
coverY: 0
layout:
  cover:
    visible: true
    size: hero
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 👋 Welcome to aPriori

{% hint style="warning" %}
**Disclaimer:** The guide is not complete yet. Please contact me at lorenzo.piu@ulb.be in case you need anything. In the meantime, I am working to complete the documentation.
{% endhint %}

## Overview

This guide explains how to use the library and the main features.&#x20;

The first section [Getting Started](broken-reference) introduces the scope of the project, explains how to install the package in your environment and gives a code that can be used immediately to perform analysis on a DNS of a hydrogen flame. The second section [Fundamentals and Usage](broken-reference) goes into more detail, giving examples of how to use the single blocks of the library to obtain the maximum from it. If you didn't understand much from the code in the quickstart, this section is the right one to get more insight. The third section [Library Structure](broken-reference) covers more in detail the code from a developer's point of view, explaining the single functions and the main classes with their attributes and methods. If you are interested in the library from the point of view of a user, you can skip this section.

***

### Getting Started

* [What is aPriori?](getting-started/what-is-apriori.md)
* [Installation](getting-started/installation.md)
* [Quickstart](getting-started/quickstart.md)

### Fundamentals and Usage

* [aPriori Fundamentals](fundamentals-and-usage/apriori-fundamentals/)
  * [Data Formatting](fundamentals-and-usage/apriori-fundamentals/data-formatting.md)
  * [Using Scalar3D class](fundamentals-and-usage/apriori-fundamentals/using-scalar3d-class.md)
  * [Using Mesh3D class](fundamentals-and-usage/apriori-fundamentals/using-mesh3d-class.md)
  * [Using Field3D class](fundamentals-and-usage/apriori-fundamentals/using-field3d-class.md)
  * [Plot Utilities](fundamentals-and-usage/apriori-fundamentals/plot-utilities.md)
* [Tutorial](fundamentals-and-usage/tutorial/)
  * [Data-Driven Closure for Turbulence-Chemistry Interaction](fundamentals-and-usage/tutorial/data-driven-closure-for-turbulence-chemistry-interaction.md)

### API Guide

* [Scalar3D class](api-guide/scalar3d-class.md)
* [Mesh3D class](api-guide/mesh3d-class.md)
* [Field3D class](api-guide/field3d-class.md)
