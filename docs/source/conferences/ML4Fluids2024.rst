Machine Learning for Fluids, Paris 2024
=======================================

**Title:**  
*A data-driven approach to correct the cell reacting fraction in the partially-stirred reactor closure for LES of premixed flames*

**Event:** `ERCOFTAC ML4Fluids Conference <https://www.ercoftac.org/special_interest_groups/54-machine-learning-for-fluid-dynamics/>`_.

**Location:** Paris, France  

**Dates:** 6th - 8th March 2024

**Authors:**  
Lorenzo Piu, Arthur Péquin, Rodolfo Freitas, Salvatore Iavarone, Heinz Pitsch, Alessandro Parente

**Full article**

The results were selected for a special issue of the conference, so the work was extended and published. 
You can access the full article at `this link <https://link.springer.com/article/10.1007/s10494-024-00626-3>`_

**Slides:**  
:download:`Download presentation (PDF) </_static/pdf/ML4Fluids-Cypher.pdf>`

.. raw:: html

   <div style="display:flex; justify-content:center; margin-top:1em;">
       <iframe 
           src="../_static/pdf/ML4Fluids-2024.pdf" 
           width="100%" 
           height="500px"
           style="border:1px solid #ddd; border-radius:8px;">
       </iframe>
   </div>

Overview
--------

This presentation introduces a **machine-learning–enhanced closure model**
to improve the **Partially Stirred Reactor (PaSR)** formulation for
Large Eddy Simulation (LES) of turbulent premixed flames.

Turbulent combustion is characterized by strong interactions between
fluid dynamics and chemical kinetics across multiple spatial and temporal
scales. As shown in the presentation, this complexity makes accurate
prediction of reaction rates particularly challenging in practical CFD
simulations.

In LES, the filtered chemical source terms generally satisfy:

.. math::

   \dot{\omega}_k(T,Y) \neq \dot{\omega}_k(\bar{T}, \bar{Y})

which highlights the need for models capable of describing
**turbulence–chemistry interaction (TCI)**.

Limitations of the PaSR Model
-----------------------------

The classical PaSR closure estimates filtered reaction rates as

.. math::

   \overline{\dot{\omega}}_{k,PaSR} =
   \frac{\tau_c}{\tau_c + \tau_m} \;
   \overline{\dot{\omega}}_{k}

where:

- :math:`\tau_c` is the chemical timescale,
- :math:`\tau_m` is the turbulent mixing timescale,
- :math:`\overline{\dot{\omega}}_{k}` is the mass exchange between fine structures and surrounding fluid.

This formulation introduces a **cell reacting fraction**

.. math::

   \gamma_{PaSR} = \frac{\tau_c}{\tau_c + \tau_m}

which determines the fraction of the computational cell where reactions
are assumed to occur.

However, DNS-based a-priori analyses reveal significant discrepancies
between PaSR predictions and the true filtered reaction rates.

Machine Learning Correction
---------------------------

To address these limitations, a **Fully Connected Neural Network (FCNN)**
is used to predict a correction term for the reacting fraction:

.. math::

   \overline{\dot{\omega}}_k =
   (\gamma_{PaSR} + \gamma_{FCNN}) \,
   \overline{\dot{\omega}}_{k,LFR}

The network is trained using **filtered DNS data**, learning the mapping
between LES-accessible quantities and the correction term
:math:`\gamma_{FCNN}`.

Training Dataset
----------------

The model was trained on a **DNS of a turbulent premixed methane jet flame**
developed by Attili et al.

Main simulation parameters include:

- Equivalence ratio: :math:`\phi = 0.7`
- Inlet temperature: 800 K
- Jet velocity: 100 m/s
- Coflow velocity: 15 m/s
- Thermal flame thickness: ~110 µm

The DNS field was filtered at different filter sizes to generate
training data representative of LES conditions.

Neural Network Architecture
---------------------------

The FCNN architecture consists of:

- **Input variables**

  - Filtered progress variable :math:`\tilde{C}`
  - Chemical timescale :math:`\tau_c`
  - Mixing timescale :math:`\tau_m`

- **Network structure**

  - 3 input neurons
  - 6 hidden layers
  - 64 neurons per layer
  - 1 output neuron (:math:`\gamma_{FCNN}`)

The model is trained by minimizing the error between predicted and DNS
heat release rates.

Role of Spatial Information
---------------------------

Including spatial information significantly improves model performance.
Additional inputs such as

- :math:`\nabla \tilde{C}`
- :math:`\nabla^2 \tilde{C}`

allow the network to better capture local flame structure.

As shown in the presentation results, models including spatial
information reduce prediction errors by more than one order of magnitude.

Generalization Across Filter Sizes
----------------------------------

A key challenge for machine-learning closures is **generalization across
filter sizes**.

Training the model using data filtered at a single filter width leads
to poor generalization when applied to different LES resolutions.

The study shows that training on **multiple filter sizes**
(:math:`\Delta = 2, 3, 6, 9`) significantly improves robustness and
accuracy for unseen filter widths.

Key Findings
------------

The main conclusions of the study are:

- Machine learning can significantly improve PaSR predictions of filtered
  reaction rates.
- Including spatial derivatives of the progress variable improves both
  accuracy and sparsity of the model.
- Training on multiple filter sizes improves generalization across LES
  resolutions.
- Optimal predictions require the corrected reacting fraction to extend
  beyond the traditional physical bounds :math:`0 \le \gamma \le 1`.

Acknowledgments
---------------

Lorenzo Piu has received funding from the European Union’s Horizon 
Europe research and innovation programme under the Marie Skłodowska-Curie 
grant agreement No 101072779 (ENCODING). 
The results of this publication/presentation reflect only the author(s) 
view and do not necessarily reflect those of the European Union. 
The European Union cannot be held responsible for them.
