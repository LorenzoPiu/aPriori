aPriori.DNS.Field
===================

.. py:class:: aPriori.DNS.Field(folder_path, reactive=True, verbose=True)

   Class representing a 3D field with various attributes and methods for visualization and data management.

   Description:
   ------------

   The `Field` class encapsulates a 3D field used in DNS, allowing for
   operations such as data loading, filtering, and accessing various properties
   of the field. This class is central to handling the volumetric data in
   computational fluid dynamics simulations.

   The field to use as input must be formatted as Blastnet's datasets:
   https://blastnet.github.io

   Attributes:
   -----------

   variables : dict
       Dictionary containing variable names and their corresponding settings.

   mesh : Mesh object
       Object containing the mesh. See the class Mesh for more information

   __field_dimension : int
       Dimensionality of the field, always set to 3


   Methods:
   --------

   __init__(self, folder_path):
       Constructor method to initialize a 3D field object.

       dynamic attributes : Scalar objects
       depending on the files contained in the main folder folder_path,
       initializes various attributes as Scalar objects.
       Example :
       Field attributes :
       +-------------+---------------------------------------------------------+
       |  Attribute  |                          Path                           |
       +-------------+---------------------------------------------------------+
       |    S_DNS    |     ../data/DNS_DATA_TEST/data/S_DNS_s-1_id000.dat      |
       |      P      |        ../data/DNS_DATA_TEST/data/P_Pa_id000.dat        |
       |     RHO     |     ../data/DNS_DATA_TEST/data/RHO_kgm-3_id000.dat      |
       |      T      |        ../data/DNS_DATA_TEST/data/T_K_id000.dat         |
       |     U_X     |      ../data/DNS_DATA_TEST/data/UX_ms-1_id000.dat       |
       |     U_Y     |      ../data/DNS_DATA_TEST/data/UY_ms-1_id000.dat       |
       |     U_Z     |      ../data/DNS_DATA_TEST/data/UZ_ms-1_id000.dat       |
       |     YH2     |        ../data/DNS_DATA_TEST/data/YH2_id000.dat         |
       |     YO2     |        ../data/DNS_DATA_TEST/data/YO2_id000.dat         |
       |    YH2O     |        ../data/DNS_DATA_TEST/data/YH2O_id000.dat        |
       |     YH      |         ../data/DNS_DATA_TEST/data/YH_id000.dat         |
       |     YO      |         ../data/DNS_DATA_TEST/data/YO_id000.dat         |
       |     YOH     |        ../data/DNS_DATA_TEST/data/YOH_id000.dat         |
       |    YHO2     |        ../data/DNS_DATA_TEST/data/YHO2_id000.dat        |
       |    YH2O2    |       ../data/DNS_DATA_TEST/data/YH2O2_id000.dat        |
       |     YN2     |        ../data/DNS_DATA_TEST/data/YN2_id000.dat         |
       |   RH2_DNS   |  ../data/DNS_DATA_TEST/data/RH2_DNS_kgm-3s-1_id000.dat  |
       |   RO2_DNS   |  ../data/DNS_DATA_TEST/data/RO2_DNS_kgm-3s-1_id000.dat  |
       |  RH2O_DNS   | ../data/DNS_DATA_TEST/data/RH2O_DNS_kgm-3s-1_id000.dat  |
       |   RH_DNS    |  ../data/DNS_DATA_TEST/data/RH_DNS_kgm-3s-1_id000.dat   |
       |   RO_DNS    |  ../data/DNS_DATA_TEST/data/RO_DNS_kgm-3s-1_id000.dat   |
       |   ROH_DNS   |  ../data/DNS_DATA_TEST/data/ROH_DNS_kgm-3s-1_id000.dat  |
       |  RHO2_DNS   | ../data/DNS_DATA_TEST/data/RHO2_DNS_kgm-3s-1_id000.dat  |
       |  RH2O2_DNS  | ../data/DNS_DATA_TEST/data/RH2O2_DNS_kgm-3s-1_id000.dat |
       |   RN2_DNS   |  ../data/DNS_DATA_TEST/data/RN2_DNS_kgm-3s-1_id000.dat  |
       +-------------+---------------------------------------------------------+

   build_attributes_list(self):
       Build lists of attribute names and corresponding file paths based on the configuration specified in variables_list.

   check_valid_attribute(self, input_attribute):
       check if the input attribute is assigned to the Field object.

   compute_chemical_timescale(self, mode='SFR', verbose=False):
       Computes the chemical timescale for the field, useful in Partially Stirred Reactor (PaSR) modeling.

   compute_kinetic_energy(self):
       Computes the velocity module and saves it to a file.

   compute_mixing_timescale(self, mode='Kolmo'):
       Computes the mixing timescale for the field, useful for turbulence modeling.

   compute_reaction_rates(self, n_chunks = 5000):
       Computes the source terms for a given chemical reaction system.
       The reaction rates are computed using Arrhenius equation for
       the given kinetic mechanism.

   compute_reaction_rates_batch(self, n_chunks = 5000, tau_c='SFR', tau_m='Kolmo'):
       Computes the source terms for a given chemical reaction system.
       The reaction rates are computed integrating an ideal reactor in time.
       This formulation corresponds to the one used in the PaSR model, without
       multiplying the output for the cell reacting fraction gamma

   compute_residual_kinetic_energy(self, mode='Yosh'):
       Function to compute the residual kinetic energy.

   compute_residual_dissipation_rate(self, mode='Smag'):
       Computes the residual dissipation rate for a filtered velocity field

   compute_strain_rate(self, save_derivatives=False, save_tensor=True, verbose=False):
       This function computes the strain rate of the velocity
       components (U, V, W) over a 3D mesh.

   compute_tau_r(self, mode='Smag', save_tensor_components=True):
       Computes the anisotropic part of the residual stress tensor, denoted as \(      au_r\).

   compute_velocity_module(self):
       Computes the velocity module and saves it to a file.

   cut(self, cut_size, mode='xyz'):
       Cut a field into a section based on a specified cut size.

   filter_favre(self, filter_size, filter_type='Gauss'):
       Filter every scalar in the field object using Favre-averaging.

   filter(self, filter_size, filter_type='Gauss'):
       Filter every scalar in the field object.

   find_path(self, attr):
       Finds a specified attribute in the attributes list and returns the corresponding element
       in the paths list.

   plot_x_midplane(self, attribute):
       Plot the midplane along the x-axis for the specified attribute.

   plot_y_midplane(self, attribute):
       Plot the midplane along the y-axis for the specified attribute.

   plot_z_midplane(self, attribute):
       Plot the midplane along the z-axis for the specified attribute.

   print_attributes(self):
       Prints the valid attributes of the class and their corresponding file paths.

   update(self, verbose=False):
       Update the attributes of the class based on the existence of files in
       the specified data path.


   .. py:attribute:: variables


   .. py:attribute:: mesh


   .. py:attribute:: folder_path


   .. py:attribute:: data_path


   .. py:attribute:: chem_path


   .. py:attribute:: grid_path


   .. py:attribute:: reactive
      :value: True



   .. py:attribute:: filter_size


   .. py:attribute:: downsampled
      :value: False



   .. py:attribute:: shape


   .. py:attribute:: id_string


   .. py:attribute:: X_m_path


   .. py:attribute:: Y_m_path


   .. py:attribute:: Z_m_path


   .. py:method:: add_variable(attr_name, file_name)


   .. py:method:: build_attributes_list(variable=None)

      Build lists of attribute names and corresponding file paths
      based on the configuration specified in variables_list.
      This operation is needed to handle different variables depending on
      the kinetic mechanism, and to build attributes corresponding to different
      models.

      Returns:
      --------
      :
      attr_list : int
          List of attribute names.

      paths_list : list
          List of corresponding file paths.



   .. py:method:: check_valid_attribute(input_attribute)

      Check if the input attribute is valid.

      Parameters:
      - input_attribute (str): The attribute to be checked for validity.

      Raises:
      - ValueError: If the input_attribute is not valid.



   .. py:method:: compute_chemical_timescale(mode='SFR', verbose=False, threshold=1e-15, replace_nonreacting=False)

      Computes the chemical timescale for the field, useful in Partially Stirred Reactor (PaSR) modeling.

      Description:
      ------------
      This method calculates the chemical timescale for the field using different modes. The available modes are 'SFR' (Slowest Fastest Reaction), 'FFR' (Fastest Fastest Reaction), and 'Ch' (Chomiak timescale). The computation is valid only for filtered fields, and it leverages reaction rates and species molar concentrations.

      Parameters:
      -----------
      mode : str, optional
          The mode of timescale computation. Valid options are 'SFR', 'FFR', and 'Ch'. Default is 'SFR'.
      verbose : bool, optional
          If True, prints detailed information during the computation. Default is False.
      threshold: float, optional
          Minimum value of the species formation rates to consider the cell reactive

      Raises:
      -------
      ValueError
          If the field is not a filtered field or if the length of species and reaction rates lists do not match the number of species.
      AttributeError
          If the required attributes 'fuel' and 'ox' are not defined when mode is 'Ch'.

      Workflow:
      ---------
      1. Validation
         - Checks if `mode` is valid and if the field is filtered.
         - Ensures species and reaction rates lists match the number of species.

      2. Mode: 'SFR' or 'FFR'
         - Collects paths to reaction rates and species concentrations.
         - Computes the timescales \( au_c^{SFR}\) and \(     au_c^{FFR}\).
         - Saves the computed timescales to files.

      3. Mode: 'Ch'
         - Validates the presence of 'fuel' and 'ox' attributes.
         - Computes the Chomiak timescale.
         - Saves the computed timescale to a file.

      Returns:
      --------
      :
      None



   .. py:method:: compute_chi_z()

      Compute and save the scalar dissipation rate Chi_Z.

      This function calculates the scalar dissipation rate Chi_Z using the mixture
      fraction Z, thermal conductivity Lambda, density RHO, specific heat Cp, and
      the gradient of Z. The result is saved to a file.

      Prerequisites:
      --------------
      - The mixture fraction Z must be computed and available as an attribute.
      - The thermal conductivity Lambda and specific heat Cp must be computed.
      - The gradient of Z (Z_grad) should be available or will be computed if missing.

      Raises:
      ------
      ValueError
          If the mixture fraction Z is not available. The error message includes
          instructions on how to compute Z using the compute_mixture_fraction() method.
      ValueError
          If Lambda or Cp are not available. The error message includes instructions
          on how to compute these using the compute_transport_properties() method.

      Returns:
      -------
      :
      None
          The function saves the computed Chi_Z to a file but does not return any value.

      Side Effects:
      -------------
      - Computes Z_grad if not already available by calling self.compute_z_grad().
      - Saves the computed Chi_Z to a file using the save_file() function.

      Note:
      -----
      This function assumes the existence of compute_z_grad(), save_file(), and
      find_path() methods, as well as RHO, Lambda, and Cp attributes.



   .. py:method:: compute_csp(n_chunks=1000, TSR=True, API_TSR=True, API_species=False, n_proc=None, parallel=False, debug=False, overwrite=True, verbose=True)


   .. py:method:: compute_gradient_C()


   .. py:method:: compute_laplacian_C()


   .. py:method:: compute_kinetic_energy()

      Computes the velocity module and saves it to a file.

      This method calculates the velocity module by squaring the values of U_X, U_Y, and U_Z,
      summing them up, and then taking the square root of the result. The computed velocity
      module is then saved to a file using the `save_file` function. The file path is determined
      by the `find_path` method with 'U' as the argument. After saving the file, the `update`
      method is called to refresh the attributes of the class.

      Note:
      -----

      - `self.U_X`, `self.U_Y`, and `self.U_Z` are assumed to be attributes of the class
        representing components of velocity. Make sure to check you have the relative files in your
        data folder. To check, use the method <your_field_name>.print_attributes.



   .. py:method:: compute_M(verbose=False)

      Compute the ratio of resolved to total turbulent kinetic energy (M).

      This method calculates M, which is the ratio of the resolved turbulent kinetic energy (TKE)
      to the total TKE. It requires both filtered and unfiltered (DNS) data.

      Parameters:
      -----------
      verbose : bool, optional
          If True, print additional information during the computation. Default is False.

      Returns:
      --------
      :
      None
          The computed M is saved to a file and the object is updated.

      Raises:
      -------
      ValueError
          If the field is not filtered (filter_size == 1) or if an unsupported filter type is used.
      AttributeError
          If the DNS_folder_path is not set.

      Notes:
      ------
      - This method only works on filtered fields. Ensure the field is filtered before calling.
      - The DNS_folder_path must be set to the location of the unfiltered DNS data.
      - The method uses the following formula to compute M:
        M = K_LES / K_DNS
        where K_LES = 0.5 * (filt(U_X)^2 + filt(U_Y)^2 + filt(U_Z)^2)
        and K_DNS = filt(0.5 * (U_X^2 + U_Y^2 + U_Z^2))
      - The computed M is saved to a file and the object is updated.
      - Supports 'box' and 'gauss' filter types, determined from the folder path.
      - Can handle both regular and Favre filtering, also determined from the folder path.

      Example:
      --------
      >>> filtered_field = Field('path/to/filtered/data')
      >>> filtered_field.DNS_folder_path = 'path/to/unfiltered/DNS/data'
      >>> filtered_field.compute_M(verbose=True)



   .. py:method:: compute_mixing_timescale(mode='Kolmo')

              Computes the mixing timescale for the field, useful for turbulence modeling.

              Description:
              ------------
              This method calculates the mixing timescale using either the Kolmogorov model or the integral length scale model. The computation relies on residual kinetic energy and residual dissipation rate fields.

              Parameters:
              -----------
              mode : str, optional
                  The mode of timescale computation. Valid options are 'Kolmo' (Kolmogorov) or 'Int' (Integral length scale). Default is 'Kolmo'.

              Raises:
              -------
              ValueError
                  If the specified mode is not valid.
              Warning
                  If the 'C_mix' attribute is not defined for the integral length scale model.

              Workflow:
              ---------
              1. Validation
                 - Checks if `mode` is valid.

              2. Mode: 'Kolmo'
                 - Computes the Kolmogorov timescale:
                   \[
                      au_m^{Kolmo} = \sqrt{
      rac{k_r}{\epsilon_r} \sqrt{
      rac{\mu}{
      ho \epsilon_r}}}
                   \]
                 - Saves the computed timescale to a file.

              3. Mode: 'Int'
                 - Ensures `C_mix` is defined, initializes to 0.1 if not.
                 - Computes the integral length scale timescale:
                   \[
                      au_m^{Int} =
      rac{C_{mix} k_r}{\epsilon_r}
                   \]
                 - Saves the computed timescale and `C_mix` value to files.

              Returns:
              --------
              None

              Example:
              --------
              >>> field = Field('your_folder_path')
              >>> field.compute_mixing_timescale(mode='Kolmo')




   .. py:method:: compute_mixture_fraction(Y_ox_2, Y_f_1, s)

      Calculate the stoichiometric mixture fraction (Z_st) for a combustion
      system based on the mass fractions of oxidizer and fuel in different
      streams and the stoichiometric ratio.

      Parameters:
      -----------
      Y_ox_2 : float
          Mass fraction of the oxidizer in the coflow (ambient) stream.

      Y_f_1 : float
          Mass fraction of the fuel in the fuel stream.

      s : float
          Stoichiometric molar ratio of fuel to oxidizer (Fuel/Oxidizer).

      Returns:
      --------
      :
      Z_st : float
          Stoichiometric mixture fraction, representing the mass fraction of the fuel stream
          required to achieve stoichiometric combustion.

      Description:
      ------------
      This function calculates the mixture fraction Z and the stoichiometric mixture fraction
      Z_st based on the input fuel and oxidizer mass fractions, as well as the stoichiometric
      ratio s. The oxidizer and fuel molecular weights are determined using Cantera.
      The results are saved to files, and the function returns the stoichiometric
      mixture fraction Z_st.

      Formula:
      --------
      The mixture fraction Z is computed using the following formula:

      Z = (ν * Y_f - Y_ox + Y_ox2) / (ν * Y_f1 + Y_ox2)

      where:

      ν = M_ox / (s * M_fuel)

      - M_ox is the molar mass of the oxidizer (kg/kmol),
      - M_fuel is the molar mass of the fuel (kg/kmol),
      - s is the stoichiometric molar ratio (Fuel/Oxidizer),
      - Y_f and Y_ox are the fuel and oxidizer mass fractions in the current state,
      - Y_f1 is the fuel mass fraction in the fuel stream,
      - Y_ox2 is the oxidizer mass fraction in the coflow stream.

      The stoichiometric mixture fraction Z_st is calculated as:

      Z_st = Y_ox2 / (ν * Y_f1 + Y_ox2)

      This formulation helps determine the mass fraction of the fuel stream required to achieve
      stoichiometric combustion.



   .. py:method:: compute_progress_variable(species=None, Y_b=None, Y_u=None, reactant=False)

      Computes the progress variable based on the mass fraction of a specified species
      (default is 'O2') and saves the result.

      Parameters:
      -----------
      specie : str, optional
          The chemical species for which the progress variable is computed. If None,
          the default is 'O2'.
      Y_b : float, optional
          The mass fraction of the species in the burnt gas. If not provided, it defaults
          to the maximum value of the species mass fraction.
      Y_u : float, optional
          The mass fraction of the species in the unburnt gas. If not provided, it defaults
          to the minimum value of the species mass fraction.

      Returns:
      --------
      :
      None
          The progress variable C is saved to a file, and the object is updated.

      Notes:
      ------
      The progress variable C is computed using the formula:

          C = 1 - (Y - Y_u) / (Y_b - Y_u)

      where Y is the mass fraction of the specified species, Y_u is the mass fraction
      in the unburnt state, and Y_b is the mass fraction in the burnt state.



   .. py:method:: compute_progress_variable_fluxes()


   .. py:method:: compute_residual_kinetic_energy(mode='Yosh')

      Description
      -----------
      Function to compute the residual kinetic energy.

      Real value computed with information at DNS level:

          .. math::   k_{SGS} = \bar{U_i U_i} - \bar{U_i} \bar{U_i}

      Yoshizawa expression:

          .. math::  k_{SGS} = 2 C_I \bar{\rho} \Delta^2 |\tilde{S}|^2

      :param mode: The default is 'Yosh'.
      :type mode: TYPE, optional

      :rtype: None.



   .. py:method:: compute_residual_dissipation_rate(mode='Smag')

      This function computes the residual dissipation rate for a filtered velocity field,
      based on the specified mode: 'DNS' or 'Smag'. It requires that the field has been
      filtered and performs different calculations depending on the selected mode.

      Parameters:
      ----------
      mode : str, optional
          The mode of operation. It can be either 'Smag' or 'DNS'. Defaults to 'Smag'.

          - 'Smag': Uses the Smagorinsky model to compute the residual dissipation rate.
          - 'DNS': Uses Direct Numerical Simulation data to compute the residual dissipation rate.

      Returns:
      --------
      :
      None:
          The function does not return any values but saves the computed residual dissipation rate
          as a file in the main folder of the field.

      Raises:
      -------
      ValueError:
          - If the field is not a filtered field (i.e., `filter_size` is 1).
          - If the filter type used is not 'box' or 'gauss'.

      AttributeError:
          - If the 'DNS' mode is selected and the `DNS_folder_path` attribute is not set.
          - If the 'Smag' mode is selected and the `S_LES` attribute is not set.

      .. warning:: - If the 'Smag' mode is selected and the `Cs` attribute is not set, it initializes `Cs` to 0.1 by default.

      Detailed Description:
      ---------------------
      This function first updates the internal state of the field. It then checks the validity
      of the provided mode against the allowed modes stored in the `variables` dictionary.

      If the field is not filtered (i.e., `filter_size` is 1), it raises a `ValueError`
      indicating that residual quantities can only be computed for filtered fields and provides
      instructions on how to filter the field.

      Depending on the mode, the function performs different computations:

      1. **DNS Mode**:
          - Ensures the `DNS_folder_path` attribute is set, raising an `AttributeError` if not.
          - Loads the associated unfiltered DNS field.
          - Determines the filter type (either 'box' or 'gauss') used for the folder to ensure consistency.
          - Computes the anisotropic residual stress tensor and then the residual dissipation rate using the
            filtered DNS field and the LES strain rate.
          - Saves the computed residual dissipation rate to a file.

      2. **Smag Mode**:
          - Checks if the `Cs` attribute is set, issuing a warning and initializing `Cs` to 0.1 if not.
          - Ensures the `S_LES` attribute is set, raising an `AttributeError` if not.
          - Computes the residual dissipation rate using the Smagorinsky model.
          - Saves the computed residual dissipation rate to a file.

      Finally, the function updates the internal state of the field again.



   .. py:method:: compute_reaction_rates(n_chunks=1000, parallel=False, n_proc=None, exist_ok=False, overwrite=False)

      Computes the source terms for a given chemical reaction system.

      This function performs several steps:
      1. Checks that all the mass fractions are in the folder.
      2. Determines if the reaction rates to be computed are in DNS or LFR mode based on the filter size.
      3. Builds a list with reaction rates paths and one with the species' Mass fractions paths.
      4. Checks that the files of the reaction rates do not exist yet. If they do, asks the user if they want to overwrite them.
      5. Computes the reaction rates in chunks to handle large data sets efficiently.
      6. Saves the computed reaction rates, heat release rate, and dynamic viscosity to files.
      7. Updates the object's state.

      Parameters:
      n_chunks (int, optional): The number of chunks to divide the data into for efficient computation. Default is 5000.

      Returns:
      None

      Raises:
      SystemExit: If the user chooses not to overwrite existing reaction rate files, or if there is a mismatch in the number of species and the length of the species paths list.

      Note:
      This function uses the Cantera library to compute the reaction rates, heat release rate, and dynamic viscosity. It assumes that the object has the following attributes: attr_list, bool_list, folder_path, filter_size, species, shape, kinetic_mechanism, T, P, and paths_list. It also assumes that the object has the following methods: find_path and update.



   .. py:method:: compute_reaction_rates_batch(n_chunks=1000, tau_c='SFR', tau_m='Kolmo', tau_star_max=0.1, parallel=False, n_proc=None, exist_ok=False, overwrite=False)

      Computes the reaction rates in batches for a filtered field.

      Description:
      ------------
      This method calculates reaction rates in chunks for a filtered field, suitable for large datasets.
      The reaction rates can be computed in different modes specified by `tau_c` and `tau_m`.

      Parameters:
      -----------
      n_chunks : int, optional
          Number of chunks to divide the field into for batch processing. Default is 5000.
      tau_c : str, optional
          Mode for computing the chemical timescale. Default is 'SFR'.
      tau_m : str, optional
          Mode for computing the mixing timescale. Default is 'Kolmo'.
      n_proc : int, optional
          Number of worker processes to use. Default is half the number of CPUs.

      Raises:
      -------
      ValueError
          If the field is not filtered or if the species' molar concentrations and reaction rates are not in the data folder.
      AttributeError
          If required attributes (`attr_list`, `bool_list`, `folder_path`) are not defined.

      Returns:
      --------
      :
      None



   .. py:method:: compute_strain_rate(save_derivatives=False, save_tensor=True, verbose=False)

      Computes the strain rate or the derivatives of the velocity components (U, V, W) over a 3D mesh.

      Parameters:
      ----------
      save_derivatives : bool, optional
          If True, saves the derivatives of the velocity components as files in the main folder. Defaults to False.

      save_tensor : bool, optional
          If True, saves the strain rate tensor as a file in the main folder. Defaults to True.

      verbose : bool, optional
          If True, prints out progress information. Defaults to False.

      Returns:
      --------
      :
      None

      Raises:
      -------
      TypeError:
          If U, V, W are not instances of Scalar or if mesh is not an instance of Mesh.
      ValueError:
          If U, V, W and mesh do not have the same shape.

      Workflow:
      ---------
      1. Preprocess
         - Sets the closure type based on the filter size.
         - Retrieves necessary attributes and initializes variables.

      2. Compute Derivatives
         - Computes derivatives of the velocity components in all three directions.
         - Saves derivatives as files if `save_derivatives` is True.

      3. Compute Strain Rate Tensor
         - Computes the strain rate tensor components by averaging appropriate velocity component derivatives.
         - Saves the strain rate tensor as a file if `save_tensor` is True.

      Example:
      --------
      >>> field = Field('your_folder_path')
      >>> field.compute_strain_rate(save_derivatives=True, save_tensor=True, verbose=True)



   .. py:method:: compute_tau_r(mode='Smag', save_tensor_components=True)

              Computes the anisotropic part of the residual stress tensor, denoted as \(      au_r\),
              for a given field in computational fluid dynamics simulations. The function can
              operate in two modes: 'Smag' and 'DNS'.

              Description:
              ------------
              $\(     au_r\)$ (TAU_r) is the **anisotropic part** of the residual stress tensor.

              Residual stress tensor:
              \[
                      au^R_{i,j} = \widetilde{(U_i U_j)} - \widetilde{U}_i \cdot \widetilde{U}_j
              \]

              Anisotropic part:
              \[
                      au^r_{i,j} =    au^R_{i,j} -
      rac{2}{3} k_r \cdot \delta_{i,j}
              \]

              where \( k_r \) is the residual kinetic energy:
              \[
              k_r =
      rac{1}{2} \left( \widetilde{(U_i U_i)} - \widetilde{U}_i \cdot \widetilde{U}_i
      ight) =
      rac{1}{2} \left( \widetilde{(U_i^2)} - \left(\widetilde{U}_i
      ight)^2
      ight)
              \]

              Parameters:
              -----------
              mode : str, optional
                  Mode of operation, either 'Smag' for the Smagorinsky model or 'DNS' for
                  Direct Numerical Simulation data. Default is 'Smag'.

              Raises:
              -------
              ValueError
                  If the field is not a filtered field (i.e., `self.filter_size == 1`).

              AttributeError
                  If required attributes (`Cs`, `S_LES`, `DNS_folder_path`) are not defined.

              Returns:
              --------
              None

              Workflow:
              ---------
              1. Initial Setup and Validation
                 - The function starts by updating the field and checking if the field is filtered.
                 - If `self.filter_size == 1`, it raises a `ValueError` because residual quantities computation only makes sense for filtered fields.

                  2. Mode: 'Smag' (Smagorinsky Model)
                     - Turbulent Viscosity:
                       - Checks if the Smagorinsky constant (`Cs`) is defined. If not, it initializes `Cs` to 0.1.
                       - Computes the turbulent viscosity (\(\mu_t\)) using:
                         $ \mu_t = (Cs \cdot \Delta \cdot l)^2 \cdot S_{LES} $
                         where \(\Delta\) is the filter size, \(l\) is the grid size, and \(S_{LES}\) is the strain rate at LES scale.
                     - Anisotropic Residual Stress Tensor:
                       - Initializes `Tau_r` as a zero matrix.
                       - For each component \((i, j)\) of the tensor:
                         - Computes \(        au^r_{ij} = -2\mu_t S_{ij}^{LES} \).
                         - Adjusts for compressibility by subtracting the isotropic part (\(S_{iso}\)) when \(i = j\).
                         - Computes the squared components and accumulates them.
                         - Saves the computed \(      au^r_{ij}\) to a file.

                  3. Mode: 'DNS' (Direct Numerical Simulation)
                     - DNS Data Setup:
                       - Checks if the path to DNS data is defined.
                       - Initializes a `DNS_field` object to read DNS data.
                       - Determines the type of filter used (box or Gaussian).
                     - Residual Kinetic Energy:
                       - Computes residual kinetic energy \( K_r^{DNS} \) as:
                         \[ K_r^{DNS} = 0.5 \left( U_x^2 + U_y^2 + U_z^2
      ight)_{DNS} - 0.5 \left( U_x^2 + U_y^2 + U_z^2
      ight) \]
                       - Saves \( K_r^{DNS} \) to a file.
                     - Anisotropic Residual Stress Tensor:
                       - Initializes `Tau_r` as a zero matrix.
                       - For each component \((i, j)\) of the tensor:
                         - Computes the filtered product \(\widetilde{(U_i U_j)}_{DNS}\).
                         - Calculates \(      au^r_{ij}\) as:
                           \[         au^r_{ij} = \widetilde{(U_i U_j)}_{DNS} - \widetilde{U}_i \widetilde{U}_j - \delta_{ij}
      rac{2}{3} K_r^{DNS} \]
                         - Computes the squared components and accumulates them.
                         - Saves the computed \(      au^r_{ij}\) to a file.




   .. py:method:: compute_temperature_fluxes()


   .. py:method:: compute_transport_properties(n_chunks=5000, Cp=True, Lambda=True, verbose=False)


   .. py:method:: compute_unresolved_pv_fluxes(mode='DNS')


   .. py:method:: compute_unresolved_temperature_fluxes(mode='DNS')


   .. py:method:: compute_velocity_module()

      Computes the velocity module and saves it to a file.

      Description:
      ------------
      This method calculates the velocity module by squaring the values of `U_X`, `U_Y`, and `U_Z`,
      summing them up, and then taking the square root of the result. The computed velocity
      module is then saved to a file using the `save_file` function. The file path is determined
      by the `find_path` method with 'U' as the argument. After saving the file, the `update`
      method is called to refresh the attributes of the class.

      Note:
      -----
      - `self.U_X`, `self.U_Y`, and `self.U_Z` are assumed to be attributes of the class
        representing components of velocity. Make sure to check you have the relative files in your
        data folder. To check, use the method <your_field_name>.print_attributes.

      Parameters:
      -----------
      None

      Returns:
      --------
      :
      None



   .. py:method:: compute_z_grad()

      Compute and save the gradient magnitude of the mixture fraction Z.

      This function calculates the gradient of the mixture fraction Z in all three
      spatial dimensions (x, y, z) and computes its magnitude. The result is saved
      to a file and the object's state is updated.

      Prerequisites:
      - The mixture fraction Z must be computed and available as an attribute.
        Use the compute_mixture_fraction() method to calculate Z if not already done.

      Raises:
      ------
      ValueError
          If the mixture fraction Z is not available. The error message includes
          instructions on how to compute Z using the compute_mixture_fraction() method.

      Returns:
      -------
      :
      None
          The function saves the computed gradient magnitude to a file and updates
          the object's state, but does not return any value.



   .. py:method:: cut(cut_size, mode='xyz', exist_ok=False, overwrite=False)

      Cut a field into smaller sections based on a specified cut size.

      Description:
      ------------
      This method cuts a field into smaller sections based on the specified cut size and mode.
      It creates a new folder to store the cut data and grid files. If the folder already exists,
      it prompts the user for confirmation before overwriting the content. The chemical path is
      copied to the new cut folder. Each attribute of the field is cut according to the specified
      cut size and mode, and the resulting sections are saved to files. Finally, the information
      file ('info.json') is updated with the new shape of the field.

      Parameters:
      -----------
      cut_size : int
          The size of the cut.
      mode : str, optional
          The mode of cutting. Default is 'xyz'.

      Returns:
      --------
      :
      str
          Path of the cut folder.

      Note:
      -----
      Add different cutting modes to the function



   .. py:method:: downsample(ds_size=None)


   .. py:method:: filter_favre(filter_size, filter_type='Gauss', exist_ok=False, overwrite=False)

      Filter every scalar in the field object using Favre-averaging.

      Description:
      ------------
      This method filters a field using the Favre-averaged filtering technique with the specified
      filter size and type. It creates a new folder to store the filtered data and grid files. If
      the folder already exists, it prompts the user for confirmation before overwriting the content.
      The chemical path and information file ('info.json') are copied to the new filtered folder.
      Each attribute of the field is filtered according to the specified filter size and type, and
      the resulting filtered sections are saved to files.

      Parameters:
      -----------
      filter_size : int
          The size of the filter.
      filter_type : str, optional
          The type of filter to use. Default is 'gauss'.

      Raises:
      -------
      TypeError
          If filter_size is not an integer.
          If filter_type is not a string.
      ValueError
          If filter_type is not one of the valid options.

      Returns:
      --------
      :
      str
          Path of the filtered field folder.

      Example:
      --------
      >>> field = Field(folder_path='../data/field1')
      >>> filtered_folder_path = field.filter_favre(filter_size=5)
      Filtering Field '../data/field1'...
      Done Filtering Field '../data/field1'.
      Filtered Field path: '../data/Filter5Favre'



   .. py:method:: filter(filter_size, filter_type='Gauss', exist_ok=False, overwrite=False)

      Filter every scalar in the field object.

      Description:
      ------------
      This method filters a field using the specified filter size and type. It creates a new folder
      to store the filtered data and grid files. If the folder already exists, it prompts the user for
      confirmation before overwriting the content. The chemical path and information file ('info.json')
      are copied to the new filtered folder. Each attribute of the field is filtered according to the
      specified filter size and type, and the resulting filtered sections are saved to files.

      Parameters:
      -----------
      filter_size : int
          The size of the filter.
      filter_type : str, optional
          The type of filter to use. Default is 'gauss'.

      Raises:
      -------
      TypeError
          If filter_size is not an integer.
          If filter_type is not a string.
      ValueError
          If filter_type is not one of the valid options.

      Returns:
      --------
      :
      str
          Path of the filtered field folder.

      Example:
      --------
      >>> field = Field(folder_path='../data/field1')
      >>> filtered_folder_path = field.filter(filter_size=5)
      Filtering Field '../data/field1'...
      Done Filtering Field '../data/field1'.
      Filtered Field path: '../data/Filter5Gauss'



   .. py:method:: find_path(attr)

      Finds a specified attribute in the attributes list and returns the corresponding element
      in the paths list.

      Parameters:
      -----------
      attr : str
          The specific element to find in the first list.

      Returns:
      --------
      :
      str
          The corresponding element in the second list if the specific element is found in the first list.

      Raises:
      -------
      TypeError
          If 'attr' is not a string.
      ValueError
          If the specific element is not found in the attributes list.




   .. py:method:: plot_x_midplane(attribute, log=False, colormap='viridis', cbar_title=None, cbar_shrink=0.7, levels=None, color='black', labels=False, linestyle='-', linecolor='black', linewidth=1, x_ticks=None, y_ticks=None, x_lim=None, y_lim=None, vmin=None, vmax=None, transparent=True, title=None, x_name='y [mm]', y_name='z [mm]', remove_cbar=False, remove_x=False, remove_y=False, transpose=False, save=False, show=True)

      Plots the x midplane of a specified attribute in the Field class.

      Description:
      ------------
      This method plots the x midplane of a specified attribute in the Field class. It verifies
      if the attribute is valid, and then uses the built in function contour_plot to generate
      the plot.

      Parameters:
      -----------
      attribute : str
          The name of the attribute to plot.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_y_midplane(attribute, log=False, colormap='viridis', cbar_title=None, cbar_shrink=0.7, levels=None, color='black', labels=False, linestyle='-', linecolor='black', linewidth=1, x_ticks=None, y_ticks=None, x_lim=None, y_lim=None, vmin=None, vmax=None, transparent=True, title=None, x_name='x [mm]', y_name='z [mm]', remove_cbar=False, remove_x=False, remove_y=False, transpose=False, save=False, show=True)

      Plots the y midplane of a specified attribute in the Field class.

      Description:
      ------------
      This method plots the z midplane of a specified attribute in the Field class. It verifies
      if the attribute is valid, and then uses the built in function contour_plot to generate
      the plot.

      Parameters:
      -----------
      attribute : str
          The name of the attribute to plot.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_z_midplane(attribute, log=False, colormap='viridis', cbar_title=None, cbar_shrink=0.7, levels=None, color='black', labels=False, linestyle='-', linecolor='black', linewidth=1, x_ticks=None, y_ticks=None, x_lim=None, y_lim=None, vmin=None, vmax=None, transparent=True, title=None, x_name=None, y_name=None, remove_cbar=False, remove_x=False, remove_y=False, remove_title=False, transpose=False, dpi=None, scale='mm', save=False, save_path=None, show=True)

      Plots the z midplane of a specified attribute in the Field class.

      Description:
      ------------
      This method plots the z midplane of a specified attribute in the Field class. It verifies
      if the attribute is valid, and then uses the built in function contour_plot to generate
      the plot.

      Parameters:
      -----------
      attribute : str
          The name of the attribute to plot.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_midplane(attribute, plane='z', log=False, colormap='viridis', cbar_title=None, cbar_shrink=0.7, levels=None, color='black', labels=False, linestyle='-', linecolor='black', linewidth=1, x_ticks=None, y_ticks=None, x_lim=None, y_lim=None, vmin=None, vmax=None, transparent=True, title=None, x_name=None, y_name=None, remove_cbar=False, remove_x=False, remove_y=False, remove_title=False, transpose=False, dpi=None, scale='mm', save=False, save_path=None, show=True)

      Plots the specified midplane of a scalar attribute of the Field class.

      Description:
      ------------
      This method plots a midplane of a specified attribute in the Field class. It verifies
      if the attribute is valid, and then uses the built in function contour_plot to generate
      the plot.

      Parameters:
      -----------
      attribute : str
          The name of the attribute to plot.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_api(api_type='TSR', n=4, plane='z_midplane', vmin=None, vmax=None, cmap='viridis', n_cols=None, figsize=None, include_reactions=True, save_path=None, dpi=400, **kwargs)

      Plot the top N reactions by API (Analytical Projected Integral) norm.

      This method identifies the reactions with the highest L1 norm of their API
      values, extracts the specified 2D plane for each, and plots them as subplots
      with a shared colorbar.

      Parameters:
      -----------
      api_type : str, optional
          Type of API to plot (e.g., "TSR"). Default is "TSR"
      n : int, optional
          Number of top reactions to plot. Default is 4
      plane : str, optional
          Plane attribute to extract from API fields. Default is 'z_midplane'
          Other options depend on your Field implementation (e.g., 'x_midplane', 'y_midplane')
      vmin : float, optional
          Minimum value for colorbar. If None, uses global minimum of fields
      vmax : float, optional
          Maximum value for colorbar. If None, uses global maximum of fields
      cmap : str, optional
          Colormap name. Default is 'viridis'
      n_cols : int, optional
          Number of columns in the subplot grid. If None, automatically determined
      figsize : tuple, optional
          Figure size (width, height). If None, automatically computed
      include_reactions : bool, optional
          If True, includes reaction equations as side text. Default is True
      save_path : str, optional
          If provided, saves the figure to this path. Default is None (no save)
      dpi : int, optional
          DPI for saved figure. Default is 400
      **kwargs : dict
          Additional keyword arguments passed to plot_multifield()

      Returns:
      --------
      :
      fig : matplotlib.figure.Figure
          The figure object
      axes : numpy.ndarray
          Array of subplot axes
      top_reactions : list of tuple
          List of (name, norm, reaction_string) for the top N reactions

      Example:
      --------
      >>> field = ap.Field('DNS_DATA_2d_cut_cut')
      >>> fig, axes, top_rxns = field.plot_api(api_type="TSR", n=4,
      ...                                       vmin=-1, vmax=1,
      ...                                       save_path="APIs_TSR.png")



   .. py:method:: print_attributes()

      Prints the valid attributes of the class and their corresponding file paths.

      Description
      -----------

      This method calls the `update` method with `print_valid_attributes` set to `True`.
      As a result, it prints out the valid attributes (those that have corresponding files
      in the data path) of the class and their corresponding file paths. This is useful
      when you want to see which attributes are currently valid in the class instance.



   .. py:method:: scatter_Z(attribute, logx=False, logy=False, c=None, s=1, alpha=1, max_dim=2000000, colormap='viridis', cbar_title=None, vmin=None, vmax=None, marker='.', avg=False, avg_label=None, num_bins=100, linestyle='--', linewidth=2, linecolor='b', Z_st=True, x_name='Z', y_name=None, title=None, x_ticks=None, y_ticks=None, remove_markers=True, remove_x=False, remove_y=False, save=False, show=True)


   .. py:method:: update(verbose=False, print_valid_attributes=False)

      Update the attributes of the class based on the existence of files in the specified data path.

      This method checks the existence of files corresponding to the attribute paths in the data path.
      If a file exists for an attribute and it was not present before, it initializes a new attribute
      in the class using Scalar with the file path. If verbose is True, it prints the new attributes
      initialized and the existing attributes with their paths.

      Parameters:
      -----------

      - verbose (bool, optional): If True, prints information about the initialization of new attributes.
                                 Default is False.


