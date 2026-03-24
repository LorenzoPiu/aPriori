# tests/test_h2_premixed_pipeline.py
import os
import numpy as np
import shutil
import pytest

import aPriori as ap

@pytest.mark.slow
@pytest.mark.integration
def test_h2_premixed_end_to_end(apriori_test_cache_dir: str):
    """
    Please run the test with the following command:
      pytest --html=pytest_report.html --self-contained-html

    End-to-end smoke/integration test:
      - download dataset if needed
      - load Field
      - compute reaction rates
      - Favre filter
      - compute a subset of derived quantities

    Assertions focus on:
      - attributes exist
      - shapes are consistent
      - values are finite
      - a couple of basic physical / numerical sanity checks
    """
    # Setup the folders names
    case_name = "h2_lifted_dns"             # case name
    test_name = "test_pipeline_h2"          # test name

    case_name_orig = case_name + '_orig'    # original dataset name. This one will not be touched by the operations
    dns_data_folder_orig = os.path.join(apriori_test_cache_dir, case_name_orig)

    tests_data_folder = os.path.join(apriori_test_cache_dir, "tmp_tests_results") # The folder where
    dns_data_folder   = os.path.join(tests_data_folder, test_name, case_name)

    os.makedirs(dns_data_folder_orig, exist_ok=True)

    # Download only if needed
    if not os.path.exists(os.path.join(dns_data_folder_orig, "data")):
        ap.download(dataset="h2_premixed", dest_folder=dns_data_folder_orig)

    # Delete the current test folder if it already exists (as it could contain previously processed data), and copy the original one to ensure a clean start
    if os.path.exists(os.path.join(tests_data_folder, test_name)):
        shutil.rmtree(os.path.join(tests_data_folder, test_name))
    else:
        os.makedirs(os.path.join(tests_data_folder, test_name), exist_ok=True)
    shutil.copytree(dns_data_folder_orig, dns_data_folder)

    dns_field = ap.Field(dns_data_folder)

    # --- DNS reaction rates (should create RH2_DNS etc.)
    dns_field.compute_reaction_rates(exist_ok=True)

    # Check a couple of key variables exist and are readable
    assert hasattr(dns_field, "YH2"), "Dataset should expose YH2"
    assert hasattr(dns_field, "RH2_DNS"), "Reaction-rate computation should expose RH2_DNS"

    yh2_dns = dns_field.YH2._3d
    rh2_dns = dns_field.RH2_DNS._3d

    assert list(yh2_dns.shape) == dns_field.shape
    assert list(rh2_dns.shape) == dns_field.shape
    assert np.isfinite(yh2_dns).all()
    assert np.isfinite(rh2_dns).all()

    # Basic sanity: mass fraction should be within [0, 1] up to tiny tolerance
    assert yh2_dns.min() >= -1e-8
    assert yh2_dns.max() <= 1.0 + 1e-8

    # --- Favre filter
    filter_size = 10
    filtered_path = dns_field.filter_favre(filter_size, exist_ok=True)
    filtered_field = ap.Field(filtered_path)

    # Filtered reaction rates
    filtered_field.compute_reaction_rates(exist_ok=True)

    # Check filtered variables exist
    assert hasattr(filtered_field, "YH2")
    assert hasattr(filtered_field, "RH2_DNS")
    assert hasattr(filtered_field, "RH2_LFR"), "LFR rates should be computed by compute_reaction_rates"

    yh2_f = filtered_field.YH2._3d
    rh2_dns_f = filtered_field.RH2_DNS._3d
    rh2_lfr_f = filtered_field.RH2_LFR._3d

    assert list(yh2_f.shape) == filtered_field.shape
    assert np.isfinite(yh2_f).all()
    assert np.isfinite(rh2_dns_f).all()
    assert np.isfinite(rh2_lfr_f).all()

    # Filtering should not explode the range
    assert yh2_f.min() >= -1e-6
    assert yh2_f.max() <= 1.0 + 1e-6

    # --- Derived quantities on filtered field (subset of your script)
    filtered_field.compute_strain_rate(save_tensor=True)
    assert hasattr(filtered_field, "S_LES") or hasattr(filtered_field, "S_DNS"), "strain-rate outputs should exist"

    filtered_field.compute_residual_dissipation_rate(mode="Smag")
    assert hasattr(filtered_field, "Epsilon_r_Smag"), "Smag model output should exist"
    eps = filtered_field.Epsilon_r_Smag._3d
    assert list(eps.shape) == filtered_field.shape
    assert np.isfinite(eps).all()

    filtered_field.compute_residual_kinetic_energy()
    assert hasattr(filtered_field, "K_r_Yosh"), "Residual kinetic energy output attribute 'K_r' should exist "

    # Timescales
    filtered_field.compute_chemical_timescale(mode="SFR", replace_nonreacting="max")
    assert hasattr(filtered_field, "Tau_c_SFR"), "Chemical timescale should exist"
    tau_c = filtered_field.Tau_c_SFR.value
    assert np.isfinite(tau_c).all()

    filtered_field.fuel = "H2"
    filtered_field.ox = "O2"
    filtered_field.compute_chemical_timescale(mode="Ch")
    assert hasattr(filtered_field, "Tau_c_Ch"), "Chomiak timescale should exist"

    filtered_field.compute_mixing_timescale(mode="Kolmo")
    assert hasattr(filtered_field, "Tau_m_Kolmo"), "Mixing timescale should exist"

    # Progress variable + gradient + laplacian(DNS field)
    dns_field.compute_progress_variable("H2O")
    assert hasattr(dns_field, "C"), "Progress variable should exist"

    dns_field.compute_gradient_C()
    assert hasattr(dns_field, "C_grad"), "Progress-variable gradient should exist"

    dns_field.compute_laplacian_C()
    assert hasattr(dns_field, "C_laplacian"), "Progress-variable laplacian should exist"


    # Progress variable + gradient + laplacian (filtered field)
    filtered_field.compute_progress_variable("H2O")
    assert hasattr(filtered_field, "C"), "Progress variable should exist"

    filtered_field.compute_gradient_C()
    assert hasattr(filtered_field, "C_grad"), "Progress-variable gradient should exist"

    filtered_field.compute_laplacian_C()
    assert hasattr(filtered_field, "C_laplacian"), "Progress-variable laplacian should exist"

    # --- Optional: midplane cut smoke test (fast check it doesn't crash)
    cut_field = ap.Field(filtered_field.cut([0, 0, 100], exist_ok=True))
    assert cut_field.shape[2] == 1 or cut_field.shape[2] < filtered_field.shape[2], "Cut should reduce domain"

    # --- Optional: plot figures for qualitative check
    def _save_and_check_fig(fig, outpath: str, dpi: int = 150):
        """
        Save a Matplotlib figure and assert it exists and is non-empty.
        """
        # Ensure folder exists
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        fig.savefig(outpath, bbox_inches="tight", dpi=dpi)
        # Close to avoid memory build-up during test runs
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        assert os.path.exists(outpath), f"Expected figure file not found: {outpath}"
        # 10 KB threshold: avoids passing with a zero-byte file
        assert os.path.getsize(outpath) > 10_000, f"Figure seems too small/corrupt: {outpath}"

    figures_dir = os.path.join(tests_data_folder, test_name, "figures")
    dpi = 200

    # --- Plot: YH2 DNS midplane
    fig1, ax1 = dns_field.plot_z_midplane(
        "YH2",
        transpose=True,
        colormap="hot_r",
        vmin=0.0,
        vmax=0.015,
        title=r"$Y_{H_2}$",
        remove_cbar=False,
    )
    _save_and_check_fig(fig1, os.path.join(figures_dir, "YH2_dns.png"), dpi=dpi)

    # --- Plot: YH2 filtered midplane
    fig2, ax2 = filtered_field.plot_z_midplane(
        "YH2",
        transpose=True,
        colormap="hot_r",
        vmin=0.0,
        vmax=0.015,
        title=r"$\widetilde{Y}_{H_2}$",
        cbar_title=r"$Y_{H_2}$ [-]",
        remove_y=False,
    )
    _save_and_check_fig(fig2, os.path.join(figures_dir, "YH2_filtered.png"), dpi=dpi)
    
    # --- Plot: Progress variable
    fig3, ax3 = dns_field.plot_z_midplane(
        "C",
        transpose=True,
        colormap="hot_r",
        # title=r"$c$",
        cbar_title=r"$c$ [-]",
        remove_y=True,
    )
    _save_and_check_fig(fig3, os.path.join(figures_dir, "C.png"), dpi=dpi)

    # --- Plot: Filtered progress variable 
    fig4, ax4 = filtered_field.plot_z_midplane(
        "C",
        transpose=True,
        colormap="hot_r",
        # title="$\widetilde{c}$",
        cbar_title=r"$\widetilde{c}$",
    )
    _save_and_check_fig(fig4, os.path.join(figures_dir, "C_filtered.png"), dpi=dpi)

    # --- Plot: Progress variable gradient
    fig5, ax5 = dns_field.plot_z_midplane(
        "C_grad",
        transpose=True,
        colormap="hot_r",
        # title=r"$c$",
        cbar_title=r"$c$ [-]",
        remove_y=True,
    )
    _save_and_check_fig(fig5, os.path.join(figures_dir, "C_grad.png"), dpi=dpi)

    # --- Plot: Filtered progress variable 
    fig6, ax6 = filtered_field.plot_z_midplane(
        "C_grad",
        transpose=True,
        colormap="hot_r",
        # title="$\widetilde{c}$",
        cbar_title=r"$\widetilde{c}$",
    )
    _save_and_check_fig(fig6, os.path.join(figures_dir, "C_grad_filtered.png"), dpi=dpi)
    
    # --- Plot: Progress variable
    fig7, ax7 = dns_field.plot_z_midplane(
        "C_laplacian",
        transpose=True,
        colormap="hot_r",
        # title=r"$c$",
        cbar_title=r"$c$ [-]",
        remove_y=True,
    )
    _save_and_check_fig(fig7, os.path.join(figures_dir, "C_laplacian.png"), dpi=dpi)

    # --- Plot: Filtered progress variable 
    fig8, ax8 = filtered_field.plot_z_midplane(
        "C_laplacian",
        transpose=True,
        colormap="hot_r",
        # title="$\widetilde{c}$",
        cbar_title=r"$\widetilde{c}$",
    )
    _save_and_check_fig(fig8, os.path.join(figures_dir, "C_laplacian_filtered.png"), dpi=dpi)
    
