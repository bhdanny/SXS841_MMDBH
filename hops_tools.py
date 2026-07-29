# -----------------------------
# Import modules
# -----------------------------
from pathlib import Path
import os
import yaml
import re
import json
import pandas as pd
from datetime import datetime
from astropy.time import Time
import numpy as np

# --------------------------------------------------
# Extract detrending method from PHOTOMETRY_APERTURE_XX/log.yaml.
#    Returns a string like 'Airmass' or 'Airmass + Background'.
# --------------------------------------------------
def extract_detrending_method_from_log(log_path: Path):
  
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    detr = data.get("detrending")

    # Case 1: detrending is missing
    if detr is None:
        return None

    # Case 2: detrending: Airmass  (string)
    if isinstance(detr, str):
        return detr

    # Case 3: detrending: { method: Airmass }
    if isinstance(detr, dict):
        method = detr.get("method")
        if isinstance(method, list):
            return " + ".join(str(m) for m in method)
        return str(method) if method is not None else None

    # Case 4: detrending: [Airmass, Background]
    if isinstance(detr, list):
        return " + ".join(str(m) for m in detr)

    # Unknown format
    return str(detr)

def extract_initial_params_from_aperture_log(log_path: Path):
    """
    Extract initial model parameters from PHOTOMETRY_APERTURE_XX/log.yaml.
    Returns a dict with period, rp_over_rs, sma_over_rs, inclination, mid_time.
    Missing values return None.
    """
    params = {
        "init_iterations": None,
        "init_burn": None,
        "init_period": None,
        "init_rp_over_rs": None,
        "init_sma_over_rs": None,
        "init_inclination": None,
        "init_mid_time": None,
        "init_transit_depth": None,
        "init_transit_depth_ppt": None,
    }

    if not log_path.exists():
        log_warn(f"Aperture log.yaml missing: {log_path}")
        return params

    try:
        with open(log_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        log_error(f"Failed to read aperture log.yaml: {e}")
        return params

    # Extract parameters if present
    params["init_iterations"]   = data.get("iterations")
    params["init_burn"]         = data.get("burn")
    params["init_period"]       = data.get("period")
    params["init_rp_over_rs"]   = data.get("rp_over_rs")
    params["init_sma_over_rs"]  = data.get("sma_over_rs")
    params["init_inclination"]  = data.get("inclination")
    params["init_mid_time"]     = data.get("mid_time")

    # Compute transit depth if rp_over_rs is present
    rp = params["init_rp_over_rs"]
    if rp is not None:
        depth = rp * rp
        params["init_transit_depth"] = depth
        params["init_transit_depth_ppt"] = depth * 1e3  # convert to ppt

    return params
# --------------------------------------------------
# Get the fitting results from the results.txt file
# and put  into a single row.
# --------------------------------------------------
    # from pathlib import Path
    # from astropy.time import Time
    
    # from pathlib import Path
    # from astropy.time import Time

def parse_results_txt(path: Path):
    """
    Parse HOPS results.txt using whitespace splitting.
    Extract ALL variables with fit/fix, value, uncertainties,
    AND initial/min/max allowed values.
    """

    print("\nMMDBH - start of parse_results_txt function")
    # SAFE FILE READ
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"❌ Could not read {path}: {e}")
        return {"read_error": str(e)}

    results = {}
    in_table = False
    in_detrended = False

    for line in lines:
        stripped = line.strip()

        # Start of table
        if stripped.startswith("# variable"):
            in_table = True
            continue

        # End of table
        if in_table and stripped == "":
            in_table = False
            continue

        # -----------------------------
        # Parse table rows
        # -----------------------------
        if in_table and not stripped.startswith("#"):
            parts = stripped.split()

            # Expected format:
            # var fitfix value unc_low unc_high initial min max
            if len(parts) >= 8:
                var = parts[0]
                fitfix = parts[1]
                value = parts[2]
                unc_low = parts[3]
                unc_high = parts[4]
                initial = parts[5]
                min_allowed = parts[6]
                max_allowed = parts[7]

                def clean(x):
                    return None if x in ("--", "") else float(x)

                # Store all fields
                results[f"{var}_fitfix"] = fitfix
                results[f"{var}_value"] = clean(value)
                results[f"{var}_unc_low"] = clean(unc_low)
                results[f"{var}_unc_high"] = clean(unc_high)
                results[f"{var}_initial"] = clean(initial)
                results[f"{var}_min_allowed"] = clean(min_allowed)
                results[f"{var}_max_allowed"] = clean(max_allowed)

            continue

        # -----------------------------
        # Detect detrended block
        # -----------------------------
        if stripped.startswith("#Detrended Residuals"):
            in_detrended = True
            continue

        # -----------------------------
        # Parse metadata lines
        # -----------------------------
        if stripped.startswith("#") and ":" in stripped:
            key, val = stripped[1:].split(":", 1)
            key = key.strip().replace(" ", "_")
            val = val.strip()

            # Convert numeric values
            try:
                if "." in val or "e" in val.lower():
                    val = float(val)
                else:
                    val = int(val)
            except:
                pass

            if in_detrended:
                results[f"Detrended_{key}"] = val
            else:
                results[key] = val

    # ---------------------------------------------------------
    # Compute expected model mid_time = average(min, max)
    # ---------------------------------------------------------
    if (
        "mid_time_min_allowed" in results
        and "mid_time_max_allowed" in results
        and results["mid_time_min_allowed"] is not None
        and results["mid_time_max_allowed"] is not None
    ):
        results["mid_time_expected_model"] = (
            results["mid_time_min_allowed"] + results["mid_time_max_allowed"]
        ) / 2.0
    #---------------------------------------------------------
    # Compute O-C (Observed minus Calculated)
    # ---------------------------------------------------------
        
    if (
         "mid_time_value" in results
         and "mid_time_expected_model" in results
         and results["mid_time_value"] is not None
         and results["mid_time_expected_model"] is not None
    ):
         oc_days = results["mid_time_value"] - results["mid_time_expected_model"]
         results["mid_time_O_C_days"] = oc_days
         # Convert O-C to hh:mm:ss
         oc_seconds = oc_days * 86400.0
         sign = "-" if oc_seconds < 0 else "+"
         oc_seconds_abs = abs(oc_seconds)

         hours = int(oc_seconds_abs // 3600)
         minutes = int((oc_seconds_abs % 3600) // 60)
         seconds = oc_seconds_abs % 60

         results["mid_time_O_C_hms"] = f"{sign}{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    # ---------------------------------------------------------
    # Convert mid_time uncertainties (days) → hh:mm:ss format
    # ---------------------------------------------------------
    def format_hms_from_days(days_value):
        """Convert days → signed hh:mm:ss.sss string."""
        if days_value is None:
            return None

        total_seconds = days_value * 86400.0
        sign = "-" if total_seconds < 0 else "+"
        total_seconds = abs(total_seconds)

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    # Lower uncertainty
    if "mid_time_unc_low" in results and results["mid_time_unc_low"] is not None:
        results["mid_time_unc_low_hms"] = format_hms_from_days(results["mid_time_unc_low"])

    # Upper uncertainty
    if "mid_time_unc_high" in results and results["mid_time_unc_high"] is not None:
        results["mid_time_unc_high_hms"] = format_hms_from_days(results["mid_time_unc_high"])

    # ---------------------------------------------------------
    # Convert BJD_TDB → UTC for mid_time
    # ---------------------------------------------------------
    if "mid_time_value" in results and results["mid_time_value"] is not None:
        bjd = results["mid_time_value"]
        if bjd > 1e6:
            print("MMDBH - BJD conversion routine - ", bjd)
            try:
                t = Time(bjd, format="jd", scale="tdb")
                results["mid_time_bjd"] = bjd
                results["mid_time_utc"] = t.utc.iso

                low = results.get("mid_time_unc_low")
                high = results.get("mid_time_unc_high")

                if low is not None:
                    low_bjd = Time(bjd+low, format="jd", scale="tdb")
                    results["mid_time_utc_unc_low"] = low_bjd.utc.iso

                if high is not None:
                    high_bjd = Time(bjd+high, format="jd", scale="tdb")
                    results["mid_time_utc_unc_high"] = high_bjd.utc.iso

               

           
            except Exception as e:
                results["mid_time_utc"] = f"conversion_failed: {e}"
                
    # ---------------------------------------------------------
    # Transit depth and SNR estimation
    # ---------------------------------------------------------
    print("\nMMDBH - start Transit depth and SNR estimation")
    rp = results.get("rp_over_rs_value")
    rp_low = results.get("rp_over_rs_unc_low")
    rp_high = results.get("rp_over_rs_unc_high")

    if rp is not None:
        print("\nMMDBH - Transit depth calculation")
        # Depth
        depth = rp * rp
        results["transit_depth"] = depth
        results["transit_depth_ppt"] = depth * 1e3

        # Depth uncertainties
        if rp_low is not None:
            depth_low = (rp - abs(rp_low))**2 - depth
            results["transit_depth_unc_low"] = depth_low
            results["transit_depth_unc_low_ppt"] = depth_low * 1e3

        if rp_high is not None:
            depth_high = (rp + abs(rp_high))**2 - depth
            results["transit_depth_unc_high"] = depth_high
            results["transit_depth_unc_high_ppt"] = depth_high * 1e3

    # ---------------------------------------------------------
    # SNR estimate using detrended RMS
    # ---------------------------------------------------------

    rms = results.get("Detrended_RMS")

    
    # ---------------------------------------------------------
    # SAFE LOAD of detrended_model.txt
    # ---------------------------------------------------------
    detrended_path = path.parent / "detrended_model.txt"

    try:
        lc_detrended = np.loadtxt(detrended_path)
        # npts = lc_detrended.shape[0]
        model = lc_detrended[:, 4]
        residuals = lc_detrended[:, 5]
        # Count points that are actually inside the transit
        in_transit = model < 1.0
        n_in_transit = np.count_nonzero(in_transit)

         # Count points that are actually outside the transit
        out_of_transit = model >= 1.0
        oot_rms = np.sqrt(np.mean(residuals[out_of_transit]**2))

        results["Detrended_Npoints"] = lc_detrended.shape[0]
        results["Transit_Npoints"] = n_in_transit
        results["oot_rms"] = oot_rms
    
        
        #results["Detrended_Npoints"] = npts

        if rms is not None and rp is not None:
            # snr = (depth / rms) * (npts ** 0.5)
            snr = (depth/oot_rms)*np.sqrt(n_in_transit)
            results["transit_snr"] = snr

    except Exception as e:
        print(f"❌ Could not read {detrended_path}: {e}")
        results["Detrended_Npoints"] = None
        results["transit_snr"] = None
    

    return results


# --------------------------------------------------
# Look for the folders that contain the results of the Aperture fitting
# --------------------------------------------------
def find_fitting_folders(root_dir):
    """
    Scan a directory tree and return a sorted list of valid fitting folders.
    A valid folder is one that contains BOTH:
        - results.txt
        - config.yaml   (or config.yml)
    This mirrors the structure produced by HOPS-like pipelines.

    Parameters
    ----------
    root_dir : str or Path
        The directory to scan.

    Returns
    -------
    list of Path
        Sorted list of folders that contain valid fitting outputs.
    """
    from pathlib import Path

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    fitting_folders = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        results_file = folder / "results.txt"
        config_yaml  = folder / "log.yaml"
        config_yml   = folder / "log.yml"

        # Folder is valid if results.txt exists AND one config file exists
        if results_file.exists() and (config_yaml.exists() or config_yml.exists()):
            fitting_folders.append(folder)

    # Sort by folder name (natural sort)
    fitting_folders = sorted(fitting_folders, key=lambda p: p.name.lower())

    return fitting_folders
# --------------------------------------------------
# Get the info from the results.txt file which contains the fitting results
# --------------------------------------------------
def collect_fitting_results(root_folder: Path):
    rows = []

    print("Scanning root:", root_folder)

    for phot_folder in root_folder.glob("**/PHOTOMETRY_*"):
        if not phot_folder.is_dir():
            continue  # skip files like PHOTOMETRY_g.txt
        print("\nFound photometry folder:", phot_folder)
        
        fitting_folders = find_fitting_folders(phot_folder)
        print("  Fitting folders:", fitting_folders)

        for fit_folder in fitting_folders:
            results_file = fit_folder / "results.txt"
            print("    Checking:", results_file)

            if not results_file.exists():
                print("      ❌ results.txt NOT FOUND")
                continue

            print("      ✔ results.txt FOUND")

            fit_results = parse_results_txt(results_file)
            print("      Extracted keys:", list(fit_results.keys())[:10], "...")

            # Extract detrending method from the fitting folder's log.yaml
            log_yaml = fit_folder / "log.yaml"
            detrending_method = extract_detrending_method_from_log(log_yaml)
            # Extract initial model parameters from aperture log.yaml
            aperture_log = fit_folder / "log.yaml"
            initial_params = extract_initial_params_from_aperture_log(aperture_log)
            
            row = {
                "root_path": str(root_folder),
                "photometry_folder": phot_folder.name,
                "fitting_folder": fit_folder.name,
                "detrending_method": detrending_method,
            }

            row.update(initial_params)   # ← NEW
            row.update(fit_results)
            
            # ---------------------------------------------------------
            # Compare expected vs fitted transit depth (N-sigma)
            # ---------------------------------------------------------
            init_depth = row.get("init_transit_depth")
            fit_depth = row.get("transit_depth")
            unc_low = row.get("transit_depth_unc_low")
            unc_high = row.get("transit_depth_unc_high")

            if init_depth is not None and fit_depth is not None and unc_low is not None and unc_high is not None:
                sigma = (abs(unc_low) + abs(unc_high)) / 2
                if sigma > 0:
                    row["transit_depth_Nsigma"] = (fit_depth - init_depth) / sigma
                else:
                    row["transit_depth_Nsigma"] = None
            else:
                row["transit_depth_Nsigma"] = None

            
            rows.append(row)
            
    print("`nMMDBH - Check a print works")
    print("\nTotal rows collected:", len(rows))
    return rows
# --------------------------------------------------
# Open the log.yaml file in the PHOTOMETRY Folder, 
# --------------------------------------------------
def load_yaml(path):
    """Safe YAML loader."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except:
        return {}
# --------------------------------------------------
# Get the Target and comparison star x,y coordindates from the log.yaml file
# in PHOTOMETRY folder and if active or deactive
# --------------------------------------------------
def extract_star_info_from_log(log_path, run_name):
    """Extracts target + comparison star info from a HOPS log.yaml file."""
    data = load_yaml(log_path)
    rows = []

    # -------------------------
    # Extract TARGET star
    # -------------------------
    target = {
        "run": run_name,
        "star_type": "target",
        "star_id": "target",
        "x": data.get("target_x_position"),
        "y": data.get("target_y_position"),
        "aperture": data.get("target_aperture"),
        "r_position": data.get("target_r_position"),
        "u_position": data.get("target_u_position"),
        "active": True  # target is always active
    }
    rows.append(target)

 # -------------------------
    # Extract COMPARISON stars
    # -------------------------
    comp_pattern = re.compile(r"comparison_(\d+)_(.+)")

    comp_dict = {}

    for key, value in data.items():
        match = comp_pattern.match(key)
        if match:
            comp_id = int(match.group(1))
            field = match.group(2)

            if comp_id not in comp_dict:
                comp_dict[comp_id] = {"run": run_name,
                                      "star_type": "comparison",
                                      "star_id": comp_id}

            # Normalize field names
            if field == "x_position":
                comp_dict[comp_id]["x"] = value
            elif field == "y_position":
                comp_dict[comp_id]["y"] = value
            elif field == "aperture":
                comp_dict[comp_id]["aperture"] = value
            elif field == "active":
                comp_dict[comp_id]["active"] = value
            elif field == "r_position":
                comp_dict[comp_id]["r_position"] = value
            elif field == "u_position":
                comp_dict[comp_id]["u_position"] = value
            else:
                # Store any extra fields without losing information
                comp_dict[comp_id][field] = value

    # Add comparison stars to rows
    for comp_id, comp_data in comp_dict.items():
        # Default inactive if not specified
        comp_data.setdefault("active", False)
        rows.append(comp_data)

    return rows
# --------------------------------------------------
# GO through the root directory and lok for all the log.yaml files
# in PHOTOMETRY folders.
# --------------------------------------------------
def extract_all_logs(root_folder):
    """Walk through all PHOTOMETRY_n folders and extract log.yaml info."""
    all_rows = []

    for item in os.listdir(root_folder):
        if item.startswith("PHOTOMETRY_"):
            run_path = os.path.join(root_folder, item)
            log_path = os.path.join(run_path, "log.yaml")

            if os.path.exists(log_path):
                rows = extract_star_info_from_log(log_path, item)
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)