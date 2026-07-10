import pytensor.tensor as pt
import exoplanet as xo
import pymc as pm
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import corner

    # 1.Load HOPS light Curve
print("1.Load HOPS light Curve")
# data = np.loadtxt("/Users/danielbhuglah/@EP_HOPS_TOI-1266cP39R3/EP_HOPS_TOI-1266cP39R3.txt")
data = np.loadtxt("/Users/danielbhuglah/@HOPSforEPP_TOI-1266c_P41R1_24Feb/TOI_1266cP41R1_forepp.txt")
t = data[:, 0]
flux = data[:, 1]
flux_err = data[:, 2]

plt.title("Step 1:HOPS data")
plt.errorbar(t, flux, yerr=flux_err, fmt=".k")
plt.xlabel("Time")
plt.ylabel("Relative Flux")
plt.show()

    # 2.Build a transit model with exoplanet
print("2.Build a transit model with exoplanet")
with pm.Model() as model:

    # Transit midpoint
    t0 = pm.Normal("t0", mu=np.median(t), sigma=0.1)

    print("median of t =", np.median(t))

    # Orbital period
    period = pm.Normal("period", mu=18.801682, sigma=0.1)

    # Planet/star radius ratio
    log_r = pm.Normal("log_r", mu=np.log(0.0346), sigma=2)
    r = pm.Deterministic("r", pt.exp(log_r))

    # Impact parameter
    b = pm.Uniform("b", lower=0, upper=1+r)

    # Limb darkening
    u = xo.distributions.quad_limb_dark(
        "u",
        initval=np.array([0.28279, 0.3201])
    )

    # Orbit
    orbit = xo.orbits.KeplerianOrbit(
        period=period,
        t0=t0,
        b=b,
    )

    # Transit model
    light_curve = (
        xo.LimbDarkLightCurve(u[0], u[1])
        .get_light_curve(
            orbit=orbit,
            r=r,
            t=t,
        )[:, 0]
        + np.mean(flux)
    )

    model_flux = pm.Deterministic(
        "model_flux",
        light_curve
    )
      
    # Likelihood
    pm.Normal(
        "obs",
        mu=model_flux,
        sigma=flux_err,
        observed=flux,
    )

    trace = pm.sample(
        tune=1000,
        draws=1000,
        chains=2,
        cores=1,
    )

# 3.Plot the fitted model
print("3.Plot the fitted model")
posterior_flux = trace.posterior["model_flux"].values

q16, q50, q84 = np.percentile(
    posterior_flux,
    [16, 50, 84],
    axis=(0, 1),
)

plt.errorbar(
    t,
    flux,
    yerr=flux_err,
    fmt=".k",
    label="Data",
)

plt.plot(t, q50, label="Median model")

plt.fill_between(
    t,
    q16,
    q84,
    alpha=0.3,
    label="1σ posterior",
)

plt.title("Step3: Plot of fitted model")
plt.xlabel("Time")
plt.ylabel("Relative Flux")
plt.legend()

plt.show()

# 4.Assess goodness to fit
print("4.Assess goodness to fit")
# 4a.Inspect residuals
print("4a.Inspect residuals")
residuals = flux - q50

plt.errorbar(
    t,
    residuals,
    yerr=flux_err,
    fmt=".k",
)

plt.title("Step 4a: plot of residuals")
plt.axhline(0, color="r")

plt.xlabel("Time")
plt.ylabel("Residuals")

plt.show()

# 4b.Check posterior summaries
print("4b.Check posterior summaries")


summary = az.summary(
    trace,
    var_names=["t0", "period", "r", "b"],
)

for idx, row in summary.iterrows():

    print(f"\nParameter: {idx}")

    for col in summary.columns:

        value = row[col]

        try:
            print(f"  {col:12s}: {float(value):.8f}")
        except:
            print(f"  {col:12s}: {value}")

print(summary)

# 4c. Posterior predictive checks
print("4c. Posterior predictive checks")
with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        var_names=["obs"],
    )

