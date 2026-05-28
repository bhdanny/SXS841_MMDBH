import pymc as pm

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import arviz as az
import corner


np.random.seed(42)

true_m = 0.5
true_b = -1.3
true_logs = np.log(0.3)

x = np.sort(np.random.uniform(0, 5, 50))
y = true_b + true_m * x + np.exp(true_logs) * np.random.randn(len(x))

plt.plot(x, y, ".k")
plt.ylim(-2, 2)
plt.xlabel("x-example 1")
_ = plt.ylabel("y-example 1")
print("example 1 complete")

with pm.Model() as model:
    # Define the priors on each parameter:
    m = pm.Uniform("m", lower=-5, upper=5)
    b = pm.Uniform("b", lower=-5, upper=5)
    logs = pm.Uniform("logs", lower=-5, upper=5)

    # Define the likelihood. A few comments:
    #  1. For mathematical operations like "exp", you can't use
    #     numpy. Instead, use the mathematical operations defined
    #     in "pm.math".
    #  2. To condition on data, you use the "observed" keyword
    #     argument to any distribution. In this case, we want to
    #     use the "Normal" distribution (look up the docs for
    #     this).
    pm.Normal("obs", mu=m * x + b,
              sigma=pm.math.exp(logs),
              observed=y)

    # This is how you will sample the model. Take a look at the
    # docs to see that other parameters that are available.
    #trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2)
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1
    )

    _=az.plot_trace(trace, var_names=["m", "b", "logs"])

    samples = np.column_stack([
        trace.posterior["m"].values.flatten(),
        trace.posterior["b"].values.flatten(),
        trace.posterior["logs"].values.flatten(),
    ])

    corner.corner(
        samples,
        labels=["m", "b", "logs"],
        truths=[true_m, true_b, true_logs],
    )


    summary = az.summary(trace,var_names=["m","b","logs"])
    

    print(summary)
    
    plt.show()
