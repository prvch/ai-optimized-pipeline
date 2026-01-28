# engine.py
import time
import os
import itertools
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def _run_single(module, combination, keys):
    param_set = dict(zip(keys, combination))
    output = module.run(**param_set)
    return param_set, output


# ===================================================
# Base Abstract Module
# ===================================================
class PipelineModule(ABC):
    def __init__(self, name, fixed_params=None, var_params=None,
                 compare_func=None):
        self.name = name
        self.fixed_params = fixed_params or {}
        self.var_params = var_params or {}
        self.compare_func = compare_func or self.default_compare
        self.latest_result = None

    @abstractmethod
    def run(self, **kwparams):
        pass

    def default_compare(self, run_output, std_output):

        """
        default comparison :
          - scalar vs scalar
          - 1D numpy arrays (compare elementwise relative error -> mean,std)
          - pandas.DataFrame with same shape: compare first column
        Returns (error, spread)
        error: mean relative error (float)
        spread: measure of spread (float or array)
        """
        # Scalars
        if np.isscalar(run_output) and np.isscalar(std_output):
            
            if std_output == 0:
                return float(abs(run_output - std_output)), 0.0
            return float(abs(run_output - std_output) / abs(std_output)), 0.0

        # pandas DataFrame
        if isinstance(run_output, pd.DataFrame) and isinstance(std_output, pd.DataFrame):
            
            if run_output.shape != std_output.shape:
                raise ValueError("DataFrame shapes must match for comparison.")
            # compare first column
            rel = np.abs(run_output.iloc[:, 0] - std_output.iloc[:, 0]) / (np.abs(std_output.iloc[:, 0]) + 1.0e-16)
            return float(rel.mean()), float(rel.std())

        # numpy arrays (1D)
        if isinstance(run_output, np.ndarray) and isinstance(std_output, np.ndarray):
            
            if run_output.shape != std_output.shape:
                raise ValueError("Array shapes must match for comparison.")
            rel = np.abs(run_output - std_output) / (np.abs(std_output) + 1e-16)
            return float(rel.mean()), float(rel.std())

        raise TypeError("default_compare: unsupported types for run_output/std_output")


# ===================================================
# Pure Python Module
# ===================================================
class PythonModule(PipelineModule):

    def __init__(self, name, function, fixed_params=None, var_params=None,
                 compare_func=None, external_command=None, workdir="."):
        super().__init__(name, fixed_params, var_params, compare_func)
        self.function = function

        #for future work
        self.external_command = external_command
        self.workdir = workdir

    def run(self, **kwparams):
        params = self.fixed_params.copy()
        params.update(kwparams)

        start = time.time()
        result = self.function(**params)
        cpu = time.time() - start

        self.latest_result = {
            **params,
            "result": result,
            "cpu_time": cpu
        }
        return self.latest_result

    def run_var_params(self, std_output, mpi=1, working_dir="."):

        if not self.var_params:
            self.latest_result = self.run()
            error, var = self.compare_func(self.latest_result["result"],
                                           std_output)
            row = {"error": error, 
                   "std": var, 
                   "cpu_time": self.latest_result["cpu_time"]}
            
            return pd.DataFrame([row])
            
        keys = list(self.var_params.keys())
        values = [self.var_params[k] for k in keys]
        combos = list(itertools.product(*values))

        rows = []

        if mpi == 1:
            for combo in tqdm(combos, desc=f"{self.name} grid"):
                param_set = dict(zip(keys, combo))
                out = self.run(**param_set)
                error, var = self.compare_func(out["result"], std_output)
                rows.append({
                    **param_set,
                    "cpu_time": out["cpu_time"],
                    "error": error,
                    "std": var
                })
            return pd.DataFrame(rows)

        else:
            # Limit workers to half of available CPUs
            max_allowed = max(1, (os.cpu_count()-2) )
            workers = min(mpi, max_allowed)
            print(f'No. of workers used = {workers}')
            
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futures = {ex.submit(_run_single, self, c, keys): c for c in combos}

                for f in tqdm(as_completed(futures), total=len(combos), desc=f"{self.name} MPI"):
                    param_set, out = f.result()
                    error, var = self.compare_func(out["result"], std_output)
                    rows.append({
                        **param_set,
                        "cpu_time": out["cpu_time"],
                        "error": error,
                        "std": var
                    })
            return pd.DataFrame(rows)


