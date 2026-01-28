# modules.py
from .engine import PythonModule
from .functions import camb_calc, class_calc, run_emcee, compare_emcee_samples

class CambModule(PythonModule):
    def __init__(self, fixed_params=None, var_params=None, compare_func=None):
        super().__init__(name="CAMB",
                         function=camb_calc,
                         fixed_params=fixed_params,
                         var_params=var_params,
                         compare_func=compare_func)

class ClassModule(PythonModule):
    def __init__(self, fixed_params=None, var_params=None, compare_func=None):
        super().__init__(name="CLASS",
                         function=class_calc,
                         fixed_params=fixed_params,
                         var_params=var_params,
                         compare_func=compare_func)

class EmceeModule(PythonModule):
    def __init__(self, fixed_params=None, var_params=None, compare_func=None):
        # default compare uses compare_emcee_samples
        compare = compare_func or compare_emcee_samples
        super().__init__(name="EMCEE",
                         function=run_emcee,
                         fixed_params=fixed_params,
                         var_params=var_params,
                         compare_func=compare)
