from pyomo.contrib.sensitivity_toolbox.sens import SolverFactory
from pyomo.opt import TerminationCondition
from pyomo.common.tempfiles import TempfileManager


def solve(model, tee=False, ws=False):
    solver = SolverFactory('ipopt', sol='nl')

    flag, results = ipopt_solve(model, solver, tee=tee, warm_start=ws)

    return model, flag


def ipopt_solve(model, solver, tee=False, warm_start=False):
    TempfileManager.push()
    tempfile = TempfileManager.create_tempfile(suffix='ipopt_out', text=True)
    opts = {'output_file': tempfile,
            'halt_on_ampl_error': 'yes',
            'linear_solver': 'ma57',
            'tol': '1e-6',
            'mu_strategy': 'adaptive',
            'bound_push': '1e-10',
            'max_iter': 5000}
    if warm_start:
        opts.update({'warm_start_init_point': 'yes'})
        opts.update({'warm_start_mult_bound_push': 1e-6})
        opts.update({'mu_init': 1e-8})

    results = solver.solve(model, options=opts, tee=tee)

    term_cond = results.solver.termination_condition
    if term_cond == TerminationCondition.optimal:
        flag = 'optimal'
    else:
        flag = 'failure'

    return flag, results


def sipopt_solve(model, solver, tee=False):
    TempfileManager.push()
    tempfile = TempfileManager.create_tempfile(suffix='ipopt_out', text=True)
    opts = {'output_file': tempfile,
            'run_sens': 'yes'}

    results = solver.solve(model, options=opts, tee=tee)

    term_cond = results.solver.termination_condition
    if term_cond == TerminationCondition.optimal:
        flag = 'optimal'
    else:
        flag = 'failure'

    return model, flag, results
