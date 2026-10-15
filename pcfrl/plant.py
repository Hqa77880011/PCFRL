import numpy as np


class Battery:
    def __init__(self, fidelity='SPMe', temperature=25., soh=1., perturbation=None):
        import pybamm
        self.pybamm = pybamm
        model_class = {'SPMe': pybamm.lithium_ion.SPMe, 'DFN': pybamm.lithium_ion.DFN}[fidelity]
        model = model_class(options={'thermal': 'isothermal', 'calculate heat source for isothermal models': 'true',
                                     'SEI': 'solvent-diffusion limited',
                                     'lithium plating': 'partially reversible'})
        parameters = pybamm.ParameterValues('OKane2022')
        parameters.update(dict(pybamm.ParameterValues('Chen2020').items()), check_already_exists=False)
        parameters.update({'Current function [A]': '[input]', 'Ambient temperature [K]': '[input]',
                           'Initial temperature [K]': temperature + 273.15, 'Nominal cell capacity [A.h]': 5. * soh})
        for name, factor in (perturbation or {}).items():
            parameters.update({name: parameters[name] * factor})
        self.simulation = pybamm.Simulation(model, parameter_values=parameters, solver=pybamm.CasadiSolver(mode='safe'))
        self.solution = self.simulation.solve([0, 1e-6], initial_soc=.1,
                                              inputs={'Current function [A]': 0., 'Ambient temperature [K]': temperature + 273.15})
        self.core, self.surface, self.ambient = temperature, temperature, temperature
        self.soh, self.soc, self.elapsed, self.previous, self.loss = soh, .1, 0., 0., 0.

    def scalar(self, solution, name):
        return float(np.asarray(solution[name].entries)[..., -1].mean())

    def state(self):
        return np.array([self.soc, self.core, self.surface, self.scalar(self.solution, 'Terminal voltage [V]'),
                         self.previous, self.soh,
                         self.scalar(self.solution, 'X-averaged negative electrode surface potential difference [V]'),
                         self.elapsed / 1800.], dtype=np.float32)

    def preview(self, current):
        solution = self.simulation.step(1., starting_solution=self.solution, save=False,
                                        inputs={'Current function [A]': -float(current),
                                                'Ambient temperature [K]': self.core + 273.15})
        voltage = self.scalar(solution, 'Terminal voltage [V]')
        margin = self.scalar(solution, 'X-averaged negative electrode surface potential difference [V]')
        heat = self.scalar(solution, 'Total heating [W]')
        core = self.core + (heat - (self.core - self.surface) / 1.2) / 60.
        surface = self.surface + ((self.core - self.surface) / 1.2 - (self.surface - self.ambient) / 3.) / 20.
        soc = self.soc + float(current) / (3600 * 5 * self.soh)
        loss = self.scalar(solution, 'Loss of capacity to negative SEI [A.h]')
        loss += self.scalar(solution, 'Loss of capacity to negative lithium plating [A.h]')
        return solution, np.array([voltage, core, margin, soc]), surface, loss

    def safe_bound(self):
        valid, observations = [], []
        for current in np.linspace(0, 20, 41):
            try:
                _, values, _, _ = self.preview(current)
                valid.append(bool(values[0] <= 4.2 and values[1] <= 45 and values[2] >= 0 and values[3] <= .8))
                observations.append(values)
            except self.pybamm.SolverError:
                valid.append(False)
                observations.append(np.array([np.inf, np.inf, -np.inf, np.inf]))
        if not valid[0]:
            return 0., True
        first_invalid = next((j for j, v in enumerate(valid) if not v), 41)
        if first_invalid == 41:
            return 20., False
        values = np.array(observations[:first_invalid + 1])
        monotone = (np.diff(values[:, 0]) >= -1e-6).all() and (np.diff(values[:, 1]) >= -1e-6).all() and (np.diff(values[:, 2]) <= 1e-6).all()
        low, high = (first_invalid - 1) * .5, first_invalid * .5
        if monotone:
            for _ in range(12):
                middle = (low + high) / 2
                try:
                    _, value, _, _ = self.preview(middle)
                    feasible = value[0] <= 4.2 and value[1] <= 45 and value[2] >= 0 and value[3] <= .8
                except self.pybamm.SolverError:
                    feasible = False
                if feasible:
                    low = middle
                else:
                    high = middle
        return low, False

    def step(self, current):
        old = self.state()
        solution, values, surface, loss = self.preview(current)
        self.solution, self.core, self.surface = solution, values[1], surface
        self.soc, self.elapsed, self.previous = values[3], self.elapsed + 1, current
        delta_loss = max(0., loss - self.loss)
        self.loss = loss
        dissipated_wh = max(0., self.scalar(solution, 'Total heating [W]')) / 3600
        reward = -1 - 8 * delta_loss / .001 - .5 * dissipated_wh - .05 * ((current - old[4]) / 20) ** 2
        reward += 120 * (self.soc >= .8 - 1e-8)
        costs = np.maximum([(values[0] - 4.2) / .2, (self.core - 45) / 10,
                            -values[2] / .1, (current - 20) / 20,
                            max(self.soc - 1, -self.soc)], 0).astype(np.float32)
        done = self.soc >= .8 - 1e-8 or self.elapsed >= 1800
        return self.state(), float(reward), costs, done, {'capacity_loss_ah': delta_loss,
                'charger_wh': values[0] * current / 3600,
                'stored_wh': max(0., values[0] * current / 3600 - dissipated_wh), 'dissipated_wh': dissipated_wh}
