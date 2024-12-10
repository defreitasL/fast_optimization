from .objectives_functions import multi_obj_indexes
from .NSGAII import nsgaii_algorithm_ts
from .SPEA2 import spea2_algorithm
from .SCE_UA import sce_ua_algorithm
from .SimulatedAnnealing import simulated_annealing

class config_cal(object):
    """
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, cfg):

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']

        if self.cal_alg == 'NSGAII': 
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.pressure = cfg['pressure']
            self.regeneration_rate = cfg['regeneration_rate']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.n_restarts = cfg['n_restarts']
            self.indexes = multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'SPEA2':
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            self.pressure = cfg['pressure']
            self.regeneration_rate = cfg['regeneration_rate']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.m = cfg['m']
            self.eta_mut = cfg['eta_mut']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.n_restarts = cfg['n_restarts']
            self.indexes = multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'SCE-UA':
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            # self.magnitude = cfg['magnitude']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.regeneration_rate = cfg['regeneration_rate']
            self.eta_mut = cfg['eta_mut']
            self.num_complexes = cfg['num_complexes']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.n_restarts = cfg['n_restarts']
            self.indexes = multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'Simulated Annealing':
            self.max_iterations = cfg['max_iterations']
            self.initial_temperature = cfg['initial_temperature']
            self.cooling_rate = cfg['cooling_rate']
            self.n_restarts = cfg['n_restarts']
            self.indexes = multi_obj_indexes(self.metrics)
        

    def calibrate(self, model):
        """
        Calibrate the model.
        """
        if self.cal_alg == 'NSGAII':
            return nsgaii_algorithm_ts(
                model.model_sim, 
                model.Obs_splited, 
                model.init_par, 
                self.num_generations, 
                self.population_size,
                self.cross_prob, 
                self.mutation_rate, 
                self.pressure, 
                self.regeneration_rate,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes,
                self.n_restarts
                )
        elif self.cal_alg == 'SPEA2':
            return spea2_algorithm(
                model.model_sim, 
                model.Obs_splited, 
                model.init_par,  
                self.num_generations, 
                self.population_size, 
                self.cross_prob, 
                self.mutation_rate, 
                self.pressure, 
                self.regeneration_rate,
                self.m,
                self.eta_mut,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes,
                self.n_restarts)
        elif self.cal_alg == 'SCE-UA':
            return sce_ua_algorithm(
                model.model_sim, 
                model.Obs_splited, 
                model.init_par, 
                self.num_generations, 
                self.population_size,
                self.cross_prob,
                self.mutation_rate,
                self.regeneration_rate,
                self.eta_mut,
                self.num_complexes,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes,
                self.n_restarts)
        elif self.cal_alg == 'Simulated Annealing':
            return simulated_annealing(
                model.model_sim, 
                model.Obs_splited, 
                model.init_par, 
                self.max_iterations, 
                self.initial_temperature, 
                self.cooling_rate,
                self.indexes,
                self.n_restarts)