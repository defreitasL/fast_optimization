import spotpy as spt

class setup_spotpy(object):
    """
    setup_spotpy
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs calibration, and writes the results to an output NetCDF file.
    
    Note: The function internally uses the Yates09 function for shoreline evolution.
    
    """

    def __init__(self, model_obj):
        
        self.model_obj = model_obj

        # if self.cal_alg == 'NSGAII':
        # cfg = xr.open_dataset(self.path+'config.nc')
        # Number of generations
        # generations = cfg['generations'].values
        # Number of individuals in the population
        # n_pop = cfg['n_pop'].values
            
    def parameters(self):
        return spt.parameter.generate(self.model_obj.params)

    def simulation(self, par):
        return self.model_obj.model_sim(par)
    
    def evaluation(self):
        return self.model_obj.observations
            
    def objectivefunction(self, simulation, evaluation, params=None):
        return self.model_obj.cal_obj.obj_func(evaluation, simulation)
    
    def setup(self):
        '''
        This function sets up the calibration algorithm and runs it.
        List os avaliable methods:
        - NSGAII
        - mle
        - mc
        - dds
        - mcmc
        - sa
        - abc
        - lhs
        - rope
        - sceua
        - demcz
        - padds
        - fscabc
        '''


        if self.model_obj.cal_obj.method == 'NSGAII':
            self.sampler = spt.algorithms.NSGAII(
                        spot_setup=self
            )
            self.sampler.sample(
                                self.model_obj.generations,
                                n_obj=self.model_obj.n_obj,
                                n_pop=self.model_obj.n_pop
                                )
        elif self.model_obj.cal_obj.method == 'mle':
            self.sampler = spt.algorithms.mle(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'mc':
            self.sampler = spt.algorithms.mc(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'dds':
            self.sampler = spt.algorithms.dds(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'mcmc':
            self.sampler = spt.algorithms.mcmc(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'sa':
            self.sampler = spt.algorithms.sa(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'abc':
            self.sampler = spt.algorithms.abc(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'lhs':
            self.sampler = spt.algorithms.lhs(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'rope':
            self.sampler = spt.algorithms.rope(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'sceua':
            self.sampler = spt.algorithms.sceua(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'demcz':
            self.sampler = spt.algorithms.demcz(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'padds':
            self.sampler = spt.algorithms.padds(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)
        elif self.model_obj.cal_obj.method == 'fscabc':
            self.sampler = spt.algorithms.fscabc(
                        spot_setup=self
            )
            self.sampler.sample(self.model_obj.repetitions)


        results = self.sampler.getdata()

        
        return results