import os
import random

class DataGen(object):
    """Solver for randomly generating the city data."""

    def __init__(self, config):
        """Initialize configurations."""

        # Data configurations.
        self.city_num = config.city_num
        self.city_max_distance = config.city_max_distance
        self.file_name = config.data_dir

    def generate(self):
        '''City data generation.'''
        with open(self.file_name, "w") as f:
            f.write(str(self.city_num))
            for i in range(self.city_num):
                f.write('\n')
                x = random.uniform(0., self.city_max_distance)
                y = random.uniform(0., self.city_max_distance)
                f.write(str(x)+' '+str(y))
        print('Successfully generated!')
