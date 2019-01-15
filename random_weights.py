"""
A posteriori multiobjective optimization method based on
random weighted sum method with Smith simplex sampling.
"""
"""
Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Smith, N. and Tromble, R.
    Sampling Uniformly from the Unit Simplex Naïve Algorithms
    2004
"""

# License: BSD 3 clause

import numpy as np
import copy
import logging
import time

from .scalarization_interface import scalar_interface, w_interface,\
                                     single_interface

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class weight_iter():
    """Class to auxiliate the weight vector calculation and
    scalarization solving.
    Parameters
    ----------
    globalL : array_like, shape = [M,]
        Utopia point -- used to normalize the objectives.
    globalU : array_like, shape = [M,]
        Nadir or pseudo-Nadir point -- used to normalize the objectives.
    weightedScalar : w_interface class
        Class capable of solving the weighted sum method of the problem.
    norm : bolean
        Parameter to indicate if the optimization is normalized by the extreme
        points (utopia and (pseudo-)nadir) or not.
    w : array_like, shape = [M,]
        Weighting vector -- ponderates the objectives of
        the weighted sum method.
    Attributes
    ----------
    w : array_like, shape = [M,]
        Weighting vector -- ponderates the objectives of
        the weighted sum method.
    """

    def __init__(self, w, globalL, globalU, weightedScalar, norm=True):
        self.__weightedScalar = weightedScalar
        self.__globalL, self.__globalU = globalL, globalU
        self.__norm = norm
        self.__w = self.__calcW(w)

    @property
    def w(self):
        return self.__w

    def optimize(self):
        self.__solution = copy.copy(self.__weightedScalar)
        self.__solution.optimize(self.w)
        return self.__solution

    def __calcW(self, w=None):
        if self.__norm:
            w = w/(self.__globalU-self.__globalL)

        return w


class random_weights():
    """A posteriori multiobjetive method based sampling random points in the
    simplex as weight vectors.
    Parameters
    ----------
    weightedScalar : w_interface class
        Class capable of solving the weighted sum method of the problem.
    singleScalar : s_interface class
        Class capable of solving the the problem for a single objective.
    targetSize : int
        Number of points of the representation.
    norm : bolean
        Parameter to indicate if the optimization is normalized by the extreme
        points (utopia and (pseudo-)nadir) or not.
    Attributes
    ----------
    solutionsList : list of scalar_interface classes.
        Solutions that represent the Pareto-frontier.
    targetSize : int
        Number of points of the representation.
    """
    def __init__(self, weightedScalar=None, singleScalar=None,
                 targetSize=None, norm=True):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if (not isinstance(weightedScalar, scalar_interface) or
            not isinstance(weightedScalar, w_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError('weightedScalar' + ' and ' + 'singleScalar' +
                             'must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm

        self.__solutionsList = []
        self.__candidatesList = {}

    def __del__(self):
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def targetSize(self): return self.__targetSize

    def inicialization(self):
        self.__M = self.__singleScalar.M
        neigO = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i+1)+'th individual minima')
            singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

    def update(self, solution):
        self.__solutionsList.append(solution)
        logger.debug(str(len(self.solutionsList))+'th solution')

    def select(self):
        '''
        Note:
        Simplex random sampling proposed in Smith 2009.
        '''
        rnd = np.array(sorted([0] +
                              [np.random.rand() for i in range(self.__M-1)] +
                              [1]))
        w = np.array([rnd[i+1]-rnd[i] for i in range(self.__M)])
        w = w/w.sum()
        return weight_iter(w, self.__globalL, self.__globalU,
                           self.__weightedScalar, norm=self.__norm)

    def optimize(self):
        start = time.clock()
        self.inicialization()

        node = self.select()

        while (node is not None and
               len(self.solutionsList) < self.targetSize):

            solution = node.optimize()
            self.update(solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start
