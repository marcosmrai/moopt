# -*- coding: utf-8 -*-
"""
Many Objective Noninferior Estimation

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Raimundo, Marcos M.
    MONISE - Many Objective Noninferior Estimation
    2017
    arXiv
"""
# License: BSD 3 clause
import numpy as np
import copy
import logging
import time

from moopt.scalarization_interface import scalar_interface, w_interface, single_interface

from .monise_utils import weight_solv

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

MAXINT = 200000000000000


class monise():
    def __init__(self, weightedScalar, singleScalar, targetGap=0.0,
                 targetSize=None, redFact=float('inf'), smoothCount=None,
                 nodeTimeLimit=float('inf'), nodeGap=0.01, hotstart=[],
                 norm=True):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if (not isinstance(weightedScalar, scalar_interface) or
            not isinstance(weightedScalar, w_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError('''weightedScalar and singleScalar must be a
                             mo_problem implementation.''')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__targetGap = targetGap
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__nodeTimeLimit = nodeTimeLimit
        self.__nodeGap = nodeGap
        self.__redFact = redFact
        self.__norm = norm
        if smoothCount is None:
            self.__smoothCount = 1 if nodeTimeLimit == float('inf') else 5
        else:
            self.__smoothCount = smoothCount

        self.__maxImp = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []

    def __del__(self):
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def targetSize(self): return self.__targetSize

    @property
    def targetGap(self): return self.__targetGap

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def hotstart(self): return self.__hotstart+self.solutionsList

    @property
    def currImp(self):
        return max(self.__importances[-self.__smoothCount:])

    @property
    def maxImp(self): return self.__maxImp

    @property
    def importances(self): return self.__importances

    def inicialization(self):
        self.__M = self.__singleScalar.M
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i+1)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
            self.__solutionsList.append(singleS)
            parents.append(singleS)
            #print(singleS.objs)
            #input()

        objsM = np.array([[o for o in p.objs] for p in parents])
        self.__globalL = objsM.min(0)
        self.__globalU = objsM.max(0)

        first_wsol = weight_solv(parents, self.__globalL, self.__globalU,
                                 self.__weightedScalar, norm=self.__norm)

        self.__maxImp = first_wsol.importance
        self.__importances = [first_wsol.importance]
        self.__goal = self.__maxImp

        return first_wsol

    def update(self, node, solution):
        self.solutionsList.append(solution)
        gap = self.currImp/self.__maxImp
        logger.debug(str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap))

    def _next(self):
        next_wsol = weight_solv(self.solutionsList, self.__globalL,
                                self.__globalU, self.__weightedScalar,
                                goal=self.currImp * self.__redFact,
                                time_limit=self.__nodeTimeLimit,
                                mip_gap=self.__nodeGap, norm=self.__norm)
        self.__importances += [next_wsol.importance]
        return next_wsol

    def optimize(self):
        start = time.perf_counter()
        next_wsol = self.inicialization()

        while (self.currImp / self.__maxImp > self.__targetGap and
               len(self.solutionsList) < self.__targetSize):
            #print(next_wsol.w)
            #input()
            solution = next_wsol.optimize(hotstart=self.hotstart)
            self.update(next_wsol, solution)
            next_wsol = self._next()

        self.__fit_runtime = time.perf_counter() - start
