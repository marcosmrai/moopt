"""
Normal Constraint
"""
"""
Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Messac, A., Mattson, C.
    Normal constraint method with guarantee of even representation of complete Pareto frontier
    2004
    AIAA Journal
"""
# License: BSD 3 clause

import numpy as np
import copy
import logging
import time

from moopt.scalarization_interface import scalar_interface, single_interface

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class nc():
    def __init__(self, normalScalar=None, singleScalar=None,
                 targetSize=None, norm='sanchis'):  # sanchis simple no
        self.__solutionsList = scalar_interface
        if (not isinstance(normalScalar, scalar_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError('''normalScalar and singleScalar must be a
                             mo_problem implementation.''')

        self.__normalScalar = normalScalar
        self.__singleScalar = singleScalar
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm

        self.__solutionsList = []
        self.__candidatesList = []

    @property
    def targetSize(self): return self.__targetSize

    @property
    def solutionsList(self): return self.__solutionsList

    def __num_sol(self, steps, accum):
        if steps == []:
            return 1
        else:
            csteps = copy.copy(steps)
            s = csteps.pop(0)
            sols = 0
            i = 0
            while True:
                if accum + i*s <= 1:
                    sols += self.__num_sol(csteps, accum + i*s)
                else:
                    break
                i += 1
            return sols

    def __find_steps(self, max_sol):
        for m0 in range(2, max_sol + 1):
            m = [int(m0*np.linalg.norm(self.__Ndir[i, :]) /
                     np.linalg.norm(self.__Ndir[0, :]))
                 for i in range(self.__M-1)]
            m = [mp if mp > 1 else 2 for mp in m]
            steps = [1/(mp-1) for mp in m]
            n_sol = self.__num_sol(steps, 0)
            if n_sol >= max_sol:
                break
        return m

    def __comb(self, mvec, comb=[]):
        if mvec == []:
            bla = comb+[1-sum(comb)]
            return [bla]
        cmvec = copy.copy(mvec)
        m = cmvec.pop(0)
        delta = 1/(m-1)
        ncomb = []
        for i in range(m):
            auxcomb = comb+[delta*i]
            if sum(auxcomb) > 1:
                break
            ncomb += self.__comb(cmvec, auxcomb)
        return ncomb

    def inicialization(self, oArgs):
        self.__M = self.__singleScalar.M
        neigO = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i)+'th individual minima')
            singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(axis=0)
        self.__globalU = neigO.max(axis=0)

        print(self.__norm)
        if self.__norm == 'sanchis':
            self.__normIndivB = (np.ones((self.__M, self.__M)) -
                                 np.eye(self.__M))
            Mu_ = self.__normIndivB
            Mu = (np.array(neigO)-self.__globalL).T
            self.__T = np.linalg.solve(Mu.T, Mu_.T).T
        elif self.__norm == 'simple':
            self.__normIndivB = ((np.array(neigO) - self.__globalL) /
                                 (self.__globalU - self.__globalL)).T
            self.__T = np.diag(1/(self.__globalU - self.__globalL))
        elif self.__norm == 'no':
            self.__normIndivB = (np.array(neigO) - self.__globalL)
            self.__T = np.diag(1)

        self.__Ndir = self.__normIndivB[:, [-1]]-self.__normIndivB[:, :-1]
        mvec = self.__find_steps(self.targetSize)
        self.__combs = [np.array(c) for c in self.__comb(mvec)]

    def update(self, solution):
        ''' if not dominated(solution.objs,self.solutionsList)
        and solution.feasible:'''
        if solution.feasible:
            self.__solutionsList.append(solution)
        logger.debug('New solution found. ' + str(len(self.__solutionsList)) +
                     ' solutions')

    def optimize(self, *oArgs):
        start = time.clock()
        self.inicialization(oArgs)

        for comb_ in self.__combs:
            X_ = self.__normIndivB @ comb_
            normalS = copy.copy(self.__normalScalar)
            normalS.optimize(X_, self.__Ndir, self.__globalL, self.__T)
            self.update(normalS)
        self.__fit_runtime = time.clock() - start
