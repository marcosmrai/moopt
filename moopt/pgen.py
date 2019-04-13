# -*- coding: utf-8 -*-
"""
Noninferior Set Estimation implementation

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Cohon, Jared L., Church, Richard L., Sheer, Daniel P.
    Generating multiobjective trade‐offs: An algorithm for bicriterion problems
    1979
    Water Resources Research
"""
# License: BSD 3 clause

import numpy as np
import copy
import logging
import time
from scipy.spatial import ConvexHull
import scipy.optimize as opt

from .scalarization_interface import scalar_interface, w_interface, \
                                     single_interface

MAXINT = 200000000000000

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def max_dist(W):
    M = W[0].shape[0]

    def calcw(a):
        return np.sum([ai * W[i]for i, ai in enumerate(a)], axis=0)

    def f(a):
        return -sum([sum((calcw(a) - wi)**2) for wi in W])

    a0 = np.ones(M)/M
    opts = {'maxiter': 100}

    def wpos(i):
        return lambda a: a[i]

    constraints = [{'type': 'eq', 'fun': lambda a: sum(a)-1}] +\
                  [{'type': 'ineq', 'fun': wpos(i)} for i in range(M)]

    out = opt.minimize(f, a0, constraints=constraints,
                       tol=0.001, options=opts)

    w = np.array(np.sum([ai * W[i] for i, ai in enumerate(out.x)], axis=0))
    return w


class w_node():
    def __init__(self, parents, globalL, globalU, maxU, weightedScalar, w=None,
                 distance='l2', norm=True, wneg='maxdist'):
        if not (isinstance(weightedScalar, scalar_interface) and
                isinstance(weightedScalar, w_interface)):
            raise ValueError(weightedScalar +
                             ' and must be a mo_problem implementation.')

        self.__distance = distance
        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__globalL, self.__globalU, self.__maxU = globalL, globalU, maxU
        self.__parents = parents
        self.__norm = norm
        self.__wneg = wneg
        self.__calcW(w)
        self.__calcImportance()

    @property
    def importance(self):
        return self.__importance

    @property
    def parents(self):
        return self.__parents

    @property
    def solution(self):
        return self.__solution

    @property
    def w(self):
        return self.__w

    def __normf(self, obj):
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w):
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/np.abs(w_).sum()
        else:
            return w

    def optimize(self):
        self.__solution = copy.copy(self.__weightedScalar)
        self.__solution.optimize(self.w)
        return self.__solution

    def wDistParents(self):
        dist = np.linalg.norm(self.__normw(self._worig) -
                              np.mean([self.__normw(p.w)
                                       for p in self.__parents], axis=0))
        return -2 + dist

    def __calcImportance(self):
        '''
        if any(self._worig < 0):
            self.__importance = self.wDistParents()
            return
        '''

        X = [[i for i in self.__normw(p.w)] for p in self.__parents]
        y = [self.__normf(p.objs) @ self.__normw(p.w) for p in self.__parents]

        if np.linalg.matrix_rank(X) < self.__M:
            self.__importance = self.wDistParents()
            return

        r = self.__normf(self.__parents[0].objs)
        p = np.linalg.solve(X, y)

        self.r = r
        self.p = p

        if all(self._worig <= 0):
            self.__importance = -float('inf')
            return

        if any(self._worig+10**-5 >= 1):
            self.__importance = -float('inf')
            return

        if (any(p < self.__normf(self.__globalL)) or
                any(p > self.__normf(self.__maxU))):
            # self.__importance = -float('inf')
            self.__importance = self.wDistParents()
            return

        if self.__distance == 'l2':
            self.__importance = self.__normw(self._worig) @ (r - p)# /
                                 #np.linalg.norm(self.__normw(self._worig)))
            # print(self.__importance)
            # print(self.__normw(self._worig))
            # print(r)
            # print(p)
            return
        else:
            self.__importance = self.__normw(self._worig)@(r-p)
            return

    def __calcW(self, w=None):
        if w is None:
            X = [[i for i in self.__normf(p.objs)] + [-1]
                 for p in self.__parents]
            X = np.array(X + [[1]*self.__M+[0]])
            y = [0]*self.__M+[1]

            w = np.linalg.solve(X, y)[:self.__M]

        if w @ self.__normf(self.__globalL) < w @ self.__normf(self.__globalU):
            w /= np.abs(w).sum()
        else:
            w /= -np.abs(w).sum()

        if self.__norm:
            w = w/(self.__globalU-self.__globalL)

        self._worig = w

        if any(w < 0):
            if self.__wneg == 'nulify':
                w = np.abs(w * (w >= 0))
            elif self.__wneg == 'maxdist':
                w = max_dist([self.__normw(p.w) for p in self.__parents])

            w = w/w.sum()
            if self.__norm:
                w = w/(self.__globalU-self.__globalL)

        self.__w = w


class pgen():
    def __init__(self, weightedScalar=None, singleScalar=None,
                 targetSize=None, norm=True, timeLimit=float('inf')):
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
        self.__target_size = (targetSize if targetSize is not None else
                              20*self.__weightedScalar.M)
        self.__norm = norm
        self.__timeLimit = timeLimit

        self.__convexHull = None

        self.__solutionsList = []
        self.__candidatesList = {}

    def __del__(self):
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def target_size(self): return self.__target_size

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def importances(self): return self.__importances

    def __normf(self, obj):
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def inicialization(self):
        self.__M = self.__singleScalar.M
        neigO = []
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i+1)+'th individual minima')
            singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)
            parents.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)
        self.__maxU = neigO.max(0)

        next_ = w_node(parents, self.__globalL, self.__globalU, self.__maxU,
                       self.__weightedScalar, norm=self.__norm)

        self.__selected = [set(range(self.__M))]

        self.__importances = [next_.importance]

        solution = next_.optimize()
        self.update(next_, solution)

    def update(self, node, solution):
        self.__solutionsList.append(solution)
        self.__branch(node, solution)
        self.__maxU = np.max([self.__maxU, solution.objs], axis=0)

        logger.debug(str(len(self.solutionsList)) +
                     'th solution, current importance ' + str(node.importance))

    def __branch(self, node, solution):
        old_candidates = self.__candidatesList
        self.__candidatesList = {}

        if self.__convexHull is None:
            self.__hullPoints = [self.__normf(sol.objs)
                                 for sol in self.solutionsList]
            self.__convexHull = ConvexHull(self.__hullPoints,
                                           qhull_options='Q12 QJ')
        else:
            self.__hullPoints += [self.__normf(solution.objs)]
            self.__convexHull = ConvexHull(self.__hullPoints,
                                           qhull_options='Q12 QJ')

        nfacets = self.__convexHull.simplices.shape[0]
        next_facet = None
        for i in range(nfacets):
            simplice = set(self.__convexHull.simplices[i])

            w_ch = self.__convexHull.equations[i][:-1]
            if simplice in self.__selected:
                continue

            if (*simplice,) in old_candidates:
                new_node = old_candidates[(*simplice,)]
            else:
                parents = [self.__solutionsList[pindex] for pindex in simplice]
                new_node = w_node(parents, self.__globalL, self.__globalU,
                                  self.__maxU, self.__weightedScalar, w=w_ch,
                                  norm=self.__norm)

            self.__candidatesList[(*simplice,)] = new_node

            if next_facet is None:
                next_facet = simplice
            else:
                if (self.__candidatesList[(*simplice,)].importance >
                        self.__candidatesList[(*next_facet,)].importance):
                    next_facet = simplice

        self.__next_facet = next_facet

    def select(self):
        if self.__candidatesList=={}:
            return None
        self.__selected += [self.__next_facet]
        next_ = self.__candidatesList[(*self.__next_facet,)]
        self.__importances += [next_.importance]
        return next_

    def optimize(self):
        start = time.perf_counter()
        self.inicialization()

        node = self.select()

        while (node is not None and
               len(self.solutionsList) < self.target_size and
               time.perf_counter()-start<self.__timeLimit):
            
            solution = node.optimize()
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.perf_counter() - start
