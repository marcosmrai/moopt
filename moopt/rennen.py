# -*- coding: utf-8 -*-
"""
A posteriori multiobjective optimization method based on
polyhedral approximation and dummy points.

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Enhancement of Sandwich Algorithms for Approximating Higher-Dimensional
    Convex Pareto Sets
    Gijs Rennen, Edwin R. van Dam, and Dick den Hertog
    INFORMS Journal on Computing 2011 23:4, 493-517
"""
# License: BSD 3 clause

import numpy as np
import pulp as lp
import copy
import logging
import time
from scipy.spatial import ConvexHull

from .scalarization_interface import scalar_interface, w_interface, \
                                    single_interface

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class weight_iter():
    """Class to auxiliate the weight vector calculation and
    scalarization solving.
    Parameters
    ----------
    w : array_like, shape = [M,]
        Weighting vector -- ponderates the objectives of
        the weighted sum method.
    solutionsList : list of scalar_interface classes.
        Previously found solutions used here to calculate the importance of
        the weight vector w.
    globalL : array_like, shape = [M,]
        Utopia point -- used to normalize the objectives.
    globalU : array_like, shape = [M,]
        Nadir or pseudo-Nadir point -- used to normalize the objectives.
    weightedScalar : w_interface class
        Class capable of solving the weighted sum method of the problem.
    point : array_like, shape = [M,]
        Facet point -- point belonging to the w facet.
    norm : bolean
        Parameter to indicate if the optimization is normalized by the extreme
        points (utopia and (pseudo-)nadir) or not.
    Attributes
    ----------
    w : array_like, shape = [M,]
        Weighting vector -- ponderates the objectives of
        the weighted sum method.
    imporantance : float
        Numerical value for how importante is this weighting vector for the
        next iteration.
    M : int
        Number of objectives.
    """
    def __init__(self, w, solutionsList, globalL, globalU, weightedScalar,
                 point, norm=True):
        if not (isinstance(weightedScalar, scalar_interface) and
                isinstance(weightedScalar, w_interface)):
            raise ValueError(weightedScalar +
                             ' and must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__globalL, self.__globalU = globalL, globalU
        self.__norm = norm
        self.__uR = None
        self.__calcW(w)
        self.__point = point
        self.calcImportance(solutionsList)

    @property
    def importance(self):
        return self.__importance

    @property
    def w(self):
        return self.__w

    @property
    def M(self):
        return self.__M

    def __normf(self, obj):
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w):
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/w_.sum()
        else:
            return w

    def optimize(self):
        self.__solution = copy.copy(self.__weightedScalar)
        self.__solution.optimize(self.w)
        return self.__solution

    def calcImportance(self, solutionsList):
        '''If uR is already known test if it is yet feasible to avoid
        unecessary LP calculations.
        '''
        if (self.__uR is not None and
            all([self.__normw(sols.w) @ self.__uR >=
                 self.__normf(sols.objs) @ self.__normw(sols.w)
                 for sols in solutionsList])):
            return
        else:
            self.__calcImportance(solutionsList)

    def __calcImportance(self, solutionsList):
        '''Calculates the importance of a weight using Equation 2 to find the
        lower point and Proposition 4 of the referenced work
        to calculate the importance.
        '''
        oidx = [i for i in range(self.M)]
        prob = lp.LpProblem("Lower point", lp.LpMinimize)

        uR = list(lp.LpVariable.dicts('uR', oidx, cat='Continuous').values())

        for value, sols in enumerate(solutionsList):
            expr = lp.lpDot(self.__normw(sols.w), uR)
            cons = self.__normf(sols.objs) @ self.__normw(sols.w)
            prob += expr >= cons

        prob += lp.lpDot(self.__normw(self.w), uR)

        grbs = lp.solvers.GUROBI(msg=False, OutputFlag=False, Threads=1)
        if grbs.available():
            prob.solve(grbs)
        else:
            cbcs = lp.solvers.PULP_CBC_CMD(threads=1)
            prob.solve(cbcs, use_mps=False)

        feasible = False if prob.status in [-1, -2] else True

        self.__uR = np.array([lp.value(uR[i]) for i in oidx])

        if feasible:
            self.__importance = ((self.__normw(self.w) @ self.__point -
                                 self.__normw(self.w) @ self.__uR) /
                                 (self.__normw(sols.w) @ np.ones(self.M)))
        else:
            raise('Non-feasible solution')

    def __calcW(self, w=None):
        if w @ self.__normf(self.__globalL) < w @ self.__normf(self.__globalU):
            w /= np.abs(w).sum()
        else:
            w /= -np.abs(w).sum()

        if self.__norm:
            w = w/(self.__globalU-self.__globalL)

        self.__w = np.abs(w)


def convexCombination(points, point):
    '''Calculates if a point is a convex combination of another points in the
    hull, this is done to avoid unecessary point in the hull calculation.
    '''
    points_idx = [i for i in range(len(points))]
    dim_idx = [j for j in range(len(point))]
    prob = lp.LpProblem("max mean", lp.LpMinimize)

    alpha = list(lp.LpVariable.dicts('alpha', points_idx,
                                     cat='Continuous').values())

    for j in dim_idx:
        prob += (lp.lpSum([alpha[i]*points[i][j] for i in points_idx]) ==
                 point[j])

    prob += lp.lpSum([alpha[i] for i in points_idx]) == 1

    for i in points_idx:
        if all(points[i] == point):
            prob += alpha[i] == 0
        else:
            prob += alpha[i] >= 0

    grbs = lp.solvers.GUROBI(msg=False, OutputFlag=False, Threads=1)
    if grbs.available():
        prob.solve(grbs)
    else:
        cbcs = lp.solvers.PULP_CBC_CMD(threads=1)
        prob.solve(cbcs, use_mps=False)

    feasible = False if prob.status in [-1, -2] else True
    return feasible


def alreadyFound(points, point):
    return any([all(p == point) for p in points])


class rennen():
    """A posteriori multiobjective optimization method based on
    polyhedral approximation and dummy points.
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
    importances : list of floats
        The importance of the found solutions.
    """
    def __init__(self, weightedScalar=None, singleScalar=None,
                 targetSize=None,  norm=True, timeLimit=float('inf')):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if (not isinstance(weightedScalar, scalar_interface) or
            not isinstance(weightedScalar, w_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError("""weightedScalar and singleScalar
                             must be a mo_problem implementation.""")

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm
        self.__timeLimit = timeLimit

        self.__solutionsList = []
        self.__candidatesList = {}

    def __del__(self):
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def M(self): return self.__singleScalar.M

    @property
    def targetSize(self): return self.__targetSize

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

        next_ = weight_iter(np.ones(self.M)/self.M, parents, self.__globalL,
                            self.__globalU, self.__weightedScalar,
                            norm=self.__norm, point=parents[0].objs)
        self.__selected = []

        self.__importances = [next_.importance]
        self.__convexHull = None

        solution = next_.optimize()
        self.update(next_, solution)

    def update(self, node, solution):
        self.__solutionsList.append(solution)
        self.__branch(node, solution)
        self.__maxU = np.max([self.__maxU, solution.objs], axis=0)

        logger.debug(str(len(self.solutionsList)) +
                     'th solution, current importance ' + str(node.importance))

    def __dummyPoints(self, solution):
        '''Generates dummy points described at Definition 7
        of the referenced paper.
        '''
        points = []
        for i in range(self.M):
            point = self.__normf(solution.objs)
            point[i] = (self.__normf(self.__globalU)[i] * (self.M) +
                        self.__normf(self.__globalU)[i] * 0.01)
            points.append(point)
        return points

    def __newPoints(self, solution):
        return [self.__normf(solution.objs)]+self.__dummyPoints(solution)

    def __isDummy(self, point):
        return any(point > self.__normf(self.__globalU) * (self.M))

    def __allDummy(self, points):
        return all([self.__isDummy(point) for point in points])

    def __selNotDummy(self, points):
        for point in points:
            if not self.__isDummy(point):
                return point

    def __branch(self, node, solution):
        old_candidates = self.__candidatesList
        self.__candidatesList = {}

        if self.__convexHull is None:
            points = [point for sol in self.solutionsList
                      for point in self.__newPoints(sol)]
            self.__hullPoints = [point for point in points
                                 if not convexCombination(points, point)]

            self.__convexHull = ConvexHull(self.__hullPoints,
                                           qhull_options='Q12')

        elif not alreadyFound(self.__hullPoints, self.__normf(solution.objs)):
            points = self.__newPoints(solution)
            self.__hullPoints += [point for point in points
                                  if not
                                  convexCombination(self.__hullPoints+points,
                                                    point)]

            self.__convexHull = ConvexHull(self.__hullPoints,
                                           qhull_options='Q12')

        nfacets = self.__convexHull.simplices.shape[0]

        next_facet = None
        next_importance = -float('inf')

        for i in range(nfacets):
            simplice = set(self.__convexHull.simplices[i])

            w_ch = self.__convexHull.equations[i][:-1]

            points = [self.__convexHull.points[s] for s in simplice]

            if (self.__allDummy(points) or  # all dummy means irrelevant facet
                (any(w_ch < -10**-20) and not  # some negative is numerical err
                 all(w_ch <= 0))):  # all negative weights can be reversed
                continue

            if simplice in self.__selected:
                continue

            if (*simplice,) in old_candidates:
                new_node = old_candidates[(*simplice,)]
                if new_node.importance >= next_importance:
                    new_node.calcImportance(self.__solutionsList)
            else:
                point = self.__selNotDummy(points)
                new_node = weight_iter(w_ch, self.__solutionsList,
                                       self.__globalL, self.__globalU,
                                       self.__weightedScalar, norm=self.__norm,
                                       point=point)

            self.__candidatesList[(*simplice,)] = new_node

            if next_facet is None:
                next_facet = simplice
            else:
                if (self.__candidatesList[(*simplice,)].importance >=
                        next_importance):
                    next_importance = self.__candidatesList[(*simplice,)].importance
                    next_facet = simplice

        self.__next_facet = next_facet

    def select(self):
        self.__selected += [self.__next_facet]
        next_ = self.__candidatesList[(*self.__next_facet,)]
        self.__importances += [next_.importance]
        return next_

    def optimize(self):
        start = time.perf_counter()
        self.inicialization()

        node = self.select()

        while (node is not None and
               len(self.solutionsList) < self.targetSize and
               time.perf_counter()-start<self.__timeLimit):
            solution = node.optimize()
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.perf_counter() - start
