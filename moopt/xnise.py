"""
Noninferior Set Estimation implementation
"""
"""
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
import pulp as lp
import copy
import logging
import time
from scipy.spatial import ConvexHull

from .scalarization_interface import scalar_interface, w_interface, single_interface

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

        self.__globalL, self.__globalU = globalL, globalU
        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__norm = norm
        self.__w = w
        self.__point = point
        self.__uR = None
        self.__calcW(w)
        self.calcImportance(solutionsList)

    @property
    def w(self):
        return self.__w

    @property
    def importance(self):
        return self.__importance

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
            if (w_ @ self.__normf(self.__globalL) <
                w_ @ self.__normf(self.__globalU)):
                w_ /= np.abs(w_).sum()
            else:
                w_ /= -np.abs(w_).sum()
            return w_
        else:
            return w

    def optimize(self):
        self.__solution = copy.copy(self.__weightedScalar)
        self.__solution.optimize(self.w, self.__globalU)
        return self.__solution

    def calcImportance(self, solutionsList):
        if (self.__uR is not None and
            all([self.__normw(sols.w) @ self.__uR >=
                 self.__normf(sols.objs) @ self.__normw(sols.w)
                 for sols in solutionsList])):
            return
        else:
            self.__calcImportance(solutionsList)

    def __calcImportance(self, solutionsList):
        '''Calculates the importance of a weight P2 of referenced work
        to calculate the importance.
        '''
        if any(self.w + 10**-5 >= 1):
            self.__importance = -float('inf')
            return

        oidx = [i for i in range(self.M)]
        prob = lp.LpProblem("xNISE procedure", lp.LpMinimize)

        uR = list(lp.LpVariable.dicts('uR', oidx, cat='Continuous').values())

        for value, sols in enumerate(solutionsList):
            expr = lp.lpDot(self.__normw(sols.w), uR)
            cons = self.__normf(sols.objs) @ self.__normw(sols.w)
            prob += expr >= cons

        for i in oidx:
            prob += uR[i] <= self.__normf(self.__globalU)[i]

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
            self.__importance = (self.__normw(self.w) @ self.__point -
                                 self.__normw(self.w) @ self.__uR)
        else:
            raise('Non-feasible solution')

    def __calcW(self, w=None):
        if w @ self.__normf(self.__globalL) < w @ self.__normf(self.__globalU):
            w /= np.abs(w).sum()
        else:
            w /= -np.abs(w).sum()

        if self.__norm:
            w = w/(self.__globalU-self.__globalL)

        self.__w = w


class xnise():
    def __init__(self, weightedScalar=None, singleScalar=None,
                 targetSize=None, norm=True):
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
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm
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
        self.__convexHull = None

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
            self.__convexHull = ConvexHull(self.__hullPoints)
        else:
            self.__hullPoints += [self.__normf(solution.objs)]
            self.__convexHull = ConvexHull(self.__hullPoints)

        nfacets = self.__convexHull.simplices.shape[0]
        next_importance = -float('inf')
        next_facet = None

        for i in range(nfacets):
            simplice = set(self.__convexHull.simplices[i])

            w_ch = self.__convexHull.equations[i][:-1]

            points = [self.__convexHull.points[s] for s in simplice]

            if simplice in self.__selected:
                continue

            if (*simplice,) in old_candidates:
                new_node = old_candidates[(*simplice,)]
                if new_node.importance >= next_importance:
                    new_node.calcImportance(self.__solutionsList)
            else:
                point = points[0]
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
        start = time.clock()
        self.inicialization()

        node = self.select()

        while (node is not None and
               len(self.solutionsList) < self.targetSize):
            solution = node.optimize()
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start
