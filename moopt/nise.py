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
import bisect
import copy
import logging
import time

import warnings


from .scalarization_interface import scalar_interface, w_interface, single_interface

MAXINT = 200000000000000

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class w_node():
    """
    Weight node class

    Parameters
    ----------
    parents :
    globalL : array_like, shape = [M,]
        Utopia point -- used to normalize the objectives.
    globalU : array_like, shape = [M,]
        Nadir or pseudo-Nadir point -- used to normalize the objectives.
    weightedScalar : w_interface class
        Class capable of solving the weighted sum method of the problem.
    distance :
    norm : bolean
        Parameter to indicate if the optimization is normalized by the extreme
        points (utopia and (pseudo-)nadir) or not.
    """
    def __init__(self, parents, globalL, globalU, weightedScalar,
                 distance='l2', norm=True):

        self.__distance = distance
        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__globalL, self.__globalU = globalL, globalU
        self.__parents = parents
        self.__norm = norm
        self.__calcW()
        self.__calcImportance()

    @property
    def importance(self):
        """
        Numerical value for how importante is this weighting vector
        for the next iteration.
        """
        return self.__importance

    @property
    def parents(self):
        return self.__parents

    @property
    def solution(self):
        return self.__solution

    @property
    def w(self):
        """
        Weighting vector, which ponderates the objectives of the
        weighted sum method.
        """
        return self.__w

    def __normf(self, obj):
        """
        Normalize the objectives
        """
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w):
        """
        Normalize the weights

        Parameters
        ----------

        w : array_like, shape = [M,]
            Weighting vector -- ponderates the objectives of the weighted sum method.
        """
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/w_.sum()
        else:
            return w

    @property
    def useful(self):
        P = np.array([[i for i in p.objs] for p in self.parents])
        between = ((self.__solution.objs >= P.min(axis=0)).all()
                   and (self.__solution.objs <= P.max(axis=0)).any())
        equal = ((self.__solution.objs == P[0, :]).all() or
                 (self.__solution.objs == P[1, :]).all())
        return between and not equal

    def optimize(self, hotstart=None):
        """
        Calculates the a multiobjective scalarization

        Parameters
        ----------
            hotstart : ?
        """
        self.__solution = copy.copy(self.__weightedScalar)

        self.__solution.optimize(self.w)

        return self.__solution

    def __calcImportance(self):
        """
        Calculates the importance of a weight
        """
        if self.__w is None:
            self.__importance = 0
        else:
            X = [[i for i in self.__normw(p.w)] for p in self.__parents]
            y = [self.__normf(p.objs)@self.__normw(p.w) for p in self.__parents]

            r = self.__normf(self.__parents[0].objs)
            p = np.linalg.solve(X, y)
            if self.__distance == 'l2':
                self.__importance = (self.__normw(self.w)@(r-p) /
                                     np.linalg.norm(self.__normw(self.w)))**2
            else:
                self.__importance = self.__normw(self.w)@(r-p)

    def __calcW(self):
        """
        Calculates the weighting vector
        """
        X = [[i for i in self.__normf(p.objs)]+[-1] for p in self.__parents]
        X = np.array(X + [[1]*self.__M+[0]])
        y = [0]*self.__M+[1]

        try:
            w_ = np.linalg.solve(X, y)[:self.__M]
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)

            self.__w = w_/w_.sum()
        except np.linalg.LinAlgError:
            self.__w = None


class nise():
    """
    Noninferior Set Estimation (NISE), a posteriori multi-objective
    optimization method based on weighted sum

    Parameters
    ----------
    weightedScalar : w_interface class
        Class capable of solving the weighted sum method of the problem.
    singleScalar : s_interface class
        Class capable of solving the the problem for a single objective.
    targetGap :
    targetSize : int
        Number of points of the representation.
    hotstart :
    norm : bolean
        Parameter to indicate if the optimization is normalized by the extreme
        points (utopia and (pseudo-)nadir) or not.
    timeLimit :
    objective :
    """
    def __init__(self, weightedScalar=None, singleScalar=None,
                 targetGap=0.0, targetSize=None, hotstart=[], norm=True,
                 timeLimit=float('inf'), objective='l2'):
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
        self.__norm = norm

        self.__currImp = 1
        self.__maxImp = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []
        self.__timeLimit = timeLimit
        self.__objective = objective

    def __del__(self):
        """
        Delete solutions (scalar_interface classes) list
        that represent the Pareto-frontier.
        """
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def targetSize(self): return self.__targetSize

    @property
    def targetGap(self): return self.__targetGap

    @property
    def maxImp(self): return self.__maxImp

    @property
    def currImp(self): return self.__currImp

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def hotstart(self): return self.__hotstart+self.solutionsList

    def inicialization(self):
        """
        Inicializate the objects of the scalarizations.
        Compute the solutions from the individual minima.
        Compute the global inferior bound and the global superior bound.
        Create the first region.
        """
        self.__M = self.__singleScalar.M
        if self.__M != 2:
            raise ValueError('''NISE only support MOO problems with
                             2 objectives.''')
        neigO = []
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            print('Finding '+str(i)+'th individual minima')
            logger.debug('Finding '+str(i)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)
            parents.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

        self.__candidatesList = w_node(parents, self.__globalL, self.__globalU,
                                       self.__weightedScalar, norm=self.__norm,
                                       distance=self.__objective)
        self.__candidatesList = [self.__candidatesList]

        self.__maxImp = self.__candidatesList[-1].importance
        self.__currImp = self.__candidatesList[-1].importance

    def select(self):
        """
        Selects candidate from candidate list,
        which is the next regions to be optimized

        return selected candidate
        """
        bounded_ = True
        while bounded_ and self.__candidatesList != []:
            candidate = self.__candidatesList.pop()
            bounded_ = (candidate.w < 0).any()

        if bounded_:
            return None
        else:
            return candidate

    def update(self, node, solution):
        """
        Update solutions list with a new solution

        Parameters
        ----------
            node     : selected candidate
            solution :
        """
        try:
            self.solutionsList.append(solution)
            if any([all(p.objs==node.solution.objs) for p in node.parents]):
                raise RuntimeError('Optimization issues.')
            if not node.useful:
                raise RuntimeError('Optimization issues.')
            self.__branch(node, solution)
        except RuntimeError:# as msg:
            warnings.warn('Not optimal solver or nonconvex problem')

        if self.__candidatesList != []:
            self.__currImp = self.__candidatesList[-1].importance
        gap = self.currImp/self.__maxImp
        print((str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap)))
        logger.debug(str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap))

    def __branch(self, node, solution):
        """


        Parameters
        ----------
            node      : selected candidate
            solution? :
        """
        for i in range(self.__M):
            parents = [p if j != i else node.solution
                       for j, p in enumerate(node.parents)]
            boxW = w_node(parents, self.__globalL, self.__globalU,
                          self.__weightedScalar, norm=self.__norm,
                          distance=self.__objective)

            #print(boxW.w)

            # avoiding over representation of some regions
            maxdist = max(abs(parents[0].objs-parents[1].objs)/(self.__globalU-self.__globalL))

            if boxW.w is not None and not (boxW.w < 0).any() and maxdist>1./self.targetSize:
                index = bisect.bisect_left([c.importance
                                            for c in self.__candidatesList],
                                           boxW.importance)
                self.__candidatesList.insert(index, boxW)

    def optimize(self):
        """
        Find a set of efficient solutions
        """
        start = time.perf_counter()
        self.inicialization()

        node = self.select()

        while (node is not None and
               self.currImp/self.maxImp > self.targetGap and
               len(self.solutionsList) < self.targetSize and
               time.perf_counter()-start<self.__timeLimit):

            solution = node.optimize(hotstart=self.hotstart)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.perf_counter() - start
