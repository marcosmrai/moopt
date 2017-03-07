import numpy as np
import bisect
import copy
import logging
import time

'''
try:
    from .mo_interface import bb_interface, node_interface
    from .scalarization_interface import scalar_interface, w_interface, single_interface
except:
    import sys
    sys.path.append('.')
    from mo_interface import bb_interface, node_interface
    from scalarization_interface import scalar_interface, w_interface, single_interface
'''

from .mo_interface import bb_interface, node_interface
from .scalarization_interface import scalar_interface, w_interface, single_interface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class w_node(node_interface):
    def __init__(self, parents, globalL, globalU, weightedScalar):
        if not (isinstance(weightedScalar, scalar_interface) and isinstance(weightedScalar, w_interface)):
            raise ValueError(weightedScalar+' and must be a mo_problem implementation.')
        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__globalL,self.__globalU=globalL, globalU
        self.__parents = parents
        self.__calcW()
        self.__calcImportance()

    @property
    def importance(self):
        return self.__importance

    @property
    def parents(self):
        return self.__parents

    @property
    def solution(self):
        """Find the optimizer class"""
        return self.__solution

    @property
    def w(self):
        return self.__w

    def optimize(self, oArgs, solutionsList = None):
        """Find the optimizer class"""
        self.__solution = copy.copy(self.__weightedScalar)
        self.__solution.mo_optimize(self.__w, *oArgs, solutionsList)
        return self.__solution


    def __calcImportance(self):
        """Calculates an importance of a box"""
        X = [[i for i in p.w] for p in self.__parents]
        y = [np.dot(p.objs, p.w) for p in self.__parents]

        p = np.linalg.solve(X,y)
        q = p-self.__w*(np.dot(self.__w, p) - np.dot(self.__w, self.__parents[0].objs))
        self.__importance = np.dot(self.__w,q-p)**2#*np.dot(self.__w,self.__w)
        #self.__importance = np.linalg.norm((self.__parents[0].objs-self.__parents[1].objs)/(self.__globalU-self.__globalL),np.inf)#np.inf)

    def __calcW(self):
        X = [[i for i in p.objs]+[-1] for p in self.__parents]
        X = np.array(X + [[1]*self.__M+[0]])
        y = [0]*self.__M+[1]
        self.__w = np.linalg.solve(X,y)[:self.__M]

class nise(bb_interface):
    def __init__(self, gap=0.01, minsize=50, weightedScalar = None, singleScalar = None):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if not isinstance(weightedScalar, scalar_interface) or not isinstance(weightedScalar, w_interface) or \
            not  isinstance(singleScalar, scalar_interface) or not  isinstance(singleScalar, single_interface):
            raise ValueError('weightedScalar'+' and '+'singleScalar'+'must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__gap = gap
        self.__minsize = minsize

        self.__lowerBound = 0
        self.__upperBound = 1
        self.__solutionsList = []
        self.__candidatesList = []

    def __del__(self):
        del self.__solutionsList

    @property
    def minsize(self): return self.__minsize

    @property
    def gap(self): return self.__gap

    @property
    def upperBound(self): return self.__upperBound

    @property
    def lowerBound(self): return self.__lowerBound

    @property
    def solutionsList(self): return self.__solutionsList


    def inicialization(self,oArgs):
        """ Inicializate the objects of the scalarizations.
            Compute the solutions from the individual minima.
            Compute the global inferior bound and the global superior bound.
            Create the first region.

        Parameters
        ----------
        oArgs: tuple
            Arguents used by baseOpt

        Returns
        -------
        """
        self.__singleScalar.mo_ini(*oArgs)
        self.__weightedScalar.mo_ini(*oArgs)

        self.__M = self.__singleScalar.M
        if self.__M != 2:
            raise ValueError('NISE only support MOO problems with 2 objectives.')
        neigO=[]; parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            singleS.mo_optimize(i,*oArgs)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)
            parents.append(singleS)


        neigO=np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

        self.__candidatesList = w_node(parents, self.__globalL, self.__globalU, self.__weightedScalar)
        self.__candidatesList = [self.__candidatesList]


        self.__upperBound = self.__candidatesList[-1].importance
        self.__lowerBound = self.__upperBound - self.__candidatesList[-1].importance


    def select(self):
        """ Selects the next regions to be optimized"""
        bounded_ = True
        while bounded_ and self.__candidatesList !=[]:
            candidate = self.__candidatesList.pop()
            bounded_ = (candidate.w<0).any()
            if bounded_:
                self.__upperBound -= candidate.importance

        if bounded_:
            return None
        else:
            return candidate

    def update(self, node, solution):
        """ Update the variables

        Parameters
        ----------
        cand: box_scalar object
            A box scalarization object already optimized and feasible
        """
        ## If isn't in between don't explore anymore and generate a warning
        P = np.array([[i for i in p.objs] for p in node.parents])
        between = (solution.objs>=P.min(axis=0)).all() and (solution.objs<=P.max(axis=0)).any()
        equal = (solution.objs==P[0,:]).all() or (solution.objs==P[1,:]).all()
        if between and not equal:
            self.__branch(node,solution)
            self.solutionsList.append(solution)
        else:
            logger.debug('Not optimal solver or nonconvex problem')

        if self.__candidatesList!=[]:
            self.__lowerBound = self.__upperBound - self.__candidatesList[-1].importance

        #gap = (self.upperBound-self.lowerBound)/self.upperBound
        logger.debug('Current state '+str(self.__upperBound)+' '+str(self.__lowerBound)+' '+str(node.importance))



    def __branch(self, node, solution):
        """ Using a non dominated center and calculates 2**K new regions

        Parameters
        ----------
        cand: box_scalar object
            A box scalarization object already optimized and feasible
        """
        for i in range(self.__M):
            parents = [p if j!=i else node.solution for j,p in enumerate(node.parents)]
            boxW = w_node(parents, self.__globalL, self.__globalU, self.__weightedScalar)

            #avoiding over representation of some regions
            maxdist = max(abs(parents[0].objs-parents[1].objs)/(self.__globalU-self.__globalL))

            if not (boxW.w<0).any() and maxdist>1./self.__minsize:
                index = bisect.bisect_left([c.importance for c in self.__candidatesList],boxW.importance)
                self.__candidatesList.insert(index,boxW)

    def optimize(self, *oArgs):
        """Find a set of efficient solutions

        Parameters
        ----------
        oArgs: tuple
            Arguments used by baseOpt
        Returns
        -------
        """
        start = time.clock()
        self.inicialization(oArgs)

        node = self.select()

        while node!=None and ((self.upperBound-self.lowerBound)/self.upperBound>self.gap or len(self.solutionsList)<self.minsize):
            solution = node.optimize(oArgs,solutionsList = self.solutionsList)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start