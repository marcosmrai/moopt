import numpy as np
import bisect
import copy
import logging
import time

from .mo_interface import bb_interface, node_interface
from .scalarization_interface import scalar_interface, w_interface, single_interface

MAXINT = 200000000000000

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

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

    def optimize(self, hotstart = None):
        """Find the optimizer class"""
        self.__solution = copy.copy(self.__weightedScalar)
        try:
            self.__solution.optimize(self.__w, hotstart)
        except:
            self.__solution.optimize(self.__w)
        return self.__solution


    def __calcImportance(self):
        """Calculates an importance of a box"""
        X = [[i for i in p.w] for p in self.__parents]
        y = [np.dot(p.objs, p.w) for p in self.__parents]

        p = np.linalg.solve(X,y)
        q = p-self.__w*(np.dot(self.__w, p) - np.dot(self.__w, self.__parents[0].objs))
        self.__importance = np.dot(self.__w,q-p)**2

    def __calcW(self):
        X = [[i for i in p.objs]+[-1] for p in self.__parents]
        X = np.array(X + [[1]*self.__M+[0]])
        y = [0]*self.__M+[1]
        self.__w = np.linalg.solve(X,y)[:self.__M]

class nise(bb_interface):
    def __init__(self, weightedScalar = None, singleScalar = None, target_gap=0.0, min_gap=0.0, target_size=None, min_size=MAXINT, hotstart=[]):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if not isinstance(weightedScalar, scalar_interface) or not isinstance(weightedScalar, w_interface) or \
            not  isinstance(singleScalar, scalar_interface) or not  isinstance(singleScalar, single_interface):
            raise ValueError('weightedScalar'+' and '+'singleScalar'+'must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__target_gap = target_gap
        self.__min_gap = min_gap
        self.__target_size = target_size if target_size!=None else 20*self.__weightedScalar.M
        self.__min_size = min_size

        self.__lowerBound = 0
        self.__upperBound = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []

    def __del__(self):
        del self.__solutionsList

    @property
    def min_size(self): return self.__min_size
    
    @property
    def target_size(self): return self.__target_size

    @property
    def min_gap(self): return self.__min_gap

    @property
    def target_gap(self): return self.__target_gap

    @property
    def upperBound(self): return self.__upperBound

    @property
    def lowerBound(self): return self.__lowerBound

    @property
    def solutionsList(self): return self.__solutionsList
    
    @property
    def hotstart(self): return self.__hotstart+self.solutionsList


    def inicialization(self):
        """ Inicializate the objects of the scalarizations.
            Compute the solutions from the individual minima.
            Compute the global inferior bound and the global superior bound.
            Create the first region.

        Returns
        -------
        """

        self.__M = self.__singleScalar.M
        if self.__M != 2:
            raise ValueError('NISE only support MOO problems with 2 objectives.')
        neigO=[]; parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
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

            if not (boxW.w<0).any() and maxdist>1./self.min_size:
                index = bisect.bisect_left([c.importance for c in self.__candidatesList],boxW.importance)
                self.__candidatesList.insert(index,boxW)

    def optimize(self):
        """Find a set of efficient solutions

        Parameters
        ----------
        Returns
        -------
        """
        start = time.clock()
        self.inicialization()

        node = self.select()

        while node!=None and \
              (self.upperBound-self.lowerBound)/self.upperBound>self.target_gap and \
              len(self.solutionsList)<self.target_size and \
              ((self.upperBound-self.lowerBound)/self.upperBound>self.min_gap or \
               len(self.solutionsList)<self.min_size):
                  
            solution = node.optimize(hotstart=self.hotstart)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start