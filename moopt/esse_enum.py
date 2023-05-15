import numpy as np
import copy
import logging
import time

#from moopt.mo_interface import node_interface
from moopt.scalarization_interface import scalar_interface, single_interface, box_interface

def dominated(objs,solutionList):
    for sol in solutionList:
        if (sol.objs<=objs).all():
            return True
    return False

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class box():
    def __init__(self, l, u, globalL, globalU, boxScalar=None):
        #if not (isinstance(boxScalar, box_interface) and isinstance(boxScalar, scalar_interface)):
        #    raise ValueError(boxScalar+' and must be a mo_problem implementation.')
        self.__boxScalar = boxScalar
        self.__l,self.__u,self.__globalL,self.__globalU=l, u, globalL, globalU
        self.__importance = self.__calcImportance()

    @property
    def l(self):
        return self.__l
    
    @property
    def u(self):
        return self.__u
    
    @property
    def importance(self)        :
        """Calculates an importance of a box"""
        return self.__importance

    @property
    def solution(self):
        """Find the optimizer class"""
        return self.__solution

    def optimize(self, **kwargs):
        if 'solutionsList' in kwargs:
            solutionsList = kwargs["solutionsList"]
        else:
            solutionsList = []

        """Find the optimizer class"""
        self.__solution = copy.copy(self.__boxScalar)
        self.__solution.optimize(self.__l,self.__u)#, solutionsList=solutionsList)
        return self.__solution

    def __calcImportance(self):
        return ((self.__u-self.__l)/(self.__globalU-self.__globalL)).prod()

class esse_enum():
    def __init__(self, boxScalar, singleScalar,
                 targetGap=0.0, targetSize=None):
        self.__solutionsList = scalar_interface
        self.__solutionsList = box_interface
        #if not isinstance(boxScalar, box_interface) or not isinstance(boxScalar, scalar_interface) or \
        #    not isinstance(singleScalar, scalar_interface) or not isinstance(singleScalar, single_interface):
        #    raise ValueError('boxScalar and singleScalar must be a mo_problem implementation.')

        self.__boxScalar = boxScalar
        self.__singleScalar = singleScalar
        self.__targetGap = targetGap
        self.__targetSize = targetSize if targetSize!=None else 100*self.__weightedScalar.M

        self.__lowerBound = 0
        self.__upperBound = 1
        self.__solutionsList = []
        self.__candidatesList = []


    @property
    def targetGap(self): return self.__targetGap

    @property
    def targetSize(self): return self.__targetSize

    @property
    def upperBound(self): return self.__upperBound

    @property
    def lowerBound(self): return self.__lowerBound

    @property
    def solutionsList(self): return self.__solutionsList


    def inicialization(self):
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
        self.__M = self.__singleScalar.M
        neigO=[]
        for i in range(self.__M):
            logger.debug('Finding '+str(i)+'th individual minima')
            singleS = copy.copy(self.__singleScalar)#.copy()
            singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)

        neigO=np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

        #logger.info('globalU'+str(self.__globalU))
        #logger.info('globalL'+str(self.__globalL))

        self.__candidatesList = box(self.__globalL, self.__globalU, self.__globalL, self.__globalU, self.__boxScalar)
        self.__candidatesList = [self.__candidatesList]
        self.__candidatesImp = [self.__candidatesList[0].importance]



    def select(self):
        """ Selects the next regions to be optimized"""
        bounded_ = True
        while bounded_ and self.__candidatesList !=[]:
            index = np.argmax(self.__candidatesImp)
            self.__candidatesImp.pop(index)
            candidate = self.__candidatesList.pop(index)
            bounded_ = dominated(candidate.l,self.solutionsList)
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
        explorable = solution.feasible and solution.alpha>=0 and solution.alpha<=1
        if explorable:
            self.__branch(solution)

        self.__solutionsList.append(solution)

        if not explorable:
            self.__upperBound -= node.importance
        else:
            self.__lowerBound += ((solution.u-solution.c)/(self.__globalU-self.__globalL)).prod()
            self.__upperBound -= solution.optimum*((solution.c-solution.l)/(self.__globalU-self.__globalL)).prod()
        gap = (self.upperBound-self.lowerBound)/self.upperBound
        logger.debug('Current state '+str(self.__upperBound)+' '+str(self.__lowerBound)+' '+str(gap))#+' '+str(solution.alpha))


    def __branch(self, solution):
        """ Using a non dominated center and calculates 2**K new regions

        Parameters
        ----------
        cand: box_scalar object
            A box scalarization object already optimized and feasible
        """
        for i in range(solution.optimum,2**(self.__M)-1):
            piv = [int(j) for j in '0'*(self.__M-len(bin(i)[2:]))+bin(i)[2:]]
            newU=np.array([piv[j]*solution.u[j] + (1-piv[j])*solution.c[j] for j in range(self.__M)])
            newL=np.array([piv[j]*solution.c[j]+(1-piv[j])*solution.l[j] for j in range(self.__M)])

            boxS = box(newL, newU, self.__globalL, self.__globalU, self.__boxScalar)

            if dominated(boxS.l,self.solutionsList):
                self.__upperBound -= boxS.importance
            else:
                self.__candidatesList.append(boxS)
                self.__candidatesImp.append(boxS.importance)

    def optimize(self):
        """Find a set of efficient solutions

        Parameters
        ----------
        oArgs: tuple
            Arguments used by baseOpt
        Returns
        -------
        """
        start = time.clock()
        self.inicialization()

        node = self.select()

        while node!=None and \
              (self.upperBound-self.lowerBound)/self.upperBound>self.targetGap and\
              len(self.solutionsList)<self.targetSize:
            solution = node.optimize(solutionsList = self.solutionsList)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start
