import numpy as np
import copy
import logging
import time
from sortedcontainers import SortedList
from itertools import combinations

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
        if not (isinstance(boxScalar, box_interface) and isinstance(boxScalar, scalar_interface)):
            raise ValueError(boxScalar+' and must be a mo_problem implementation.')
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
        self.__solution.optimize(self.__l,self.__u, solutionsList=solutionsList)
        return self.__solution

    def __calcImportance(self):
        return (self.__u-self.__l).prod()
        
class BoxGenerator:
    def __init__(self, l, u, alpha, globalL, globalU, optimal=True):
        self.M = u.size
        self.idx = 1 if optimal else 0
        self.l = l
        self.u = u
        self.alpha = alpha
        self.c = self.l + self.alpha*(self.u - self.l)
        
        self.parent_volume = np.prod((u-l)/(globalU-globalL))
        
        self.n_comb = 1 if self.alpha>0.5 else self.M
        self.comb_gen = combinations(range(self.M), self.n_comb)
        self.calc_next_comb()
        
    def __iter__(self):
        return self
    
    def max_volume(self):
        return self.__max_volume
    
    def calc_next_comb(self):
        try:
            self.next_comb = next(self.comb_gen)
        except:
            if self.alpha>0.5:
                self.n_comb+=1
                if self.n_comb>self.M:
                    self.next_comb = None
                    self.__max_volume = 0
                    return
            else:
                self.n_comb-=1
                if self.n_comb<1:
                    self.next_comb = None
                    self.__max_volume = 0
                    return
            self.comb_gen = combinations(range(self.M), self.n_comb)
            self.next_comb = next(self.comb_gen)
        self.__max_volume = self.parent_volume*(self.alpha**(self.M-self.n_comb))*((1-self.alpha)**(self.n_comb))

    def __next__(self):
        if self.next_comb is None:
            raise StopIteration
            
        piv = np.zeros(self.M)
        piv[list(self.next_comb)] = 1
        newU=np.array([piv[j]*self.u[j] + (1 - piv[j])*self.c[j] for j in range(self.M)])
        newL=np.array([piv[j]*self.c[j] + (1 - piv[j])*self.l[j] for j in range(self.M)])
        
        self.calc_next_comb()
        
        return newL, newU
        
class boxCandidatesNew():
    def __init__(self, globalL, globalU, boxScalar=None):
        self.boxScalar = boxScalar
        self.globalL = globalL
        self.globalU = globalU
        
        self.cand_list = SortedList(key=lambda x: x.max_volume())
            
    def append(self, boxGen):
        self.cand_list.add(boxGen)
        
    def pop(self):
        current_max = self.cand_list[-1].max_volume()
        currentL, currentU = next(self.cand_list[-1])
        
        if current_max!=self.cand_list[-1].max_volume():
            boxGen = self.cand_list.pop()
            self.cand_list.add(boxGen)
            
        return box(currentL, currentU,
                   self.globalL, self.globalU,
                   boxScalar=self.boxScalar)
    
class boxCandidates():
    def __init__(self, globalL, globalU, boxScalar=None):
        self.boxScalar = boxScalar
        self.globalL = globalL
        self.globalU = globalU
        
        self.cand_list = SortedList(key=lambda x: x.importance)
            
    def append(self, boxGen):
        for currentL, currentU in boxGen:
            box_ = box(currentL, currentU,
                   self.globalL, self.globalU,
                   boxScalar=self.boxScalar)
            self.cand_list.add(box_)
        
    def pop(self):        
        return self.cand_list.pop()

class esse():
    def __init__(self, boxScalar, singleScalar,
                 targetGap=0.0, targetSize=None):
        self.__solutionsList = scalar_interface
        self.__solutionsList = box_interface
        if not isinstance(boxScalar, box_interface) or not isinstance(boxScalar, scalar_interface) or \
            not isinstance(singleScalar, scalar_interface) or not isinstance(singleScalar, single_interface):
            raise ValueError('boxScalar and singleScalar must be a mo_problem implementation.')

        self.__boxScalar = boxScalar
        self.__singleScalar = singleScalar
        self.__targetGap = targetGap
        self.__targetSize = targetSize if targetSize!=None else 100*self.__weightedScalar.M

        self.__lowerBound = 0
        self.__upperBound = 1
        self.__solutionsList = []#solutionsList
        self.__candidatesList = []
        #self.__candidatesImp = []


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

        self.__candidatesList = boxCandidates(self.__globalL, self.__globalU,
                                              boxScalar=self.__boxScalar)

    def select(self):
        """ Selects the next regions to be optimized"""
        bounded_ = True
        while bounded_ and self.__candidatesList !=[]:
            candidate = self.__candidatesList.pop()
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
            self.__lowerBound += (solution.u-solution.c).prod()
            self.__upperBound -= solution.optimum*(solution.c-solution.l).prod()
        gap = (self.upperBound-self.lowerBound)/self.upperBound
        logger.debug('Current state '+str(self.__upperBound)+' '+str(self.__lowerBound)+' '+str(gap))#+' '+str(solution.alpha))


    def __branch(self, solution):
        """ Using a non dominated center and calculates 2**K new regions

        Parameters
        ----------
        cand: box_scalar object
            A box scalarization object already optimized and feasible
        """
        currBoxGen = BoxGenerator(solution.l, solution.u, solution.alpha, self.__globalL, self.__globalU)
        self.__candidatesList.append(currBoxGen)


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

        node = box(self.__globalL, self.__globalU,
                   self.__globalL, self.__globalU,
                   boxScalar=self.__boxScalar)

        while node!=None and \
              (self.upperBound-self.lowerBound)/self.upperBound>self.targetGap and\
              len(self.solutionsList)<self.targetSize:
            solution = node.optimize(solutionsList = self.solutionsList)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.clock() - start
