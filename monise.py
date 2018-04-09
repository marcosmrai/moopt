"""
Many Objective Noninferior Estimation
"""
"""
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
    def __init__(self, weightedScalar, singleScalar, target_gap=0.0, 
                 target_size=None, red_fact=float('inf'), smoth_count=None, 
                 node_time_limit=float('inf'), node_gap = 0.01, hotstart=[], norm=True):
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if not isinstance(weightedScalar, scalar_interface) or not isinstance(weightedScalar, w_interface) or \
            not  isinstance(singleScalar, scalar_interface) or not  isinstance(singleScalar, single_interface):
            raise ValueError('weightedScalar'+' and '+'singleScalar'+'must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__target_gap = target_gap
        self.__target_size = target_size if target_size!=None else 100*self.__weightedScalar.M
        self.__node_time_limit = node_time_limit
        self.__node_gap = node_gap
        self.__red_fact = red_fact
        self.__norm = norm
        if smoth_count==None:
            self.__smoth_count = 1 if node_time_limit==float('inf') else 5
        else:
            self.__smoth_count = smoth_count

        self.__maxImp = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []

    def __del__(self):
        del self.__solutionsList
        
    @property
    def target_size(self): return self.__target_size

    @property
    def target_gap(self): return self.__target_gap

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def hotstart(self): return self.__hotstart+self.solutionsList
    
    @property
    def currImp(self):
        return max(self.__importances[-self.__smoth_count:])

    @property
    def maxImp(self): return self.__maxImp
    
    @property
    def importances(self): return self.__importances

    def inicialization(self):
        self.__M = self.__singleScalar.M
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
            self.__solutionsList.append(singleS)
            parents.append(singleS)


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
        logger.info('Node nbr '+str(len(self.solutionsList))+' - importances - '
                    +str(self.__maxImp)+' '+str(self.currImp)+' '+str(self.__goal)
                    +' '+str(node.importance)+' '+str(gap))
        
    def _next(self):
        next_wsol = weight_solv(self.solutionsList, self.__globalL, self.__globalU, 
                                self.__weightedScalar, goal = self.currImp*self.__red_fact, 
                                time_limit = self.__node_time_limit, mip_gap = self.__node_gap, 
                                norm=self.__norm)
        self.__importances+=[next_wsol.importance]
        return next_wsol

    def optimize(self):
        start = time.clock()
        next_wsol = self.inicialization()

        while self.currImp/self.__maxImp>self.__target_gap and \
              len(self.solutionsList)<self.__target_size:
                  
            solution = next_wsol.optimize(hotstart=self.hotstart)
            self.update(next_wsol, solution)
            next_wsol = self._next()
            
        self.__fit_runtime = time.clock() - start
