from abc import ABCMeta, abstractmethod
import logging
'''
try:
    from .mo_utils import mo_show
except:
    import sys
    sys.path.append('.')
    from mo_utils import mo_show
'''
from .mo_utils import mo_show


logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class node_interface(metaclass=ABCMeta):
    @abstractmethod
    def optimize(self):
        """Finds the argument of the optimizer"""
        pass

    @property
    @abstractmethod
    def solution(self):
        """Finds the argument of the optimizer"""
        pass

    @property
    @abstractmethod
    def importance(self):
        """Finds the argument of the optimizer"""
        pass

    def dominated(self,solutionsList):
        """ Finds if a box_scalar object is dominated (if the inferior limit is dominated)
            by any solution already optimized

        Parameters
        ----------
        cand: box_scalar object
            A box scalarization object already optimized and feasible
        Returns
        -------
        dom_: bool
            If is dominated or not
        """
        for solution in solutionsList:
            if (self.l>=solution.objs).all():
                return True
        return False



class bb_interface(mo_show, metaclass=ABCMeta):
    @property
    @abstractmethod
    def gap(self):
        pass


    @property
    @abstractmethod
    def upperBound(self):
        pass

    @property
    @abstractmethod
    def lowerBound(self):
        pass

    @property
    @abstractmethod
    def solutionsList(self):
        pass

    @abstractmethod
    def inicialization(self, *args, **kwargs):
        """Inicialiate the mo optimizer"""
        pass

    @abstractmethod
    def select(self, *args, **kwargs):
        """From a list of potencial solution, choose one to optimize"""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update gobal variables"""
        pass

    @property
    def fit_runtime(self):
        return self.__fit_runtime
    '''
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
    '''