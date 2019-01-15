from abc import ABCMeta, abstractmethod
"""
Multiobjective interface
"""
"""
Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause


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

    def dominated(self, solutionsList):
        """ Finds if a box_scalar object is dominated
            (if the inferior limit is dominated)
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
