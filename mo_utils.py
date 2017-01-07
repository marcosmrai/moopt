import logging

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def dominated(objs,solutionList):
    for sol in solutionList:
        if (sol.objs<=objs).all():
            return True
    return False

# starting here, must be refactorated
'''
try:
    from .hv import _HyperVolume
except:
    import sys
    sys.path.append('.')
    from hv import _HyperVolume
'''
from .hv import _HyperVolume

class mo_metrics():
    def hipervolume(self, globalU, globalL):
        """Calculates the hipervolume of a set of solutions

        Parameters
        ----------

        Returns
        -------
        """
        hv = _HyperVolume((globalU-globalL)/(globalU-globalL))
        return hv.compute([(solution.objs-globalL)/(globalU-globalL) for solution in self.solutionsList])

    def inverse_hipervolume(self, globalU, globalL):
        """Calculates the hipervolume of a set of solutions

        Parameters
        ----------

        Returns
        -------
        """
        hv = _HyperVolume((globalL-globalU)/(globalL-globalU))
        return hv.compute([(solution.objs-globalU)/(globalL-globalU) for solution in self.solutionsList])


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class mo_show():
    def print3D(self,path=None):
        """Make a 3D figure of the multiobjective solutions

        Parameters
        ----------

        Returns
        -------
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        x = [solution.objs[0] for solution in self.solutionsList[:3]]
        y = [solution.objs[1] for solution in self.solutionsList[:3]]
        z = [solution.objs[2] for solution in self.solutionsList[:3]]
        # put 0s on the y-axis, and put the y axis on the z-axis
        ax.plot(xs=x, ys=y, zs=z, linestyle=' ', marker='o', color='r')

        x = [solution.objs[0] for solution in self.solutionsList[3:]]
        y = [solution.objs[1] for solution in self.solutionsList[3:]]
        z = [solution.objs[2] for solution in self.solutionsList[3:]]
        plt.hold(False)
        ax.plot(xs=x, ys=y, zs=z, linestyle=' ', marker='o', color='b')

        if path==None:
            plt.show()
        else:
            plt.savefig(path)


    def print2D(self,path=None):
        """Make a 3D figure of the multiobjective solutions

        Parameters
        ----------

        Returns
        -------
        """

        #fig,ax = plt.subplots()

        plt.figure()
        plt.hold(True)
        x = [solution.objs[0] for solution in self.solutionsList[:2]]
        y = [solution.objs[1] for solution in self.solutionsList[:2]]
        # put 0s on the y-axis, and put the y axis on the z-axis
        plt.plot(x, y, linestyle=' ', marker='o', color='r')


        #plt.plot(x, y, linestyle=' ', marker='o', color='r')

        x = [solution.objs[0] for solution in self.solutionsList[2:]]
        y = [solution.objs[1] for solution in self.solutionsList[2:]]

        #plt.plot(x, y, linestyle=' ', marker='o', color='b')
        plt.plot(x, y, linestyle=' ', marker='o', color='b')

        plt.grid(True)
        plt.ylabel('Regularization Cost')
        plt.xlabel('Error')

        if path==None:
            plt.show()
        else:
            plt.savefig(path)
        plt.hold(False)
        plt.close('all')
