# -*- coding: utf-8 -*-
"""
Many Objective Noninferior Estimation utils

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

import mip


MAXINT = 2000000000


class weight_solv():
    def __init__(self, solutionsList, globalL, globalU, weightedScalar,
                 goal=float('inf'), time_limit=10, mip_gap=0.01, norm=False):
        self.__weightedScalar = weightedScalar
        self.__M = solutionsList[0].M
        self.__globalL, self.__globalU = globalL, globalU
        self.solutionsList = solutionsList
        self.__goal = goal
        self.__time_limit = time_limit
        self.__mip_gap = mip_gap
        self.__norm = norm
        if len(self.solutionsList) == self.M:
            self.__calcFirstW()
        else:
            self.__calcW(goal=goal)

    @property
    def M(self):
        return self.__M

    @property
    def importance(self):
        return self.__importance

    #@property
    #def parents(self):
    #    return self.__parents

    @property
    def solution(self):
        return self.__solution

    @property
    def w(self):
        return self.__w

    def optimize(self, hotstart=None):
        self.__solution = copy.copy(self.__weightedScalar)
        try:
            self.__solution.optimize(self.w, [])
        except:
            self.__solution.optimize(self.w)
        return self.__solution

    def __normf(self, obj):
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w):
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/w_.sum()
        else:
            return w

    def __calcD(self, solT, solList=None):

        if solList is None:
            solList = self.solutionsList

        vec = [max(self.__normf(solT.objs)-self.__normf(sol.objs))
               for sol in solList]
        value = max(vec)

        return value, vec

    def __calcFirstW(self, goal=1, eps=0.00):
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        assert self.M == Nsols, 'only fist W'

        # Create a gurobi model
        #prob = lp.LpProblem("max mean", lp.LpMaximize)
        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except:
            prob = mip.Model(sense=mip.MAXIMIZE)

        prob.verbose = 0

        # Creation of linear integer variables
        #w = list(lp.LpVariable.dicts('w', oidx, lowBound=0, upBound=1,
        #                             cat='Continuous').values())
        w = [prob.add_var(name='w', var_type=mip.CONTINUOUS, lb=0, ub=1) for i in oidx]
        
        uR = self.__globalL

        #v = lp.LpVariable('v', cat='Continuous')
        v = prob.add_var(name='v', var_type=mip.CONTINUOUS) 

        for conN, sols in enumerate(self.solutionsList):
            d, dvec = self.__calcD(sols)
            #expr = v-lp.lpDot(w, self.__normf(sols.objs))
            expr = v-mip.xsum(w[i]*self.__normf(sols.objs)[i] for i in oidx)
            prob += expr <= 0 #manter

        for i in oidx:
            prob += w[i] >= 0 #manter

        #prob += lp.lpSum([w[i] for i in oidx]) == 1
        prob += mip.xsum(w[i] for i in oidx) == 1
        #prob += v-lp.lpDot(w, self.__normf(uR))
        prob += v-mip.xsum(w[i]*self.__normf(uR)[i] for i in oidx)

        #try:
        #    grbs = lp.GUROBI(msg=False, OutputFlag=False)
        #    prob.solve(grbs)
        #except:
        #prob.solve()
            
        status = prob.optimize()

        #feasible = False if prob.status in [-1, -2] else True
        feasible = status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE

        if feasible:
            #w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
            #               for i in oidx])
            w_ = np.array([w[i].x if w[i].x >= 0 else 0 for i in oidx])
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            #fobj = lp.value(prob.objective)
            fobj = prob.objective_value
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            raise('Somethig wrong')

    def __calcW(self, goal=1, eps=0.00):
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        # Create a gurobi model
        #prob = lp.LpProblem("max mean", lp.LpMaximize)
        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except:
            prob = mip.Model(sense=mip.MAXIMIZE)
            
        prob.verbose = 0

        # Creation of linear integer variables
        w =  [ prob.add_var(name='w', lb=0, var_type=mip.CONTINUOUS) for i in oidx ]
        uR = [ prob.add_var(name='uR', lb=-np.inf, var_type=mip.CONTINUOUS) for i in oidx ]
        kp = [ prob.add_var(name='kp', lb=0, var_type=mip.CONTINUOUS) for i in range(Nsols) ]
        nu = [ prob.add_var(name='nu', lb=0, var_type=mip.CONTINUOUS) for i in oidx ]

        kpB = [ prob.add_var(name='kpB', var_type=mip.BINARY) for i in range(Nsols) ]
        nuB = [ prob.add_var(name='nuB', var_type=mip.BINARY) for i in oidx ]

        v = prob.add_var(name='v', lb=0, var_type=mip.CONTINUOUS)
        mu = prob.add_var(name='mu', lb=-np.inf, var_type=mip.CONTINUOUS)

        # Inherent constraints of this problem
        for value, sols in enumerate(self.solutionsList):
            expr = mip.xsum(self.__normw(sols.w)[i]*uR[i] for i in oidx)
            cons = self.__normw(sols.w) @ self.__normf(sols.objs)
            prob += expr >= cons*(1-eps)

        for i in oidx:
            expr = uR[i]-mip.xsum([kp[conN]*self.__normf(sols.objs)[i]
                                   for conN, sols in
                                   enumerate(self.solutionsList)])-nu[i]+mu
            prob += expr == 0

        for i in oidx:
            prob += uR[i] >= self.__normf(self.__globalL)[i] #mantem

        bigC = max(self.__normf(self.__globalU))

        for conN, sols in enumerate(self.solutionsList):
            expr = mip.xsum(w[i]*self.__normf(sols.objs)[i] for i in oidx)-v
            prob += expr >= 0 #mantem
            prob += expr <= kpB[conN]*bigC #mantem
            prob += kp[conN] >= 0 #mantem
            prob += kp[conN] <= (1-kpB[conN]) #mantem

        prob += mip.xsum(w[i] for i in oidx) == 1
        prob += mip.xsum(kp[i] for i in range(Nsols)) == 1

        for i in oidx:
            prob += w[i] >= 0 #mantem
            prob += w[i] <= nuB[i] #mantem
            prob += nu[i] >= 0 #mantem
            prob += nu[i] <= (1-nuB[i])*2*bigC #mantem
        
        prob += mip.xsum([mu])

        # desigualdades válidas
        prob += mu <= v

        rnd = np.array(sorted([0] +
                              [np.random.rand() for i in range(self.M-1)]
                              + [1]))
        w_ini = np.array([rnd[i+1]-rnd[i] for i in range(self.__M)])
        w_ini = w_ini/w_ini.sum()
        prob.start = [(wi, wii) for wi, wii in zip(w, w_ini)]

        #grbs = lp.GUROBI(epgap=self.__mip_gap, SolutionLimit=1,
        #                         msg=False, OutputFlag=False, Threads=1)

        prob.max_gap = self.__mip_gap
        prob.threads = 1
        prob.max_solutions = MAXINT
        prob.max_seconds = self.__time_limit

        #if self.__goal != float('inf'):
            #grbs = lp.GUROBI(timeLimit=self.__time_limit,
            #                         epgap=self.__mip_gap,
            #                         SolutionLimit=MAXINT,
            #                         msg=False, BestObjStop=self.__goal,
            #                         OutputFlag=False, Threads=1)


        #else:
            #grbs = lp.GUROBI(timeLimit=self.__time_limit,
            #                         epgap=self.__mip_gap,
            #                         SolutionLimit=MAXINT,
            #                         msg=False, OutputFlag=False,
            #                         Threads=1)

        status = prob.optimize()#max_solutions=None
            

        #feasible = False if prob.status in [-1, -2] else True
        feasible = status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE

        if feasible:
            #w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
            #               for i in oidx])
            w_ = np.array([w[i].x if w[i].x >= 0 else 0 for i in oidx])
            w_ = w_/w_.sum()
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            #fobj = lp.value(prob.objective)
            fobj = prob.objective_value
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            raise('Non-feasible solution')
