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

import pulp as lp
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

    @property
    def parents(self):
        return self.__parents

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
        prob = lp.LpProblem("max_mean", lp.LpMaximize)
        # prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 

        # Creation of linear integer variables
        w = list(lp.LpVariable.dicts('w', oidx, lowBound=0, upBound=1,
                                     cat='Continuous').values())
        # w = list(prob.add_var())
        
        uR = self.__globalL

        v = lp.LpVariable('v', cat='Continuous')
        # v = prob.add_var(name='v', var_type=mip.CONTINUOUS) 

        for conN, sols in enumerate(self.solutionsList):
            d, dvec = self.__calcD(sols)
            expr = v-lp.lpDot(w, self.__normf(sols.objs))
            # expr = v-mip.xsum(w[i]*self.__normf(sols.objs)[i] for i in range())
            prob += expr <= 0 #manter

        for i in oidx:
            prob += w[i] >= 0 #manter

        prob += lp.lpSum([w[i] for i in oidx]) == 1
        # prob += mip.xsum([w[i] for i in oidx]) == 1
        prob += v-lp.lpDot(w, self.__normf(uR))
        # prob += v-mip.xsum(w*self.__normf(uR)[i] for i in range())

        try:
            grbs = lp.GUROBI(msg=False, OutputFlag=False)
            prob.solve(grbs)
        except:
            prob.solve()
            
        # status = prob.optimize()

        feasible = False if prob.status in [-1, -2] else True
        # feasible = status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE

        if feasible:
            w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
                           for i in oidx])
            # w_ = np.array([w[i].x if w[i].x >= 0 else 0 for i in oidx])
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            fobj = lp.value(prob.objective)
            # fobj = prob.objective_values[0]
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            raise('Somethig wrong')

    def __calcW(self, goal=1, eps=0.00):
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        # Create a gurobi model
        prob = lp.LpProblem("max_mean", lp.LpMaximize)
        # prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 

        # Creation of linear integer variables
        w = list(lp.LpVariable.dicts('w', oidx, lowBound=0,
                                     cat='Continuous').values())
        # w = list(prob.add_var(name='w', var_type=mip.CONTINUOUS, lb=0) for i in range(len(Ingredients)))
        uR = list(lp.LpVariable.dicts('uR', oidx, cat='Continuous').values())
        # uR = list()
        kp = list(lp.LpVariable.dicts('kp', list(range(Nsols)),
                                      cat='Continuous').values())
        # kp = list()
        nu = list(lp.LpVariable.dicts('nu', oidx, lowBound=0,
                                      cat='Continuous').values())
        # nu = list()

        test = False
        if test:
            kpB = list(lp.LpVariable.dicts('kpB', list(range(Nsols)),
                                           lowBound=0, upBound=1,
                                           cat='Continuous').values())
            nuB = list(lp.LpVariable.dicts('nuB', oidx, lowBound=0, upBound=1,
                                           cat='Continuous').values())
        else:
            kpB = list(lp.LpVariable.dicts('kpB', list(range(Nsols)),
                                           cat='Binary').values())
            nuB = list(lp.LpVariable.dicts('nuB', oidx, cat='Binary').values())

        v = lp.LpVariable('v', cat='Continuous')
        mu = lp.LpVariable('mu', cat='Continuous')

        # Inherent constraints of this problem
        for value, sols in enumerate(self.solutionsList):
            expr = lp.lpDot(self.__normw(sols.w), uR)
            cons = self.__normf(sols.objs) @ self.__normw(sols.w)
            prob += expr >= cons*(1-eps)

        for i in oidx:
            expr = uR[i]-lp.lpSum([kp[conN]*self.__normf(sols.objs)[i]
                                   for conN, sols in
                                   enumerate(self.solutionsList)])-nu[i]+mu
            prob += expr == 0

        bigC = max(self.__normf(self.__globalU))

        for conN, sols in enumerate(self.solutionsList):
            # d, dvec = self.__calcD(sols)
            expr = v-lp.lpDot(w, self.__normf(sols.objs))
            prob += -expr >= 0 #mantem
            prob += -expr <= kpB[conN]*bigC #mantem
            prob += kp[conN] >= 0 #mantem
            prob += kp[conN] <= (1-kpB[conN]) #mantem

        for i in oidx:
            prob += uR[i] >= self.__normf(self.__globalL)[i]

        for i in oidx:
            prob += w[i] >= 0 #mantem
            prob += w[i] <= nuB[i] #mantem
            prob += nu[i] >= 0 #mantem
            prob += nu[i] <= (1-nuB[i])*2*bigC #mantem

        prob += lp.lpSum([w[i] for i in oidx]) == 1
        prob += lp.lpSum([kp[i] for i in range(Nsols)]) == 1

        prob += mu

        # desigualdades válidas
        prob += mu <= v

        try:
            rnd = np.array(sorted([0] +
                                  [np.random.rand() for i in range(self.M-1)]
                                  + [1]))
            w_ini = np.array([rnd[i+1]-rnd[i] for i in range(self.__M)])
            w_ini = w_ini/w_ini.sum()
            for wi, wii in zip(w, w_ini):
                wi.start = wii
            grbs = lp.GUROBI(epgap=self.__mip_gap, SolutionLimit=1,
                                     msg=False, OutputFlag=False, Threads=1)
            prob.solve(grbs)

            if self.__goal != float('inf'):
                grbs = lp.GUROBI(timeLimit=self.__time_limit,
                                         epgap=self.__mip_gap,
                                         SolutionLimit=MAXINT,
                                         msg=False, BestObjStop=self.__goal,
                                         OutputFlag=False, Threads=1)
            else:
                grbs = lp.GUROBI(timeLimit=self.__time_limit,
                                         epgap=self.__mip_gap,
                                         SolutionLimit=MAXINT,
                                         msg=False, OutputFlag=False,
                                         Threads=1)
            prob.solve(grbs)
        except:
            cbcs = lp.COIN_CMD(maxSeconds=self.__time_limit,
                                           fracGap=self.__mip_gap, threads=1)
            prob.solve(cbcs, use_mps=False)

        feasible = False if prob.status in [-1, -2] else True

        if feasible:
            w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
                           for i in oidx])
            w_ = w_/w_.sum()
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            fobj = lp.value(prob.objective)
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            #raise('Non-feasible solution')
            self.__w = w_ini
            self.__importance = 0
            
