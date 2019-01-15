"""
Many Objective Noninferior Estimation utils
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
import itertools as it
import copy
import gurobipy as grb

import pulp as lp

def opt_code(code_n):
    code_s=['LOADED',
            'OPTIMAL',
            'INFEASIBLE',
            'INF_OR_UNBD',
            'UNBOUNDED',
            'CUTOFF',
            'ITERATION_LIMIT',
            'NODE_LIMIT',
            'TIME_LIMIT',
            'SOLUTION_LIMIT',
            'INTERRUPTED',
            'NUMERIC',
            'SUBOPTIMAL',
            'INPROGRESS']
    return code_s[code_n-1]


class w_node_heur():
    def __lpHeur(self,w=None,uR=None,eps=0.01):
        if type(w)==type(None) and type(uR)==type(None):
            w = np.random.rand(self.M)
            w = w/w.sum()
        
        if type(w)!=type(None) and type(uR)!=type(None):
            raise('w and uR are not None')
            
        searchw = type(w)==type(None)
        oidx = list(range(self.M))
        
        # Create a gurobi model
        prob = lp.LpProblem("max mean heur",lp.LpMinimize)

        if searchw:
            w = list(lp.LpVariable.dicts('w', oidx, lowBound=0, upBound=1, cat='Continuous').values())
            prob += lp.lpSum([w[i] for i in oidx]) == 1
        else:
            uR = list(lp.LpVariable.dicts('uR', oidx, lowBound=0, upBound=1, cat='Continuous').values())
        v = lp.LpVariable('v', cat='Continuous')

        muBaux = []
        for value,sols in enumerate(self.solutionsList):
            aux   = lp.lpDot(w,self.__norm(sols.objs))\
                    if searchw else w@self.__norm(sols.objs)
            prob += v-aux<=0
            muBaux += [v-aux]

        if not searchw:
            for value,sols in enumerate(self.solutionsList):
                expr  = lp.lpDot(uR, sols.w)
                cons  = self.__norm(sols.objs) @ sols.w
                prob += expr>=cons*(1-eps)

            for i in oidx:
                prob += uR[i]>=self.__norm(self.__globalL)[i]                

        if searchw:
            wuR = lp.lpDot(w,uR)
        else:
            wuR = lp.lpDot(w,uR)

        prob += wuR-v

        try:
            grbs = lp.solvers.GUROBI(timeLimit=10, epgap=0.1, msg=False)
            prob.solve(grbs)
        except:
            prob.solve()
        
        feasible = False if prob.status in [-1, -2, -3] else True
                
        if feasible:
            muB_ = [0 if lp.value(mu)>=0 else 1 for mu in muBaux]
            w_ = np.array([lp.value(w[i]) if lp.value(w[i])>=0 else 0 for i in oidx])\
                 if searchw else w
            uR_ = np.array([lp.value(uR[i]) for i in oidx])\
                  if not searchw else uR
            fobj=lp.value(prob.objective)
        else:
            w_ , uR_, muB_, fobj = [None, None, None, None]
        
        return w_, uR_,muB_ ,fobj

    def __calcHeur(self, eps, free=False):
        w = np.random.rand(self.M);w = w/w.sum()
        bw = w
        bmuB_ = []
        bobj = float('inf')
        stop = False
        for turn in it.cycle(['tic','toc']):
            if turn == 'tic':
                w, uR, muB_, fobj = self.__lpHeur(w,None,eps=eps)
            else:
                w, uR, muB_, fobj = self.__lpHeur(None,uR,eps=eps)
            if type(w)==type(None):
                break
            if abs(1-(bobj+fobj)/(2*bobj))<0.01 or free==False:
                stop = True
            if bobj>fobj:
                bw = w
                bmuB_ = muB_
                bobj=fobj
            if stop:
                break
        
        return bw, bmuB_, bobj

class w_node():
    def __init__(self, solutionsList, globalL, globalU, weightedScalar,goal=float('inf'), time_limit=30):
        self.__weightedScalar = weightedScalar
        self.__M = solutionsList[0].M
        self.__globalL,self.__globalU=globalL, globalU
        self.solutionsList = solutionsList
        self.__goal = goal
        self.__time_limit = time_limit
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

    def optimize(self, oArgs, solutionsList = None):
        self.__solution = copy.copy(self.__weightedScalar)
        try:
            self.__solution.optimize(self.__w, solutionsList)
        except:
            self.__solution.optimize(self.__w)
        return self.__solution        
    
    def __norm(self, objs):
            ret = (objs-self.__globalL)/(self.__globalU-self.__globalL)
            return ret*(ret>10**-13)
    
    def __calcD(self, solT, solList=None):
        
        if solList==None:
            solList=self.solutionsList
        
        vec = [max(self.__norm(solT.objs)-self.__norm(sol.objs)) for sol in solList]
        value = max(vec)
        
        return value,vec

    def __lpHeur(self,w=None,uR=None,eps=0.01):
        if type(w)==type(None) and type(uR)==type(None):
            w = np.random.rand(self.M)
            w = w/w.sum()
            
        oidx = [i for i in range(self.M)]
        
        # Create a gurobi model
        m = grb.Model("Heurs")

        if type(w)==type(None):
            w = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='w')
            m.addConstr(grb.quicksum(w[i] for i in oidx)==1)
            m.update()

        if type(uR)==type(None):
            uR = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='uR')
        
        v = m.addVar(vtype=grb.GRB.CONTINUOUS)

        # Inherent constraints of this problem
        for value,sols in enumerate(self.solutionsList):
            expr=grb.quicksum(uR[i]*sols.w[i] for i in oidx)
            cons=self.__norm(sols.objs) @ sols.w
            m.addConstr(expr>=cons*(1-eps))
        m.update()

        muBaux = []
        debug = []
        for value,sols in enumerate(self.solutionsList):
            expr=v-grb.quicksum(w[i]*self.__norm(sols.objs)[i] for i in oidx)
            cons=0
            m.addConstr(expr<=cons)
            debug +=[sum(w[i]*self.__norm(sols.objs)[i] for i in oidx)]
            muBaux+=[expr]
        m.update()

        try:
            for i in oidx:
                m.addConstr(uR[i]>=self.__norm(self.__globalL)[i])
        except:
            pass

        m.update()

        wuR = grb.quicksum(w[i]*uR[i] for i in oidx)

        m.setObjective(wuR-v)
        m.update()

        
        # Setting parameters for the guroby solver
        m.params.OutputFlag = False
        m.params.TimeLimit = 10
        m.optimize()
        feasible = True
        if m.status in [1,3,4,5]:
            feasible = False
            m.write("model.lp")
            m.computeIIS()
            m.write("constraints.ilp")
        pass
                
        if feasible:
            muB_ = [0 if mu.getValue()>=0 else 1 for mu in muBaux]
            try:
                w_ = np.array([w[i].x if w[i].x>=0 else 0 for i in oidx])
            except:
                w_ = w
            try:
                uR_ = np.array([uR[i].x for i in oidx])
            except:
                uR_ = uR
            fobj=m.ObjVal
        else:
            w_ = None
            uR_ = None
            muB_ = None
            fobj = None
        
        return w_, uR_,muB_ ,fobj

    def __calcHeur(self, eps, free=False):
        w = np.random.rand(self.M);w = w/w.sum()
        bw = w
        bmuB_ = []
        bobj = float('inf')
        stop = False
        for turn in it.cycle(['tic','toc']):
            if turn == 'tic':
                w, uR, muB_, fobj = self.__lpHeur(w,None,eps=eps)
            else:
                w, uR, muB_, fobj = self.__lpHeur(None,uR,eps=eps)
            if type(w)==type(None):
                break
            if abs(1-(bobj+fobj)/(2*bobj))<0.01 or free==False:
                stop = True
            if bobj>fobj:
                bw = w
                bmuB_ = muB_
                bobj=fobj
            if stop:
                break
        
        return bw, bmuB_, bobj


    def __calcW(self, goal=1, eps=0.01):
        sw, smuB, sobj = self.__calcHeur(eps)
        #sobj = float('inf')
        
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        
        # Create a gurobi model
        m = grb.Model("MONISE")
        
        # Creation of linear integer variables
        w = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='w')
        uR = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='uR')
        mu = m.addVars(list(range(Nsols)),vtype=grb.GRB.CONTINUOUS,name='mu')
        nu = m.addVars(oidx,vtype=grb.GRB.CONTINUOUS,name='nu')
        
        test = False
        if test:
            muB = m.addVars(list(range(Nsols)),lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='muB')
            nuB = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.CONTINUOUS,name='nuB')
        else:
            muB = m.addVars(list(range(Nsols)),lb=0,ub=1,vtype=grb.GRB.BINARY,name='muB')
            if sobj!=float('inf'):
                for i in range(Nsols):
                    muB[i].start = smuB[i]
            nuB = m.addVars(oidx,lb=0,ub=1,vtype=grb.GRB.BINARY,name='nuB')
            if sobj!=float('inf'):
                for i in oidx:
                    nuB[i].start = 0 if sw[i]==0 else 1
        
        v = m.addVar(vtype=grb.GRB.CONTINUOUS)
        xi = m.addVar(vtype=grb.GRB.CONTINUOUS)
        
        # Inherent constraints of this problem
        for value,sols in enumerate(self.solutionsList):
            expr=grb.quicksum(uR[i]*sols.w[i] for i in oidx)
            cons=self.__norm(sols.objs) @ sols.w
            m.addConstr(expr>=cons*(1-eps))

        m.update()
        
        for i in oidx:
            expr = uR[i]-grb.quicksum(mu[conN]*self.__norm(sols.objs)[i] for conN,sols in enumerate(self.solutionsList))-nu[i]+xi
            m.addConstr(expr==0)

        

        for conN,sols in enumerate(self.solutionsList):
            d, dvec = self.__calcD(sols)
            expr=v-grb.quicksum(w[i]*self.__norm(sols.objs)[i] for i in oidx)
            m.addConstr(expr<=0)
            m.addConstr(expr>=-muB[conN])
            
            m.addConstr(mu[conN]>=0)
            m.addConstr(mu[conN]<=(1-muB[conN]))
        m.update()

        for i in oidx:
            m.addConstr(uR[i]>=self.__norm(self.__globalL)[i])

        m.update()

        for i in oidx:
            m.addConstr(w[i]>=0)
            m.addConstr(w[i]<=nuB[i])
            
            m.addConstr(nu[i]>=0)
            m.addConstr(nu[i]<=(1-nuB[i]))

        m.update()

        m.addConstr(grb.quicksum(w[i] for i in oidx)==1)
        m.update()
        
        m.addConstr(grb.quicksum(mu[i] for i in range(Nsols))==1)
        m.update()

        m.setObjective(-xi)
        m.update()

        # Setting parameters for the guroby solver
        m.params.OutputFlag = False
        m.params.MIPGapAbs = 0.01
        MAXINT = m.params.SolutionLimit
        m.params.SolutionLimit = 1
        m.optimize()
        
        if self.__goal!=float('inf'):
            m.params.BestObjStop = -self.__goal
            pass
        m.params.TimeLimit = self.__time_limit
        m.params.SolutionLimit = MAXINT
        m.optimize()
        
        feasible = True
        if m.status==3:
            feasible = False
            m.computeIIS()
            m.write("constraints.ilp")
        pass

        #input()

        if feasible:
            w_ = np.array([w[i].x if w[i].x>=0 else 0 for i in oidx])
            w_ = w_/(self.__globalU-self.__globalL)
            w_ = w_/w_.sum()
            w_ = w_+(w_>0)*(w_<=10**-12)*10**-12
            fobj=m.ObjVal#m.ObjBound
            print(fobj, sobj)
            self.__w = np.array(w_)
            self.__importance = -fobj
        else:
            raise('Somethig wrong')

import time

if __name__ == '__main__':
    start = time.clock()
    class mootest():
        def __init__(self,w,objs):
            self.objs = objs
            self.w = w
            self.M = objs.size
    solutionsList = []
    for w in [[0.98,0.01,0.01],[0.01,0.98,0.01],[0.01,0.01,0.98],[0.33,0.33,0.34],[0.66,0.33,0.01], 
              [0.01,0.33,0.66],[0.33,0.01,0.66],[0.01,0.66,0.33],[0.2,0.5,0.3]]:
        w = np.array(w)
        alpha = -np.prod(w)/sum(a*b for a,b in it.combinations(w,2))
        x = -alpha/w
        objs =x**2
        solutionsList+=[mootest(w,objs)]
    wclass = w_node(solutionsList, np.array([0,0,0]), np.array([1,1,1]),None)
    print(time.clock()-start)
    print(wclass.w,wclass.importance)
