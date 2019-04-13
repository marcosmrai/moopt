# -*- coding: utf-8 -*-
"""
Scalarization Interface

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod


class scalar_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def M(self):
        pass

    @property
    @abstractmethod
    def feasible(self):
        pass

    @property
    @abstractmethod
    def optimum(self):
        pass

    @property
    @abstractmethod
    def objs(self):
        pass

    @property
    @abstractmethod
    def x(self):
        pass

    @abstractmethod
    def optimize(self, *args):
        pass


class w_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def w(self):
        pass


class single_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def w(self):
        pass


class box_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def u(self):
        pass

    @property
    @abstractmethod
    def l(self):
        pass

    @property
    @abstractmethod
    def c(self):
        pass
