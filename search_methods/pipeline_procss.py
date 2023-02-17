# coding:utf-8
import torch
import torch.multiprocessing 


class MyThread(torch.multiprocessing.Process):
    def __init__(self, func,args1,args2,args3,args4,args5,args6,args7,args8,args9):

        torch.multiprocessing.Process.__init__(self)
        self.func = func
        self.args1 = args1
        self.args2 = args2
        self.args3 = args3
        self.args4 = args4
        self.args5 = args5
        self.args6 = args6
        self.args7 = args7
        self.args8 = args8
        self.args9 = args9


        self.result = self.func(self.args1,self.args2,self.args3,self.args4,self.args5,self.args6,self.args7,self.args8,self.args9)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
