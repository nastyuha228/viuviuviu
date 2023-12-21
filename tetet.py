import math
import random

        
       
    
    
class biba():
    def __init__(self, i, h, o, lr):
        self.input_nodes = i
        self.hidden_nodes = h
        self.output_nodes = o
        self.lr = lr
        self.ab = [[0]*self.input_nodes for i in range(self.hidden_nodes)]
        self.ba = [[0]*self.hidden_nodes for i in range(self.output_nodes)]



        
        for i in range(self.hidden_nodes):
            for j in range(self.output_nodes):
                self.ab[i][j] = random.random()
        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                self.ba[i][j] = random.random()
  
    def query(self, inputs_lists):
        O_h = self.f_activate(self.mult_matrix(self.ab, inputs_lists))
        
        O = self.f_activate(self.mult_matrix(self.ba, O_h))
        return O


    def train(self, inputs_lists, targets_list, epochs):
        for epoch in range(epochs):
            O_h = self.f_activate(self.mult_matrix(self.ab, inputs_lists))
            O = self.f_activate(self.mult_matrix(self.ba, O_h))
            E_o = self.sub_matrix(targets_list, O)
            E_h = self.mult_matrix(self.transpose(self.ba), E_o)

            dW_h_o = [[0]*self.input_nodes for i in range(self.hidden_nodes)]
            dW_h_o = self.mult_const(self.mult_matrix(self.proizv(self.proizv(E_o, O), self.sub_from_const(O, 1)), self.transpose(O_h)), self.lr)
            self.ba = self.summ_matrix(self.ba, dW_h_o)

            dW_i_h = [[0]*self.input_nodes for i in range(self.hidden_nodes)]
            dW_i_h = self.mult_const(self.mult_matrix(self.proizv(self.proizv(E_h, O_h), self.sub_from_const(O_h, 1)), self.transpose(inputs_lists)), self.lr)
            self.ab = self.summ_matrix(self.ab, dW_i_h)

            print(f"\rЭпоха {epoch + 1} из {epochs} выполнена.", end='')

        print()

    
        
    def sigma(self, x):
        return 1 / (1 + math.exp(-x))
    
    def f_activate(self, x):
        for i in range(len(x)):
            x[i][0] = self.sigma(x[i][0])
        return x
    
    def mult_matrix(self, a, b):
        a_r, a_c = len(a), len(a[0])
        b_r, b_c = len(b), len(b[0])
        # print (f'a_r = {a_r}')
        # print (f'a_c = {a_c}')
        # print (f'b_r = {b_r}')
        # print (f'b_c = {b_c}')

        if a_c != b_r:
            # Обработка случая, когда размерности матриц несовместимы
            print ("Размерности матриц несовместимы для умножения")
            return

        c = [[0 for row in range(b_c)] for col in range(a_r)]
        for i in range(a_r):
            for j in range(b_c):
                for k in range(a_c):
                    c[i][j] += a[i][k] * b[k][j]

        return c

    def transpose(self, A):
        M = len(A)
        N = len(A[0])
        A_t = [[0]*M for i in range(N)]
        for i in range(N):
            for j in range(M):
                A_t[i][j] = A[j][i]
        return A_t
    
    
    def mult_const(self,A,k):
        M = len(A)
        N = len(A[0])
        C = [[1]*N for i in range (M)]
        for i in range (M):
            for j in range (N):
                C[i][j] = A[i][j] * k
        return C
    
    def proizv(self,A,B):
        M = len (A)
        N = len (A[0])
        C = [[0]*N for i in range(M)]
        for i in range (M):
            for j in range (N):
                C[i][j] = A[i][j] * B[i][j]
        return C
        
    def sub_from_const(self,A,k):
        M = len(A)
        N = len (A[0])
        C = [[0]*N for i in range(M)]
        for i in range (M):
            for j in range (N):
                C[i][j] = k - A[i][j]
        return C


    def sub_matrix(self,A,B):
        M = len(A)
        N = len (A[0])
        C = [[0]*N for i in range(M)]
        for i in range (M):
            for j in range (N):
                C[i][j] = A[i][j] - B[i][j]
        return C


    def summ_matrix(self,A,B):
        M = len(A)
        N = len (A[0])
        C = [[0]*N for i in range(M)]
        for i in range (M):
            for j in range (N):
                C[i][j] = A[i][j] + B[i][j]
        return C