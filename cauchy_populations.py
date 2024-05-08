import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
import time
warnings.filterwarnings("ignore")

class CauchySolving:
    def __init__(self, case):
        self.set_initial(case[:-1])
        self.shift_status(case[-1])
    
    def derivatives_vector(self, w):
        F = np.zeros(4)
        F[0] = w[0]*(2*w[2]-0.5*w[0]-w[2]**2*w[3]**(-2)*w[1])
        F[1] = w[1]*(2*w[3]-w[2]**(-2)*w[3]**(-2)*w[0]-0.5*w[1])
        F[2] = self.eps_const*(2-2*w[2]*w[3]**(-2)*w[1])
        F[3] = self.eps_const*(2-2*w[3]*w[2]**(-2)*w[1])
        return F
    
    def set_initial(self, initial):
        self.initial = np.array([initial])[0]
        self.reset()
    
    def reset(self):
        self.data = np.array([self.initial[:-1]])
        self.timeline = np.array([0.0])
        self.eps_const = self.initial[-1]

    def current_vector(self):
        return self.data[-1, :]
    
    def update_data(self, update_vector):
        self.data = np.append(self.data, [update_vector], axis=0)
    
    def shift_status(self, value=None):
        self.shift = value

    def dynamic_shift(self, bottom=None, top=None):
        w = self.current_vector()
        if bottom is None: bottom = 0.0001
        if top is None: top = 2000 - self.timeline[-1]
        try:
            JacDerMat = np.zeros((4, 4))
            JacDerMat[0,0] = -w[2]**2*w[1]*w[3]**(-2)+2*w[2]-w[0]
            JacDerMat[0,1] = -w[2]**2*w[0]*w[3]**(-2)
            JacDerMat[0,2] = w[0]*(-2*w[2]*w[1]*w[3]**(-2)+2)
            JacDerMat[0,3] = 2*w[2]**2*w[0]*w[1]*w[3]**(-3)
            JacDerMat[1,0] = -w[1]*w[2]**(-2)*w[3]**(-2)
            JacDerMat[1,1] = 2*w[3]-w[1]-w[0]*w[2]**(-2)*w[3]**(-2)
            JacDerMat[1,2] = 2*w[0]*w[1]*w[2]**(-3)*w[3]**(-2)
            JacDerMat[1,3] = w[1]*(2+2*w[0]*w[2]**(-2)*w[3]**(-3))
            JacDerMat[2,0] = 0
            JacDerMat[2,1] = -2*w[2]*self.eps_const*w[3]**(-2)
            JacDerMat[2,2] = -2*self.eps_const*w[1]*w[2]**(-2)
            JacDerMat[2,3] = 4*w[2]*self.eps_const*w[1]*w[3]**(-3)
            JacDerMat[3,0] = 0
            JacDerMat[3,1] = -2*w[3]*self.eps_const*w[2]**(-2)
            JacDerMat[3,2] = 4*w[3]*self.eps_const*w[1]*w[2]**(-3)
            JacDerMat[3,3] = -2*self.eps_const*w[1]*w[2]**(-2)
            eigenvalues = np.linalg.eig(JacDerMat).eigenvalues.real
            eig_max = eigenvalues.max(None)
            eig_mix = eigenvalues.min(None)
            eig_mx = abs(np.where(-eig_mix > eig_max, eig_mix, eig_max))
            tau = 1/eig_mx
        except:
            tau = bottom
        if tau < bottom: return bottom
        if tau > top: return top
        return tau
    
    def draw_one_plot(self, leader, driven):
        buf_dct = {"x": 0, "y": 1, "a1": 2, "a2": 3}
        if leader == "t":
            x = list(self.timeline)
        else:
            x = list(self.data[:, buf_dct[leader]])
        if driven == "t":
            y = list(self.timeline)
        else:
            y = list(self.data[:, buf_dct[driven]])
        ax = plt.axes()
        ax.set_facecolor("#0A0A0A")
        plt.title(f'Dependence of {leader} on {driven}', weight='bold')
        plt.xlabel(f'{leader}-axis', weight='bold')
        plt.ylabel(f'{driven}-axis', weight='bold')
        plt.plot(x, y, color="#FF6E4A", linewidth=4)
        x1 = [x[i] for i in range(0, len(x), int(len(x)/5))]
        y1 = [y[i] for i in range(0, len(y), int(len(y)/5))]
        plt.plot(x1, y1, 'D', markersize=7, color="#FF6E4A")
        plt.grid(which='major', linewidth=0.3, color="white")
        plt.savefig(f'last_run_results/Dependence of {leader} on {driven}.png') 
        plt.show()
        
    def iter_explicit_Euler(self):
        vector = self.current_vector()
        shift = self.shift
        if shift is None: shift = self.dynamic_shift()
        next_vector = vector + shift*self.derivatives_vector(vector)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)
        
    def method_explicit_Euler(self, iterations_num=None):
        self.reset()
        start = time.time()
        self.iter_implicit_Euler(10**(-5))
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2000:
                self.iter_explicit_Euler()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Euler()
                if self.timeline[-1] == 2000:
                    iterations_num = i + 1
                    break
        end = time.time()
        with open('last_run_results/about_last_run.txt', 'w') as f:
            f.write(f"Initial conditions (x, y, alpha1, alpha2, epsilon):")
            f.write(f" {list(self.initial)};\n")
            f.write(f"Method: explicit Euler;\n")
            f.write(f"The processed time interval: 0-{round(self.timeline[-1], 4)};\n")
            f.write(f"Number of useful iterations: {iterations_num};\n")
            f.write(f"The time of run: {round(end - start, 4)} seconds;")
        self.draw_one_plot("t", "x")
        self.draw_one_plot("t", "y")
        self.draw_one_plot("t", "a1")
        self.draw_one_plot("t", "a2")
        self.draw_one_plot("x", "y")
        
    def iter_explicit_Euler_recalc(self):
        vector = self.current_vector()
        shift = self.shift
        if shift is None: shift = self.dynamic_shift()
        additional = vector + shift/2*self.derivatives_vector(vector)
        next_vector = vector +\
                      shift*self.derivatives_vector(additional)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)

    def method_explicit_Euler_recalc(self, iterations_num=None):
        self.reset()
        start = time.time()
        self.iter_implicit_Euler(10**(-5))
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2000:
                self.iter_explicit_Euler_recalc()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Euler_recalc()
                if self.timeline[-1] == 2000:
                    iterations_num = i + 1
                    break
        end = time.time()
        with open('last_run_results/about_last_run.txt', 'w') as f:
            f.write(f"Initial conditions (x, y, alpha1, alpha2, epsilon):")
            f.write(f" {list(self.initial)};\n")
            f.write(f"Method: explicit Euler recalc;\n")
            f.write(f"The processed time interval: 0-{round(self.timeline[-1], 4)};\n")
            f.write(f"Number of useful iterations: {iterations_num};\n")
            f.write(f"The time of run: {round(end - start, 4)} seconds;")
        self.draw_one_plot("t", "x")
        self.draw_one_plot("t", "y")
        self.draw_one_plot("t", "a1")
        self.draw_one_plot("t", "a2")
        self.draw_one_plot("x", "y")
    
    def iter_explicit_Runge_Kutta4(self):
        vector = self.current_vector()
        shift = self.shift
        if shift is None: shift = self.dynamic_shift()
        k1 = self.derivatives_vector(vector)
        k2 = self.derivatives_vector(vector + shift/2*k1)
        k3 = self.derivatives_vector(vector + shift/2*k2)
        k4 = self.derivatives_vector(vector + shift*k3)
        next_vector = vector + shift/6*(k1 + 2*k2 + 2*k3 + k4)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)
    
    def method_explicit_Runge_Kutta4(self, iterations_num=None):
        self.reset()
        start = time.time()
        self.iter_implicit_Euler(10**(-5))
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2000:
                self.iter_explicit_Runge_Kutta4()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Runge_Kutta4()
                if self.timeline[-1] == 2000:
                    iterations_num = i + 1
                    break
        end = time.time()
        with open('last_run_results/about_last_run.txt', 'w') as f:
            f.write(f"Initial conditions (x, y, alpha1, alpha2, epsilon):")
            f.write(f" {list(self.initial)};\n")
            f.write(f"Method: explicit RungeKutta4;\n")
            f.write(f"The processed time interval: 0-{round(self.timeline[-1], 4)};\n")
            f.write(f"Number of useful iterations: {iterations_num};\n")
            f.write(f"The time of run: {round(end - start, 4)} seconds;")
        self.draw_one_plot("t", "x")
        self.draw_one_plot("t", "y")
        self.draw_one_plot("t", "a1")
        self.draw_one_plot("t", "a2")
        self.draw_one_plot("x", "y")
    
    def iter_implicit_Euler(self, offset_to_break_out=None):
        vector = self.current_vector()
        shift = self.shift
        eps = self.eps_const
        if shift is None: shift = self.dynamic_shift()
            
        def implctEuler(w):
            F = np.zeros(4)
            F[0] = w[0]-vector[0]-shift*\
                   w[0]*(2*w[2]-0.5*w[0]-w[2]**2*w[3]**(-2)*w[1])
            F[1] = w[1]-vector[1]-shift*\
                   w[1]*(2*w[3]-w[2]**(-2)*w[3]**(-2)*w[0]-0.5*w[1])
            F[2] = w[2]-vector[2]-shift*\
                   eps*(2-2*w[2]*w[3]**(-2)*w[1])
            F[3] = w[3]-vector[3]-shift*\
                   eps*(2-2*w[3]*w[2]**(-2)*w[1])
            return F
        
        def jacobian_implctEuler(w):
            JacImpEul = np.zeros((4, 4))
            JacImpEul[0,0] = 1-shift*\
                        (-w[2]**2*w[1]*w[3]**(-2)+2*w[2]-w[0])
            JacImpEul[0,1] = -shift*\
                        (-w[2]**2*w[0]*w[3]**(-2))
            JacImpEul[0,2] = -shift*\
                        (w[0]*(-2*w[2]*w[1]*w[3]**(-2)+2))
            JacImpEul[0,3] = -shift*\
                        (2*w[2]**2*w[0]*w[1]*w[3]**(-3))
            JacImpEul[1,0] = -shift*\
                        (-w[1]*w[2]**(-2)*w[3]**(-2))
            JacImpEul[1,1] = 1-shift*\
                        (2*w[3]-w[1]-w[0]*w[2]**(-2)*w[3]**(-2))
            JacImpEul[1,2] = -shift*\
                        (2*w[0]*w[1]*w[2]**(-3)*w[3]**(-2))
            JacImpEul[1,3] = -shift*\
                        (w[1]*(2+2*w[0]*w[2]**(-2)*w[3]**(-3)))
            JacImpEul[2,0] = 0
            JacImpEul[2,1] = -shift*\
                        (-2*w[2]*self.eps_const*w[3]**(-2))
            JacImpEul[2,2] = 1-shift*\
                        (-2*self.eps_const*w[1]*w[2]**(-2))
            JacImpEul[2,3] = -shift*\
                        (4*w[2]*self.eps_const*w[1]*w[3]**(-3))
            JacImpEul[3,0] = 0
            JacImpEul[3,1] = -shift*\
                        (-2*w[3]*self.eps_const*w[2]**(-2))
            JacImpEul[3,2] = -shift*\
                        (4*w[3]*self.eps_const*w[1]*w[2]**(-3))
            JacImpEul[3,3] = 1-shift*\
                        (-2*self.eps_const*w[1]*w[2]**(-2))
            return JacImpEul
        
        if offset_to_break_out is None:
            initial_guess = vector
        else:
            initial_guess = vector +\
                np.array([list(offset_to_break_out for _ in range(4))])
        next_vector = fsolve(implctEuler,
                           initial_guess,fprime=jacobian_implctEuler)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)
                        
    def method_implicit_Euler(self, iterations_num=None):
        self.reset()
        start = time.time()
        self.iter_implicit_Euler(10**(-5))
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2000:
                self.iter_implicit_Euler()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_implicit_Euler()
                if self.timeline[-1] == 2000:
                    iterations_num = i + 1
                    break
        end = time.time()
        with open('last_run_results/about_last_run.txt', 'w') as f:
            f.write(f"Initial conditions (x, y, alpha1, alpha2, epsilon):")
            f.write(f" {list(self.initial)};\n")
            f.write(f"Method: implicit Euler;\n")
            f.write(f"The processed time interval: 0-{round(self.timeline[-1], 4)};\n")
            f.write(f"Number of useful iterations: {iterations_num};\n")
            f.write(f"The time of run: {round(end - start, 4)} seconds;")
        self.draw_one_plot("t", "x")
        self.draw_one_plot("t", "y")
        self.draw_one_plot("t", "a1")
        self.draw_one_plot("t", "a2")
        self.draw_one_plot("x", "y")

    def iter_implicit_trapezoid(self):
        vector = self.current_vector()
        drvtv_vector = self.derivatives_vector(vector)
        shift = self.shift
        eps = self.eps_const
        if shift is None: shift = self.dynamic_shift()
            
        def implct_trpzd(w):
            F = np.zeros(4)
            F[0] = w[0]-vector[0]-shift/2*drvtv_vector[0]-shift/2*\
                   w[0]*(2*w[2]-0.5*w[0]-w[2]**2*w[3]**(-2)*w[1])
            F[1] = w[1]-vector[1]-shift/2*drvtv_vector[1]-shift/2*\
                   w[1]*(2*w[3]-w[2]**(-2)*w[3]**(-2)*w[0]-0.5*w[1])
            F[2] = w[2]-vector[2]-shift/2*drvtv_vector[2]-shift/2*\
                   eps*(2-2*w[2]*w[3]**(-2)*w[1])
            F[3] = w[3]-vector[3]-shift/2*drvtv_vector[3]-shift/2*\
                   eps*(2-2*w[3]*w[2]**(-2)*w[1])
            return F
        
        def jacobian_implct_trpzd(w):
            JacImpTrap = np.zeros((4, 4))
            JacImpTrap[0,0] = 1-shift/2*\
                        (-w[2]**2*w[1]*w[3]**(-2)+2*w[2]-w[0])
            JacImpTrap[0,1] = -shift/2*\
                        (-w[2]**2*w[0]*w[3]**(-2))
            JacImpTrap[0,2] = -shift/2*\
                        (w[0]*(-2*w[2]*w[1]*w[3]**(-2)+2))
            JacImpTrap[0,3] = -shift/2*\
                        (2*w[2]**2*w[0]*w[1]*w[3]**(-3))
            JacImpTrap[1,0] = -shift/2*\
                        (-w[1]*w[2]**(-2)*w[3]**(-2))
            JacImpTrap[1,1] = 1-shift/2*\
                        (2*w[3]-w[1]-w[0]*w[2]**(-2)*w[3]**(-2))
            JacImpTrap[1,2] = -shift/2*\
                        (2*w[0]*w[1]*w[2]**(-3)*w[3]**(-2))
            JacImpTrap[1,3] = -shift/2*\
                        (w[1]*(2+2*w[0]*w[2]**(-2)*w[3]**(-3)))
            JacImpTrap[2,0] = 0
            JacImpTrap[2,1] = -shift/2*\
                        (-2*w[2]*self.eps_const*w[3]**(-2))
            JacImpTrap[2,2] = 1-shift/2*\
                        (-2*self.eps_const*w[1]*w[2]**(-2))
            JacImpTrap[2,3] = -shift/2*\
                        (4*w[2]*self.eps_const*w[1]*w[3]**(-3))
            JacImpTrap[3,0] = 0
            JacImpTrap[3,1] = -shift/2*\
                        (-2*w[3]*self.eps_const*w[2]**(-2))
            JacImpTrap[3,2] = -shift/2*\
                        (4*w[3]*self.eps_const*w[1]*w[2]**(-3))
            JacImpTrap[3,3] = 1-shift/2*\
                        (-2*self.eps_const*w[1]*w[2]**(-2))
            return JacImpTrap
        
        initial_guess = vector
        next_vector = fsolve(implct_trpzd,
                           initial_guess,fprime=jacobian_implct_trpzd)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)

    def method_implicit_trapezoid(self, iterations_num=None):
        self.reset()
        start = time.time()
        self.iter_implicit_Euler(10**(-5))
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2000:
                self.iter_implicit_trapezoid()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_implicit_trapezoid()
                if self.timeline[-1] == 2000:
                    iterations_num = i + 1
                    break
        end = time.time()
        with open('last_run_results/about_last_run.txt', 'w') as f:
            f.write(f"Initial conditions (x, y, alpha1, alpha2, epsilon):")
            f.write(f" {list(self.initial)};\n")
            f.write(f"Method: implicit trapezoid;\n")
            f.write(f"The processed time interval: 0-{round(self.timeline[-1], 4)};\n")
            f.write(f"Number of useful iterations: {iterations_num};\n")
            f.write(f"The time of run: {round(end - start, 4)} seconds;")
        self.draw_one_plot("t", "x")
        self.draw_one_plot("t", "y")
        self.draw_one_plot("t", "a1")
        self.draw_one_plot("t", "a2")
        self.draw_one_plot("x", "y")
    
#     def iter_implicit_Runge_Kutta4(self, offset_to_break_out=None):
#         vector = self.current_vector()
#         k1_d_vect = self.derivatives_vector(vector)
#         k2_d_vect = self.derivatives_vector(vector)
#         k1_d_vect = self.derivatives_vector(vector)
#         k1_d_vect = self.derivatives_vector(vector)
#         shift = self.shift
#         eps = self.eps_const
#         if shift is None: shift = self.dynamic_shift()
            
#         def implct_RK4_kfunc(w):
#             F = np.zeros(4)
#             F[0] = w[0]-vector[0]-shift/6*()
#             F[1] =
#             F[2] =
#             F[3] =
#             return F    
    
#     def method_implicit_Runge_Kutta4(self):
#         ...
        
if __name__ == "__main__":
    default_case = [40, 10, 0, 10, 0.005, 0.01]
    c = CauchySolving(default_case)
    c.method_implicit_Euler(10000)