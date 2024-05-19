import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings("ignore")

class CauchySolving:
    def __init__(self, step=0.01):
        case = [6, 5]
        self.set_initial(case)
        self.shift_status(step)
    
    def derivatives_vector(self, w):
        F = np.zeros(2)
        F[0] = 2*w[0] - 5*w[1] + 3
        F[1] = 5*w[0] - 6*w[1] + 1
        return F
    
    def set_initial(self, initial):
        self.initial = np.array([initial])[0]
        self.reset()
    
    def reset(self):
        self.data = np.array([self.initial])
        self.timeline = np.array([0.0])

    def current_vector(self):
        return self.data[-1, :]
    
    def update_data(self, update_vector):
        self.data = np.append(self.data, [update_vector], axis=0)
    
    def shift_status(self, value=0.01):
        self.shift = value
    
    def draw_one_plot(self, driven, name):
        x_orig, y_orig = [], []
        x = list(self.timeline)
        for t in x:
            x_orig.append( 5*np.exp(-2*t)*np.cos(3*t)+1 )
            y_orig.append( np.exp(-2*t)*(4*np.cos(3*t)+3*np.sin(3*t))+1 )
        orig = []
        buf_dct = {"x": 0, "y": 1}
        if driven == "x":
            y = list(self.data[:, buf_dct[driven]])
            orig = x_orig
        elif driven == "y":
            y = list(self.data[:, buf_dct[driven]])
            orig = y_orig
        else:
            return 0
        ax = plt.axes()
        ax.set_facecolor("#0A0A0A")
        plt.title(f'Dependence of t on {driven}', weight='bold')
        plt.xlabel(f't-axis', weight='bold')
        plt.ylabel(f'{driven}-axis', weight='bold')
        plt.plot(x, orig, color="#1CAC78", linewidth=1, label='original')
        plt.plot(x, y, color="#FF6E4A", linewidth=1, label=name)
        plt.grid(which='major', linewidth=0.3, color="white")
        plt.legend() 
        plt.show()
        miss = max(round(abs(orig[i] - y[i])/orig[i], 3) for i in range(len(orig)))
        print(f"{driven}-miss: {miss}%")

    def iter_explicit_Euler(self):
        vector = self.current_vector()
        shift = self.shift
        next_vector = vector + shift*self.derivatives_vector(vector)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)
        
    def method_explicit_Euler(self, iterations_num=None):
        self.reset()
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2:
                self.iter_explicit_Euler()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Euler()
                if self.timeline[-1] == 2:
                    iterations_num = i + 1
                    break
        self.draw_one_plot("x", "explicit Euler")
        self.draw_one_plot("y", "explicit Euler")
        
    def iter_explicit_Euler_recalc(self):
        vector = self.current_vector()
        shift = self.shift
        additional = vector + shift/2*self.derivatives_vector(vector)
        next_vector = vector +\
                      shift*self.derivatives_vector(additional)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)

    def method_explicit_Euler_recalc(self, iterations_num=None):
        self.reset()
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2:
                self.iter_explicit_Euler_recalc()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Euler_recalc()
                if self.timeline[-1] == 2:
                    iterations_num = i + 1
                    break
        self.draw_one_plot("x", "explicit Euler with recalculation")
        self.draw_one_plot("y", "explicit Euler with recalculation")
        
    def iter_explicit_Runge_Kutta4(self):
        vector = self.current_vector()
        shift = self.shift
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
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2:
                self.iter_explicit_Runge_Kutta4()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_explicit_Runge_Kutta4()
                if self.timeline[-1] == 2:
                    iterations_num = i + 1
                    break
        self.draw_one_plot("x", "explicit Runge Kutta4")
        self.draw_one_plot("y", "explicit Runge Kutta4")
        
    def iter_implicit_Euler(self):
        vector = self.current_vector()
        shift = self.shift
            
        def implctEuler(w):
            F = np.zeros(2)
            F[0] = w[0]-vector[0]-shift*\
                   (2*w[0] - 5*w[1] + 3)
            F[1] = w[1]-vector[1]-shift*\
                   (5*w[0] - 6*w[1] + 1)
            return F
        
        def jacobian_implctEuler(w):
            JacImpEul = np.zeros((2, 2))
            JacImpEul[0,0] = 1-shift*\
                        (2)
            JacImpEul[0,1] = -shift*\
                        (-5)
            JacImpEul[1,0] = -shift*\
                        (5)
            JacImpEul[1,1] = 1-shift*\
                        (-6)
            return JacImpEul
        
        initial_guess = vector
        next_vector = fsolve(implctEuler,
                           initial_guess,fprime=jacobian_implctEuler)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)
                        
    def method_implicit_Euler(self, iterations_num=None):
        self.reset()
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2:
                self.iter_implicit_Euler()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_implicit_Euler()
                if self.timeline[-1] == 2:
                    iterations_num = i + 1
                    break
        self.draw_one_plot("x", "implicit Euler")
        self.draw_one_plot("y", "implicit Euler")
        
    def iter_implicit_trapezoid(self):
        vector = self.current_vector()
        drvtv_vector = self.derivatives_vector(vector)
        shift = self.shift
            
        def implct_trpzd(w):
            F = np.zeros(2)
            F[0] = w[0]-vector[0]-shift/2*drvtv_vector[0]-shift/2*\
                   (2*w[0] - 5*w[1] + 3)
            F[1] = w[1]-vector[1]-shift/2*drvtv_vector[1]-shift/2*\
                   (5*w[0] - 6*w[1] + 1)
            return F
        
        def jacobian_implct_trpzd(w):
            JacImpTrap = np.zeros((2, 2))
            JacImpTrap[0,0] = 1-shift/2*\
                        (2)
            JacImpTrap[0,1] = -shift/2*\
                        (-5)
            JacImpTrap[1,0] = -shift/2*\
                        (5)
            JacImpTrap[1,1] = 1-shift/2*\
                        (-6)
            return JacImpTrap
        
        initial_guess = vector
        next_vector = fsolve(implct_trpzd,
                           initial_guess,fprime=jacobian_implct_trpzd)
        self.timeline = np.append(self.timeline,
                                  [self.timeline[-1] + shift], axis=0)
        self.update_data(next_vector)

    def method_implicit_trapezoid(self, iterations_num=None):
        self.reset()
        if iterations_num is None:
            iterations_num = 0
            while self.timeline[-1] < 2:
                self.iter_implicit_trapezoid()
                iterations_num += 1
        else:
            for i in range(iterations_num):
                self.iter_implicit_trapezoid()
                if self.timeline[-1] == 2:
                    iterations_num = i + 1
                    break
        self.draw_one_plot("x", "implicit trapezoid")
        self.draw_one_plot("y", "implicit trapezoid")

    
if __name__ == "__main__":
    c = CauchySolving(0.1)
    c.method_explicit_Euler()