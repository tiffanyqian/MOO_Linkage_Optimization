import pandas as pd
import numpy as np
import math
import cmath
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import csv


pop = 50
gen = 10
# Delta_2 = 6.83*cmath.exp(complex(0, 148.63*math.pi/180))
# Delta_3 = 11.43*cmath.exp(complex(0, 172.53*math.pi/180))
Delta_2 = 6.83*cmath.exp(complex(0, 88.55*math.pi/180))
Delta_3 = 11.43*cmath.exp(complex(0, 139.81*math.pi/180))
Delta_2_A = 6.62*cmath.exp(complex(0, 96.88*math.pi/180))
Delta_2_B = 7.17*cmath.exp(complex(0, 80.86*math.pi/180))
Delta_3_A = 11.63*cmath.exp(complex(0, 142.4*math.pi/180))
Delta_3_B = 11.25*cmath.exp(complex(0, 137.14*math.pi/180))
# Theta_1 = 0 deg, Theta_2 = 60.08 deg, Theta_3 = 32.72 deg
Alpha_2 = 60.08*math.pi/180
Alpha_3 = 32.72*math.pi/180
log = pd.DataFrame(columns=["A Transmission Angle", "B Transmission Angle", "Beta_2_A", "Beta_3_A",
                            "Beta_2_B", "Beta_3_B", "W_A_x", "W_A_y", "Z_A_x", "Z_A_y", "W_B_x",
                            "W_B_y", "Z_B_x", "Z_B_y", "mag_W_A", "mag_Z_A", "mag_W_B", "mag_Z_B",
                            "A Trans pos 2", "B Trans pos 2", "A Trans pos 3", "B Trans pos 3",
                            "Abs Dev A", "Abs Dev B"])
fig, ax = plt.subplots(figsize=(6, 6))
num = 1


class Linkage(Problem):

    def __init__(self):
        super().__init__(n_var=4, n_obj=5, n_ieq_constr=4, xl=[0, 0, 0, 0],
                         xu=[math.pi, math.pi, math.pi, math.pi])

    def _evaluate(self, x, out, *args, **kwargs):
        global log

        # print("Beta_2_A, Beta_3_A, Beta_2_B, Beta_3_B:", x)
        f = np.empty([len(x), 5])
        g = np.empty([len(x), 4])

        for i in range(len(x)):
            Beta_2_A, Beta_3_A, Beta_2_B, Beta_3_B = x[i, :]

            ## SIDE A ##
            top_W_A = (Delta_2_A*cmath.exp(complex(0, Alpha_2))) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                        - (Delta_3_A*cmath.exp(complex(0, Alpha_3))) * (cmath.exp(complex(0, Alpha_2)) - 1)
            bot_W_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                        - (cmath.exp(complex(0, Beta_3_A)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_W_A = top_W_A/bot_W_A
            W_A = [im_W_A.real, im_W_A.imag]
            mag_W_A = math.sqrt(W_A[0] ** 2 + W_A[1] ** 2)
            print(W_A)

            top_Z_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (Delta_3_A*cmath.exp(complex(0, Alpha_3))) \
                        - (cmath.exp(complex(0, Beta_3_A)) - 1) * (Delta_2_A*cmath.exp(complex(0, Alpha_2)))
            bot_Z_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                        - (cmath.exp(complex(0, Beta_3_A)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_Z_A = top_Z_A / bot_Z_A
            Z_A = [im_Z_A.real, im_Z_A.imag]
            mag_Z_A = math.sqrt(Z_A[0] ** 2 + Z_A[1] ** 2)
            print(Z_A)

            ## SIDE B ##
            top_W_B = (Delta_2_B * cmath.exp(complex(0, Alpha_2))) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                      - (Delta_3_B * cmath.exp(complex(0, Alpha_3))) * (cmath.exp(complex(0, Alpha_2)) - 1)
            bot_W_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                      - (cmath.exp(complex(0, Beta_3_B)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_W_B = top_W_B / bot_W_B
            W_B = [im_W_B.real, im_W_B.imag]
            mag_W_B = math.sqrt(W_B[0] ** 2 + W_B[1] ** 2)
            print(W_B)

            top_Z_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (Delta_3_B * cmath.exp(complex(0, Alpha_3))) \
                      - (cmath.exp(complex(0, Beta_3_B)) - 1) * (Delta_2_B * cmath.exp(complex(0, Alpha_2)))
            bot_Z_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) \
                      - (cmath.exp(complex(0, Beta_3_B)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_Z_B = top_Z_B / bot_Z_B
            Z_B = [im_Z_B.real, im_Z_B.imag]
            mag_Z_B = math.sqrt(Z_B[0]**2+Z_B[1]**2)
            print(Z_B)

            # transmission angles
            trans_A_1 = np.arccos((W_A[0]*Z_A[0]+W_A[1]*Z_A[1])/(mag_W_A*mag_Z_A))
            trans_B_1 = np.arccos((W_B[0]*Z_B[0]+W_B[1]*Z_B[1])/(mag_W_B*mag_Z_B))
            trans_A_2 = trans_A_1+Alpha_2-Beta_2_A
            trans_B_2 = trans_B_1+Alpha_2-Beta_2_B
            trans_A_3 = trans_A_1+Alpha_3-Beta_3_A
            trans_B_3 = trans_B_1+Alpha_3-Beta_3_B
            trans_A_1 = abs(trans_A_1 - math.pi / 2)
            trans_B_1 = abs(trans_B_1 - math.pi / 2)
            trans_A_2 = abs(trans_A_2 - math.pi / 2)
            trans_B_2 = abs(trans_B_2 - math.pi / 2)
            trans_A_3 = abs(trans_A_3 - math.pi / 2)
            trans_B_3 = abs(trans_B_3 - math.pi / 2)

            abs_dev_A = math.pi/2 - min([trans_A_1, trans_A_2, trans_A_3])
            abs_dev_B = math.pi/2 - min([trans_B_1, trans_B_2, trans_B_3])
            print("ABS DEV A", abs_dev_A)
            print("ABS DEV B", abs_dev_B)

            # log log
            temp_df = pd.DataFrame({"A Transmission Angle": [trans_A_1], "B Transmission Angle": [trans_B_1],
                                    "Beta_2_A": [Beta_2_A], "Beta_3_A": [Beta_3_A], "Beta_2_B": [Beta_2_B],
                                    "Beta_3_B": [Beta_3_B], "W_A_x": [W_A[0]], "W_A_y": [W_A[1]],
                                    "Z_A_x": [Z_A[0]], "Z_A_y": [Z_A[1]], "W_B_x": [W_B[0]],
                                    "W_B_y": [W_B[1]], "Z_B_x": [Z_B[0]], "Z_B_y": [Z_B[1]],
                                    "mag_W_A": [mag_W_A], "mag_Z_A": [mag_Z_A], "mag_W_B": [mag_W_B],
                                    "mag_Z_B": [mag_Z_B], "A Trans pos 2": [trans_A_2],
                                    "B Trans pos 2": [trans_B_2], "A Trans pos 3": [trans_A_3],
                                    "B Trans pos 3": [trans_B_3], "Abs Dev A": [abs_dev_A],
                                    "Abs Dev B": [abs_dev_B]})
            log = pd.concat([log, temp_df], ignore_index=True)

            # Fitness
            # f[i, :] = [trans_A_1, trans_A_2, trans_A_3, abs(mag_W_A), abs(mag_Z_A),
            #            abs(mag_W_B), abs(mag_Z_B)]
            # f[i, :] = [abs_dev_A, abs(mag_W_A), abs(mag_Z_A), abs(mag_W_B), abs(mag_Z_B)]
            f[i, :] = [abs_dev_B, abs(mag_W_A), abs(mag_Z_A), abs(mag_W_B), abs(mag_Z_B)]
            print(f[i, :])

            # CONSTRAINT HANDLING
            A1_1 = [-0.71 - Z_A[0], 0.71 - Z_A[1]]
            A0 = [A1_1[0] - W_A[0], A1_1[1] - W_A[1]]
            B1_1 = [0.71 - Z_B[0], -0.71 - Z_B[1]]
            B0 = [B1_1[0] - W_B[0], B1_1[1] - W_B[1]]
            g[i, :] = [-11.71-A0[1], -11.71-B0[1], -16.8-A0[0], -16.8-B0[0]]

        out["F"] = f
        out["G"] = g


def run(**kwargs):
    global pop
    global gen
    pop = kwargs.get('pop')
    gen = kwargs.get('gen')

    problem = Linkage()
    algorithm = NSGA2(pop_size=pop)

    # Edit termination and seed parameters
    res = minimize(problem, algorithm, termination=('n_gen', gen))

    print(log)
    log.to_csv('log.csv', index=False)

    print("Best solution (Betas) found: %s" % res.X)
    print("Transmission Angle values: %s" % res.F)

    df_res = pd.DataFrame(np.array(res.F))
    best_ind_B = df_res.iloc[:, 1] - math.pi / 2
    best_ind_B = best_ind_B.idxmin()

    val_B = log[log["mag_W_A"] == res.F[best_ind_B][1]]
    val_B = val_B[val_B["mag_Z_A"] == res.F[best_ind_B][2]]
    val_B = val_B[val_B["mag_W_B"] == res.F[best_ind_B][3]]
    val_B = val_B[val_B["mag_Z_B"] == res.F[best_ind_B][4]].reset_index(drop=True)
    print(val_B)
    graph_em(val_B)

    return res


def graph_em(val):
    global num

    Beta_2_A, Beta_3_A, Beta_2_B, Beta_3_B, \
        W_A_x, W_A_y, Z_A_x, Z_A_y, W_B_x, W_B_y, Z_B_x, Z_B_y,\
        mag_W_A, mag_Z_A, mag_W_B, mag_Z_B = val.iloc[0, 2:18]

    ta1 = abs(val.iloc[0, 20] * 180 / math.pi)
    tb1 = abs(val.iloc[0, 21] * 180 / math.pi)
    ta2 = abs(val.iloc[0, 18] * 180 / math.pi)
    tb2 = abs(val.iloc[0, 19] * 180 / math.pi)
    ta3 = abs(val.iloc[0, 0] * 180 / math.pi)
    tb3 = abs(val.iloc[0, 1] * 180 / math.pi)
    print("Trans A pos 1:", ta1)
    print("Trans B pos 1:", tb1)
    print("Trans A pos 2:", ta2)
    print("Trans B pos 2:", tb2)
    print("Trans A pos 3:", ta3)
    print("Trans B pos 3:", tb3)
    print("Abs deviation trans A:", val.iloc[0, -2] * 180 / math.pi)
    print("Abs deviation trans B:", val.iloc[0, -1] * 180 / math.pi)

    if val.empty:
        print("Broke.")
    else:
        plt.figure(num)
        num = num+1

        # math set up
        # pos 1
        A1_1 = [-0.71-Z_A_x, 0.71-Z_A_y]
        A0 = [A1_1[0]-W_A_x, A1_1[1]-W_A_y]
        B1_1 = [0.71-Z_B_x, -0.71-Z_B_y]
        B0 = [B1_1[0]-W_B_x, B1_1[1]-W_B_y]
        # pos 2
        W_A_2 = cmath.exp(complex(0, Beta_2_A)) * complex(W_A_x, W_A_y)
        W_A_2 = [W_A_2.real, W_A_2.imag]
        Z_A_2 = cmath.exp(complex(0, Alpha_2)) * complex(Z_A_x, Z_A_y)
        Z_A_2 = [Z_A_2.real, Z_A_2.imag]
        A2_1 = [A0[0]+W_A_2[0], A0[1]+W_A_2[1]]
        A2_2 = [A2_1[0]+Z_A_2[0], A2_1[1]+Z_A_2[1]]
        W_B_2 = cmath.exp(complex(0, Beta_2_B)) * complex(W_B_x, W_B_y)
        W_B_2 = [W_B_2.real, W_B_2.imag]
        Z_B_2 = cmath.exp(complex(0, Alpha_2)) * complex(Z_B_x, Z_B_y)
        Z_B_2 = [Z_B_2.real, Z_B_2.imag]
        B2_1 = [B0[0] + W_B_2[0], B0[1] + W_B_2[1]]
        B2_2 = [B2_1[0] + Z_B_2[0], B2_1[1] + Z_B_2[1]]
        # pos 3
        W_A_3 = cmath.exp(complex(0, Beta_3_A)) * complex(W_A_x, W_A_y)
        W_A_3 = [W_A_3.real, W_A_3.imag]
        Z_A_3 = cmath.exp(complex(0, Alpha_3)) * complex(Z_A_x, Z_A_y)
        Z_A_3 = [Z_A_3.real, Z_A_3.imag]
        A3_1 = [A0[0] + W_A_3[0], A0[1] + W_A_3[1]]
        A3_2 = [A3_1[0] + Z_A_3[0], A3_1[1] + Z_A_3[1]]
        W_B_3 = cmath.exp(complex(0, Beta_3_B)) * complex(W_B_x, W_B_y)
        W_B_3 = [W_B_3.real, W_B_3.imag]
        Z_B_3 = cmath.exp(complex(0, Alpha_3)) * complex(Z_B_x, Z_B_y)
        Z_B_3 = [Z_B_3.real, Z_B_3.imag]
        B3_1 = [B0[0] + W_B_3[0], B0[1] + W_B_3[1]]
        B3_2 = [B3_1[0] + Z_B_3[0], B3_1[1] + Z_B_3[1]]

        ## PLOT LINKS ##
        # pos 1
        plt.plot([A0[0], A1_1[0]], [A0[1], A1_1[1]], color="red")
        plt.plot([-0.71, A1_1[0]], [0.71, A1_1[1]], color="orange")
        plt.plot([B0[0], B1_1[0]], [B0[1], B1_1[1]], color="blue")
        plt.plot([0.71, B1_1[0]], [-0.71, B1_1[1]], color="green")
        plt.legend(['W_A', 'Z_A', 'W_B', 'Z_B'], loc='upper left')
        # pos 2
        plt.plot([A0[0], A2_1[0]], [A0[1], A2_1[1]], color="red", linestyle="dashed")
        plt.plot([A2_1[0], A2_2[0]], [A2_1[1], A2_2[1]], color="orange", linestyle="dashed")
        plt.plot([B0[0], B2_1[0]], [B0[1], B2_1[1]], color="blue", linestyle="dashed")
        plt.plot([B2_1[0], B2_2[0]], [B2_1[1], B2_2[1]], color="green", linestyle="dashed")
        # pos 3
        plt.plot([A0[0], A3_1[0]], [A0[1], A3_1[1]], color="red", linestyle="dotted")
        plt.plot([A3_1[0], A3_2[0]], [A3_1[1], A3_2[1]], color="orange", linestyle="dotted")
        plt.plot([B0[0], B3_1[0]], [B0[1], B3_1[1]], color="blue", linestyle="dotted")
        plt.plot([B3_1[0], B3_2[0]], [B3_1[1], B3_2[1]], color="green", linestyle="dotted")

        ## PLOT BOARD & HOLES ##
        plt.scatter([A0[0], B0[0]], [A0[1], B0[1]], marker='*', color="black")
        plt.plot([7.21, 7.21, -16.8, -16.8, 7.21], [-11.71, 12.3, 12.3, -11.71, -11.71], color="black")
        plt.plot([-6.3, -6.3, -12.3, -12.3, -6.3], [-10.21, -4.21, -4.21, -10.21, -10.2], color="black",
                 linestyle="dashed")
        ax.add_patch(plt.Circle((-0.71, 0.71), 0.56, fill=False))
        ax.add_patch(plt.Circle((0.71, -0.71), 0.56, fill=False))
        ax.add_patch(plt.Circle((-4.87, 3.82), 0.56, fill=False))
        ax.add_patch(plt.Circle((-6.80, 3.3), 0.56, fill=False))
        ax.add_patch(plt.Circle((-10.37, 1.28), 0.56, fill=False))
        ax.add_patch(plt.Circle((-12.30, 1.70), 0.56, fill=False))
        ax.add_patch(plt.Circle((-1.8, 6.3), 0.56, fill=False))


run(pop=200, gen=100)
plt.show()
