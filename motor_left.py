import pandas as pd
import numpy as np
import math
import cmath
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt

## *** THIS PYTHON SCRIPT ASSUMES MOTOR ON THE LEFT SIDE *** ##

## PROBLEM DEFINITION IS HERE, CHANGE HERE TO OPTIMIZE A DIFF PROBLEM ##
# Define (0,0) as shifted to the midpoint of the rightmost button set. See class notes for where the convention
# comes from- pls do not ask me where the angles for the deltas came from they are sooo trial and error
Delta_2 = complex(-5.83, 3.56)
Delta_3 = complex(-11.33, 1.49)
# Theta_1 = 0 deg, Theta_2 = 60.08 deg, Theta_3 = 32.72 deg, and then Alpha_j is diff between Theta_j and Theta_1
Alpha_2 = 60.08 * math.pi / 180
Alpha_3 = 32.72 * math.pi / 180

# log file definition, don't touch!
log = pd.DataFrame(columns=["A_trans_ang_1", "B_trans_ang_1", "Beta_2_A", "Beta_3_A",
                            "Beta_2_B", "Beta_3_B", "W_A_x", "W_A_y", "Z_A_x", "Z_A_y", "W_B_x",
                            "W_B_y", "Z_B_x", "Z_B_y", "mag_W_A", "mag_Z_A", "mag_W_B", "mag_Z_B",
                            "A_trans_ang_2", "B_trans_ang_2", "A_trans_ang_3", "B_trans_ang_3"])

# some hyperparameters for the number of objectives and number of inequality constraints for pymoo, please
# change here only!!!
obj = 7
ieq = 13


class Linkage(Problem):
    # This is the class that defines the problem and is passed into pymoo to run the optimiziation alg NSGA-II on
    # Note that this is a MINIMIZING ALGORITHM, so the alg tries to make all objectives as small as possible

    def __init__(self):
        # This is the initialization of the problem and how many things you guess and constrain for
        super().__init__(n_var=4, n_obj=obj, n_ieq_constr=ieq, xl=[0, 0, 0, 0],
                         xu=[math.pi, math.pi, math.pi, math.pi])
        # -> n_var is the number of random variables that you generate to evaluate, for this problem
        # it is the angles Beta_2 and Beta_3, two for both side A and side B. The bounds of this variable are
        # set by xl and xu, both of length n_var, where xl is the lower bound of each possible variable and
        # xu is the upper bound of each possible variable
        # -> n_obj is the number of objectives you are constraining for, i.e. transmission angle or link length
        # -> n_ieq_constr is the number of constraints for the solutions, i.e. the ground links can't go outside
        # the board. The alg ensure that every n_ieq_constr <= 0.
        # NOTE: if you change n_obj or n_ieq_constr, please change it above (search obj, ieq)

    def _evaluate(self, x, out, *args, **kwargs):
        global log
        #
        f = np.empty([len(x), obj])
        g = np.empty([len(x), ieq])

        for i in range(len(x)):
            # These are the randomly generated guesses for the two Betas on both side
            Beta_2_A, Beta_3_A, Beta_2_B, Beta_3_B = x[i, :]

            # Matrix math for left side (side A) and right side (side B) below. For convention, see MD class
            # notes.
            ## SIDE A ##
            top_W_A = (Delta_2) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (Delta_3) * (cmath.exp(complex(0, Alpha_2)) - 1)
            bot_W_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (cmath.exp(complex(0, Beta_3_A)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_W_A = top_W_A / bot_W_A
            W_A = [im_W_A.real, im_W_A.imag]
            mag_W_A = math.sqrt(W_A[0] ** 2 + W_A[1] ** 2)
            # print(W_A)
            top_Z_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (Delta_3) - \
                      (cmath.exp(complex(0, Beta_3_A)) - 1) * (Delta_2)
            bot_Z_A = (cmath.exp(complex(0, Beta_2_A)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (cmath.exp(complex(0, Beta_3_A)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_Z_A = top_Z_A / bot_Z_A
            Z_A = [im_Z_A.real, im_Z_A.imag]
            mag_Z_A = math.sqrt(Z_A[0] ** 2 + Z_A[1] ** 2)
            # print(Z_A)
            ## SIDE B ##
            top_W_B = (Delta_2) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (Delta_3) * (cmath.exp(complex(0, Alpha_2)) - 1)
            bot_W_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (cmath.exp(complex(0, Beta_3_B)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_W_B = top_W_B / bot_W_B
            W_B = [im_W_B.real, im_W_B.imag]
            mag_W_B = math.sqrt(W_B[0] ** 2 + W_B[1] ** 2)
            # print(W_B)
            top_Z_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (Delta_3) - \
                      (cmath.exp(complex(0, Beta_3_B)) - 1) * (Delta_2)
            bot_Z_B = (cmath.exp(complex(0, Beta_2_B)) - 1) * (cmath.exp(complex(0, Alpha_3)) - 1) - \
                      (cmath.exp(complex(0, Beta_3_B)) - 1) * (cmath.exp(complex(0, Alpha_2)) - 1)
            im_Z_B = top_Z_B / bot_Z_B
            Z_B = [im_Z_B.real, im_Z_B.imag]
            mag_Z_B = math.sqrt(Z_B[0] ** 2 + Z_B[1] ** 2)
            # print(Z_B)

            # Define joint positions. A1 and B1 are where the coupler joins with the input and follower link,
            # i.e. Z_A to W_A. A0 and B0 are the ground positions.
            A1_1 = [-Z_A[0], -Z_A[1]]
            A0 = [A1_1[0] - W_A[0], A1_1[1] - W_A[1]]
            B1_1 = [-Z_B[0], -Z_B[1]]
            B0 = [B1_1[0] - W_B[0], B1_1[1] - W_B[1]]
            # Pos 2 (middle)
            W_A_2 = cmath.exp(complex(0, Beta_2_A)) * complex(W_A[0], W_A[1])
            W_A_2 = [W_A_2.real, W_A_2.imag]
            A1_2 = [A0[0] + W_A_2[0], A0[1] + W_A_2[1]]
            W_B_2 = cmath.exp(complex(0, Beta_2_B)) * complex(W_B[0], W_B[1])
            W_B_2 = [W_B_2.real, W_B_2.imag]
            B1_2 = [B0[0] + W_B_2[0], B0[1] + W_B_2[1]]
            # Pos 3 (leftmost)
            W_A_3 = cmath.exp(complex(0, Beta_3_A)) * complex(W_A[0], W_A[1])
            W_A_3 = [W_A_3.real, W_A_3.imag]
            A1_3 = [A0[0] + W_A_3[0], A0[1] + W_A_3[1]]
            W_B_3 = cmath.exp(complex(0, Beta_3_B)) * complex(W_B[0], W_B[1])
            W_B_3 = [W_B_3.real, W_B_3.imag]
            B1_3 = [B0[0] + W_B_3[0], B0[1] + W_B_3[1]]
            A1_2 = [B1_2[0] - A1_2[0], B1_2[1] - A1_2[1]]
            A1_3 = [B1_3[0] - A1_3[0], B1_3[1] - A1_3[1]]
            coup_bot_1 = [B1_1[0] - A1_1[0], B1_1[1] - A1_1[1]]
            mag_coup_bot = math.sqrt(coup_bot_1[0] ** 2 + coup_bot_1[1] ** 2)

            # *** TRANSMISSION ANGLES ***
            # Use the cos(theta) = (a dot b) / (|a||b|) eq to calculate first transmission angle
            trans_A_1 = np.arccos((W_A[0] * coup_bot_1[0] + W_A[1] * coup_bot_1[1]) / (mag_W_A * mag_coup_bot))
            trans_B_1 = np.arccos((W_B[0] * coup_bot_1[0] + W_B[1] * coup_bot_1[1]) / (mag_W_B * mag_coup_bot))
            # Every transmission angle past first position adds alpha and subtracts alpha according to
            # my lil mental math. pls tell me if I'm wrong
            trans_A_2 = trans_A_1 + Alpha_2 - Beta_2_A
            trans_B_2 = trans_B_1 + Alpha_2 - Beta_2_B
            trans_A_3 = trans_A_1 + Alpha_3 - Beta_3_A
            trans_B_3 = trans_B_1 + Alpha_3 - Beta_3_B

            # Process transmission angle math
            t_angs = [trans_A_1, trans_B_1, trans_A_2, trans_B_2, trans_A_3, trans_B_3]
            for ang in range(len(t_angs)):
                # This makes sure all angles are "acute" (less than 180 deg)
                t_angs[ang] = min(abs(t_angs[ang]), math.pi - abs(t_angs[ang]))
                # This gets the deviation from 90 degrees
                t_angs[ang] = abs(math.pi / 2 - t_angs[ang])
            trans_A_1, trans_B_1, trans_A_2, trans_B_2, trans_A_3, trans_B_3 = t_angs

            # This calculates the absolute transmission angle deviation
            abs_dev_A = max([trans_A_1, trans_A_2, trans_A_3])
            abs_dev_B = max([trans_B_1, trans_B_2, trans_B_3])
            # print("ABS DEV A", abs_dev_A)
            # print("ABS DEV B", abs_dev_B)

            # Log everything so you can access it later >:)
            temp_df = pd.DataFrame({"A_trans_ang_1": [trans_A_1], "B_trans_ang_1": [trans_B_1],
                                    "Beta_2_A": [Beta_2_A], "Beta_3_A": [Beta_3_A], "Beta_2_B": [Beta_2_B],
                                    "Beta_3_B": [Beta_3_B], "W_A_x": [W_A[0]], "W_A_y": [W_A[1]],
                                    "Z_A_x": [Z_A[0]], "Z_A_y": [Z_A[1]], "W_B_x": [W_B[0]],
                                    "W_B_y": [W_B[1]], "Z_B_x": [Z_B[0]], "Z_B_y": [Z_B[1]],
                                    "mag_W_A": [mag_W_A], "mag_Z_A": [mag_Z_A], "mag_W_B": [mag_W_B],
                                    "mag_Z_B": [mag_Z_B], "A_trans_ang_2": [trans_A_2],
                                    "B_trans_ang_2": [trans_B_2], "A_trans_ang_3": [trans_A_3],
                                    "B_trans_ang_3": [trans_B_3]})
            log = pd.concat([log, temp_df], ignore_index=True)

            # *** FITNESS ***
            # (These are the objectives)
            f[i, :] = [trans_B_1, trans_B_2, trans_B_3, abs(mag_W_A), abs(mag_Z_A),
                       abs(mag_W_B), abs(mag_Z_B)]
            # f[i, :] = [abs_dev_A, abs(mag_W_A), abs(mag_Z_A), abs(mag_W_B), abs(mag_Z_B)]

            # *** CONSTRAINT HANDLING ***
            # Define joint positions. A1 and B1 are where the coupler joins with the input and follower link,
            # i.e. Z_A to W_A. A0 and B0 are the ground positions.
            A1_1 = [-Z_A[0], -Z_A[1]]
            A0 = [A1_1[0] - W_A[0], A1_1[1] - W_A[1]]
            B1_1 = [-Z_B[0], -Z_B[1]]
            B0 = [B1_1[0] - W_B[0], B1_1[1] - W_B[1]]
            # Trying to save space by not redefining extra variables, so ineq. constraints are as follows:
            g[i, :] = [-10 - A0[1], -10 - B0[1], A0[1] - 2, B0[1] - 2, -16.8 - A0[0], -16.8 - B0[0],
                       A0[0] - 7.21, B0[0] - 7.21, mag_W_A - 10, mag_Z_A - 10, mag_W_B - 10, mag_Z_B - 10,
                       A0[0] - B0[0]]
            # -10 - A0[1], -10 - B0[1] -> y-coord of ground links A0 and B0 must be above y = -10
            # A0[1] - 2, B0[1] - 2 -> y-coord of ground links A0 and B0 must be below y = 2
            # -16.8 - A0[0], -16.8 - B0[0] -> x-coord of ground links A0 and B0 must be to the right of x = -16.8
            # A0[0] - 7.21, B0[0] - 7.21 ->  x-coord of ground links A0 and B0 must be to the left of x = 7.21
            # mag_W_A - 10, mag_Z_A - 10, mag_W_B - 10, mag_Z_B - 10 -> link lengths must be less than 10 inches
            # A0[0] - B0[0] -> x-coord of ground link A0 must be to the left of x-coord of ground link B0

        out["F"] = f
        out["G"] = g


def run(pop, gen):
    # Pymoo set up, you can read the documentation if you really want
    problem = Linkage()
    algorithm = NSGA2(pop_size=pop)
    # Edit termination and seed parameters
    res = minimize(problem, algorithm, termination=('n_gen', gen))
    ## Uncomment below to see all solutions:
    # print("Best solution (Betas) found: %s" % res.X)
    # print("Transmission Angle values: %s" % res.F)
    # print("Constraints: %s" % res.G)

    # Log only best final results
    res_log = pd.DataFrame(columns=["A_trans_ang_1", "B_trans_ang_1", "Beta_2_A", "Beta_3_A",
                                    "Beta_2_B", "Beta_3_B", "W_A_x", "W_A_y", "Z_A_x", "Z_A_y", "W_B_x",
                                    "W_B_y", "Z_B_x", "Z_B_y", "mag_W_A", "mag_Z_A", "mag_W_B", "mag_Z_B",
                                    "A_trans_ang_2", "B_trans_ang_2", "A_trans_ang_3", "B_trans_ang_3"])
    for i in range(0, len(res.F)):
        temp = log[log["mag_W_A"] == res.F[i][3]]
        temp = temp[temp["mag_Z_A"] == res.F[i][4]]
        temp = temp[temp["mag_W_B"] == res.F[i][5]]
        temp = temp[temp["mag_Z_B"] == res.F[i][6]].reset_index(drop=True).dropna()
        if not temp.empty:
            res_log = pd.concat([res_log, temp], ignore_index=True)
    res_log.to_csv('res_log.csv', index=False)

    ## *** BEST TRANSMISSION ANGLE *** ###
    # This tries to find the index of the best transmission angle in the results and then pull the data
    # In this case, minimizing abs dev of transmission angle of side A, meaning the motor is on the right side
    best_ind_ang = 0
    abs_dev = 90
    for i in range(0, len(res.pop.get("F"))):
        temp_abs_dev = max(res.F[i][0], res.F[i][1], res.F[i][2])
        if temp_abs_dev <= abs_dev:
            best_ind_ang = i
            abs_dev = temp_abs_dev
    val_ang = pd.DataFrame(res_log.loc[[best_ind_ang]])
    print(val_ang)
    if val_ang.empty:
        print("Broke.")
        exit("Your result function is empty! Or something else is broken...")
    else:
        print("Fig 1. index in log file is:", best_ind_ang)
        plt.figure(1)
        plt.title("Minimizing Abs Dev Transmission Angle, side B (right)")
        graph_em(val_ang)

    ## *** SHORTEST LINK LENGTHS *** ###
    # This tries to find the index of the shortest follower links (in this case: side A, red and yellow),
    # in the results and then pull the data
    df_res = pd.DataFrame(np.array(res.F))
    best_ind_len = df_res.iloc[:, 5].idxmin()
    val_len = pd.DataFrame(res_log.loc[[best_ind_len]])
    print(val_len)
    if val_len.empty:
        print("Broke.")
        exit("Your result function is empty! Or something else is broken...")
    else:
        print("Fig 2. index in log file is:", best_ind_len)
        plt.figure(2)
        plt.title("Minimizing Follower Link Length, side B (right)")
        graph_em(val_len)

    return res


def graph_em(val):
    # NOTE: val must be a dataframe

    # This unpacks the values that are needed for calculating positions
    Beta_2_A, Beta_3_A, Beta_2_B, Beta_3_B, \
        W_A_x, W_A_y, Z_A_x, Z_A_y, W_B_x, W_B_y, Z_B_x, Z_B_y, \
        mag_W_A, mag_Z_A, mag_W_B, mag_Z_B = val.iloc[0, 2:18]

    # Gets transmission angles: will just pull if val is long enough to contain transmission angles.
    # Else, it'll just calculate them.
    trans_A_1 = val.iloc[0, 0] * 180 / math.pi
    trans_B_1 = val.iloc[0, 1] * 180 / math.pi
    if len(val.columns) > 18:
        trans_A_2 = val.iloc[0, 18] * 180 / math.pi
        trans_B_2 = val.iloc[0, 19] * 180 / math.pi
        trans_A_3 = val.iloc[0, 20] * 180 / math.pi
        trans_B_3 = val.iloc[0, 21] * 180 / math.pi
    else:
        trans_A_2 = trans_A_1 + (Alpha_2 - Beta_2_A) * 180 / math.pi
        trans_B_2 = trans_B_1 + (Alpha_2 - Beta_2_B) * 180 / math.pi
        trans_A_3 = trans_A_1 + (Alpha_3 - Beta_3_A) * 180 / math.pi
        trans_B_3 = trans_B_1 + (Alpha_3 - Beta_3_B) * 180 / math.pi
        trans_A_2 = abs(90 - trans_A_2)
        trans_B_2 = abs(90 - trans_B_2)
        trans_A_3 = abs(90 - trans_A_3)
        trans_B_3 = abs(90 - trans_B_3)
    abs_dev_A = max([trans_A_1, trans_A_2, trans_A_3])
    abs_dev_B = max([trans_B_1, trans_B_2, trans_B_3])
    # Print transmission angles in readable form (degrees)
    print("Trans A pos 1:", trans_A_1)
    print("Trans B pos 1:", trans_B_1)
    print("Trans A pos 2:", trans_A_2)
    print("Trans B pos 2:", trans_B_2)
    print("Trans A pos 3:", trans_A_3)
    print("Trans B pos 3:", trans_B_3)
    print("Abs deviation trans A:", abs_dev_A, "degrees")
    print("Abs deviation trans B:", abs_dev_B, "degrees")

    ## *** MATH SET UP *** ##
    # This is where all the math is done to calculate the positions of the other positions other than the initially
    # defined one.
    # General format is that A2 or B2 is the position of the acrylic piece on the button (pos 1 doesn't have one
    # because we know where the first buttons are), A1 or B1 is the joint where the coupler will connect to the follower
    # or coupler links. A0 and B0 are the ground positions, and are the same. A1 and A2 will be different for each
    # position.
    # See Yevzo's notes for where this convention is from... it's messy so just ask me if you need more clarification.
    # Pos 1 (rightmost)
    A1_1 = [-Z_A_x, -Z_A_y]
    A0 = [A1_1[0] - W_A_x, A1_1[1] - W_A_y]
    B1_1 = [-Z_B_x, - Z_B_y]
    B0 = [B1_1[0] - W_B_x, B1_1[1] - W_B_y]
    # Pos 2 (middle)
    W_A_2 = cmath.exp(complex(0, Beta_2_A)) * complex(W_A_x, W_A_y)
    W_A_2 = [W_A_2.real, W_A_2.imag]
    Z_A_2 = cmath.exp(complex(0, Alpha_2)) * complex(Z_A_x, Z_A_y)
    Z_A_2 = [Z_A_2.real, Z_A_2.imag]
    A1_2 = [A0[0] + W_A_2[0], A0[1] + W_A_2[1]]
    A2_2 = [A1_2[0] + Z_A_2[0], A1_2[1] + Z_A_2[1]]
    W_B_2 = cmath.exp(complex(0, Beta_2_B)) * complex(W_B_x, W_B_y)
    W_B_2 = [W_B_2.real, W_B_2.imag]
    Z_B_2 = cmath.exp(complex(0, Alpha_2)) * complex(Z_B_x, Z_B_y)
    Z_B_2 = [Z_B_2.real, Z_B_2.imag]
    B1_2 = [B0[0] + W_B_2[0], B0[1] + W_B_2[1]]
    B2_2 = [B1_2[0] + Z_B_2[0], B1_2[1] + Z_B_2[1]]
    # Pos 3 (leftmost)
    W_A_3 = cmath.exp(complex(0, Beta_3_A)) * complex(W_A_x, W_A_y)
    W_A_3 = [W_A_3.real, W_A_3.imag]
    Z_A_3 = cmath.exp(complex(0, Alpha_3)) * complex(Z_A_x, Z_A_y)
    Z_A_3 = [Z_A_3.real, Z_A_3.imag]
    A1_3 = [A0[0] + W_A_3[0], A0[1] + W_A_3[1]]
    A2_3 = [A1_3[0] + Z_A_3[0], A1_3[1] + Z_A_3[1]]
    W_B_3 = cmath.exp(complex(0, Beta_3_B)) * complex(W_B_x, W_B_y)
    W_B_3 = [W_B_3.real, W_B_3.imag]
    Z_B_3 = cmath.exp(complex(0, Alpha_3)) * complex(Z_B_x, Z_B_y)
    Z_B_3 = [Z_B_3.real, Z_B_3.imag]
    B1_3 = [B0[0] + W_B_3[0], B0[1] + W_B_3[1]]
    B2_3 = [B1_3[0] + Z_B_3[0], B1_3[1] + Z_B_3[1]]

    # some rando functions for graphing pretty, ignore
    fig = plt.gcf()
    fig.gca().set_aspect("equal")
    ax = fig.gca()

    ## *** PLOT LINKS *** ##
    # General format: plotting order goes W_A (red), Z_A (orange), W_B (blue), Z_B (green)
    # Pos 1 (rightmost position)
    plt.plot([A0[0], A1_1[0]], [A0[1], A1_1[1]], color="red")
    plt.plot([0, A1_1[0]], [0, A1_1[1]], color="orange")
    plt.plot([B0[0], B1_1[0]], [B0[1], B1_1[1]], color="blue")
    plt.plot([0, B1_1[0]], [0, B1_1[1]], color="green")
    plt.legend(['W_A', 'Z_A', 'W_B', 'Z_B'], loc='upper left')
    # Pos 2 (middle position)
    plt.plot([A0[0], A1_2[0]], [A0[1], A1_2[1]], color="red", linestyle="dashed")
    plt.plot([A1_2[0], A2_2[0]], [A1_2[1], A2_2[1]], color="orange", linestyle="dashed")
    plt.plot([B0[0], B1_2[0]], [B0[1], B1_2[1]], color="blue", linestyle="dashed")
    plt.plot([B1_2[0], B2_2[0]], [B1_2[1], B2_2[1]], color="green", linestyle="dashed")
    # Pos 3 (leftmost position)
    plt.plot([A0[0], A1_3[0]], [A0[1], A1_3[1]], color="red", linestyle="dotted")
    plt.plot([A1_3[0], A2_3[0]], [A1_3[1], A2_3[1]], color="orange", linestyle="dotted")
    plt.plot([B0[0], B1_3[0]], [B0[1], B1_3[1]], color="blue", linestyle="dotted")
    plt.plot([B1_3[0], B2_3[0]], [B1_3[1], B2_3[1]], color="green", linestyle="dotted")

    ## *** PLOT BOARD & HOLES *** ##
    # This scatter plots the ground positions
    plt.scatter([A0[0], B0[0]], [A0[1], B0[1]], marker='*', color="black")
    # This plots the board bounds
    plt.plot([7.21, 7.21, -16.8, -16.8, 7.21], [-11.71, 12.3, 12.3, -11.71, -11.71], color="black")
    # This plots the section where attachment holes are
    plt.plot([-6.3, -6.3, -12.3, -12.3, -6.3], [-10.21, -4.21, -4.21, -10.21, -10.2], color="black",
             linestyle="dashed")
    # These patches plot the button positions
    ax.add_patch(plt.Circle((-0.71, 0.71), 0.56, fill=False))
    ax.add_patch(plt.Circle((0.71, -0.71), 0.56, fill=False))
    ax.add_patch(plt.Circle((-4.87, 3.82), 0.56, fill=False))
    ax.add_patch(plt.Circle((-6.80, 3.3), 0.56, fill=False))
    ax.add_patch(plt.Circle((-10.37, 1.28), 0.56, fill=False))
    ax.add_patch(plt.Circle((-12.30, 1.70), 0.56, fill=False))
    ax.add_patch(plt.Circle((-1.8, 6.3), 0.56, fill=False))


# This is what runs & graphs everything. Population (pop) controls the number of solutions
# per generation to evaluate and pick the best few solutions of, and number of generations
# before termination (gen) controls how many generations or sets of the population is run
# before the algorithm stops.
# TLDR: number of evaluations = pop * gen
run(pop=100, gen=20)
plt.show()
