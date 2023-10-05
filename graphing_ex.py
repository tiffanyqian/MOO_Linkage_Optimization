import pandas as pd
import matplotlib.pyplot as plt
from main import graph_em

# MAKE SURE YOU'RE IMPORTING THE GRAPH FUNCTION FROM THE RIGHT PYTHON FILE THAT YOU WERE OPTIMIZING FROM

# Every solution saved to the res_log file and also the random string of numbers will be in this format.
# It must be a dataframe of either length 18 or 22. Replace ALL the numbers in brackets to graph.

# EXAMPLE 1) Favorite example that was modeled in CAD. String of numbers I dropped in chat was:
# 0.33256606304546904,1.5707445001710345,0.7057477017560323,1.6035321615525338,1.3742771559144533,
# 3.050202305022688,5.72393689209206,4.80248622340289,0.8269277482433366,1.3790662858808194,4.838867931543022,
# -0.8543692397700136,0.4643171275780114,2.628944608649182,7.4717686976128395,1.6079904606893392,4.91371444599507,
# 2.669632961721123
# Note that this set of strings only has length 18, meaning it doesn't have the transmission angles for positions
# 2 and 3. There's a catch for this that will automatically calculate it, so don't worry
graph_ex_18 = pd.DataFrame({"A_trans_ang_1": [0.33256606304546904], "B_trans_ang_1": [1.5707445001710345],
                            "Beta_2_A": [0.7057477017560323], "Beta_3_A": [1.6035321615525338],
                            "Beta_2_B": [1.3742771559144533], "Beta_3_B": [3.050202305022688],
                            "W_A_x": [5.72393689209206], "W_A_y": [4.80248622340289], "Z_A_x": [0.8269277482433366],
                            "Z_A_y": [1.3790662858808194], "W_B_x": [4.838867931543022], "W_B_y": [-0.8543692397700136],
                            "Z_B_x": [0.4643171275780114], "Z_B_y": [2.628944608649182],
                            "mag_W_A": [7.4717686976128395], "mag_Z_A": [1.6079904606893392],
                            "mag_W_B": [4.91371444599507], "mag_Z_B": [2.669632961721123]})
# This plt.figure is optional, I always add one before to differentiate the plots.
plt.figure(1)
graph_em(graph_ex_18)

# EXAMPLE 2) One example of a 22 length val dataframe where you do have the transmission angles of the other
# positions:
# 0.9971502171152586,0.30931435167789734,0.705472308691022,2.0155132164456617,1.3821313476061323,2.5757661164909442,
# 5.157488996291121,2.8149880122420705,1.4475599339572318,2.6653497585024826,5.66765977808321,0.40014962058223935,
# -0.341102085517529,0.8545060194720291,5.87569998008157,3.0330709021629394,5.681767953635059,0.9200712853134476,
# 0.6540287112080874,0.024223181330041532,2.4415917023083757,1.6953800335605025
graph_ex_22 = pd.DataFrame({"A_trans_ang_1": [0.9971502171152586], "B_trans_ang_1": [0.30931435167789734],
                            "Beta_2_A": [0.7057477017560323], "Beta_3_A": [2.0155132164456617],
                            "Beta_2_B": [1.3821313476061323], "Beta_3_B": [2.5757661164909442],
                            "W_A_x": [5.157488996291121], "W_A_y": [2.8149880122420705], "Z_A_x": [1.4475599339572318],
                            "Z_A_y": [2.6653497585024826], "W_B_x": [5.66765977808321], "W_B_y": [0.40014962058223935],
                            "Z_B_x": [-0.341102085517529], "Z_B_y": [0.8545060194720291],
                            "mag_W_A": [5.87569998008157], "mag_Z_A": [3.0330709021629394],
                            "mag_W_B": [5.681767953635059], "mag_Z_B": [0.9200712853134476],
                            "A_trans_ang_2": [0.6540287112080874], "B_trans_ang_2": [0.024223181330041532],
                            "A_trans_ang_3": [2.4415917023083757], "B_trans_ang_3": [1.6953800335605025]})
plt.figure(2)
graph_em(graph_ex_22)

# EXAMPLE 3) You can even pull from the log file!
# NOTE: obv make sure you have a log file...
res_log = pd.DataFrame(pd.read_csv("res_log.csv"))
if not res_log.empty:
    plt.figure(3)
    # This will graph the first row in the log file. Replace the 0 with any row number you'd like to graph.
    graph_em(pd.DataFrame(res_log.loc[[1]]))

plt.show()
