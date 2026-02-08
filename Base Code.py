import gurobipy as gp
import pickle

class Bin:
    def __init__(self, ID, length, height, cost, a, b):
        self.ID = ID
        self.length = length
        self.height = height
        self.cost = cost
        self.a = a
        self.b = b

class Item:
    def __init__(self, ID, length, height, rotate, fragile, perishable, radioactive):
        self.ID = ID
        self.length = length
        self.height = height
        self.rotate = rotate
        self.fragile = fragile
        self.perishable = perishable
        self.radioactive = radioactive

with open('B.pickle', 'rb') as handle:
    B = pickle.load(handle)
with open('I.pickle', 'rb') as handle:
    I = pickle.load(handle)

B_values = list(B.values())
Bins = []
for b in range(len(B_values)):
    Bins.append(Bin(b, B_values[b][1][0], B_values[b][1][1], B_values[b][1][3], B_values[b][1][4], B_values[b][1][5]))
Lmax = max([bin.length for bin in Bins])

I_values = list(I.values())
Items = []
for i in range(len(I_values)):
    Items.append(Item(i, I_values[i][0], I_values[i][1], I_values[i][2], I_values[i][3], I_values[i][4], I_values[i][5]))


model = gp.Model("Bin Packing Problem")


z = {}
for b in Bins:
    z[b.ID] = model.addVar(obj= b.cost, vtype=gp.GRB.BINARY, name="z_{}".format(b.ID))

I = {}
for i in Items:
    for j in Items:
        if i.ID != j.ID:
            I[i.ID, j.ID] = model.addVar(vtype=gp.GRB.BINARY, name="I_{}_{}".format(i.ID, j.ID))

b = {}
for i in Items:
    for j in Items:
        if i.ID != j.ID:
            b[i.ID, j.ID] = model.addVar(vtype=gp.GRB.BINARY, name="b_{}_{}".format(i.ID, j.ID))

p = {}
for i in Items:
    for bin in Bins:
        p[i.ID, bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="p_{}_{}".format(i.ID, bin.ID))

g = {}
rho = {}
gamma = {}
for i in Items:
    rho[i.ID] = model.addVar(vtype=gp.GRB.BINARY, name="rho_{}".format(i.ID))
    g[i.ID] = model.addVar(vtype=gp.GRB.BINARY, name="g_{}".format(i.ID))
    gamma[i.ID] = model.addVar(vtype=gp.GRB.BINARY, name="gamma_{}".format(i.ID))

radioactive = {}
for bin in Bins:
    radioactive[bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="radioactive_{}".format(bin.ID))

B1 = {}
B2 = {}
for i in Items:
    for j in Items:
        if i.ID != j.ID:
            B1[i.ID, j.ID] = model.addVar(vtype=gp.GRB.BINARY, name="B1_{}_{}".format(i.ID, j.ID))
            B2[i.ID, j.ID] = model.addVar(vtype=gp.GRB.BINARY, name="B2_{}_{}".format(i.ID, j.ID))

x = {}
y = {}
for i in Items:
    x[i.ID] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x_{}".format(i.ID))
    y[i.ID] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="y_{}".format(i.ID))




overlapping = {}
for bin in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                overlapping[bin.ID, i.ID, j.ID] = model.addConstr(I[i.ID, j.ID] + I[j.ID, i.ID] + b[i.ID, j.ID] + b[j.ID, i.ID] >= p[i.ID, bin.ID] + p[j.ID, bin.ID] - 1, name="overlapping_{}_{}_{}".format(bin.ID, i.ID, j.ID))

horizontal1 = {}
horizontal2 = {}

for i in Items:
    for j in Items:
        if i.ID != j.ID:
            horizontal1[i.ID, j.ID] = model.addConstr(x[j.ID] >= x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] - Lmax * (1- I[i.ID, j.ID]), name="horizontal1_{}_{}".format(i.ID, j.ID))
            horizontal2[i.ID, j.ID] = model.addConstr(x[j.ID] + 0.00001 <= x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] + Lmax *  I[j.ID, i.ID], name="horizontal2_{}_{}".format(i.ID, j.ID))

xbound = {}
ybound = {}
assigned = {}
for i in Items:
    xbound[i.ID] = model.addConstr(x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] <= gp.quicksum(bin.length * p[i.ID, bin.ID] for bin in Bins), name="xbound_{}".format(i.ID))
    ybound[i.ID] = model.addConstr(y[i.ID] + i.height * (1 - rho[i.ID]) + i.length * rho[i.ID] <= gp.quicksum(bin.height * p[i.ID, bin.ID] for bin in Bins), name="ybound_{}".format(i.ID))
    assigned[i.ID] = model.addConstr(gp.quicksum(p[i.ID, bin.ID] for bin in Bins) == 1, name="assigned_{}".format(i.ID))

for bin in Bins:
    model.addConstr(gp.quicksum(p[i.ID, bin.ID] for i in Items) <= 1000 * z[bin.ID], name="bin_usage_{}".format(bin.ID))

grounding = {}
for i in Items:
    grounding[i.ID] = model.addConstr(y[i.ID] <= 1000 * (1-g[i.ID]), name="grounding_{}".format(i.ID))

bcon = {}
rcon = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                bcon[i.ID, j.ID, bin.ID] = model.addConstr(y[i.ID] - (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 1000 * (3 - B1[i.ID, j.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="bcon_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                rcon[i.ID, j.ID, bin.ID] = model.addConstr(y[i.ID] - (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 1000 * (3 - B2[i.ID, j.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="rcon_{}_{}_{}".format(i.ID, j.ID, bin.ID))

bcon2 = {}
rcon2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                bcon2[i.ID, j.ID, bin.ID] = model.addConstr(-y[i.ID] + (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 1000 * (3 - B1[i.ID, j.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="bcon2_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                rcon2[i.ID, j.ID, bin.ID] = model.addConstr(-y[i.ID] + (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 1000 * (3 - B2[i.ID, j.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="rcon2_{}_{}_{}".format(i.ID, j.ID, bin.ID))

cornerl = {}
corneru = {}
for i in Items:
    for bin in Bins:
        if bin.b>0:
            cornerl[i.ID, bin.ID] = model.addConstr(y[i.ID]>=-bin.b/bin.a * x[i.ID] + bin.b - (1-p[i.ID, bin.ID])*(1000000), name="corner_{}_{}".format(i.ID, bin.ID))
            corneru[i.ID, bin.ID] = model.addConstr(y[i.ID]<=-bin.b/bin.a * x[i.ID] + bin.b + (2-p[i.ID, bin.ID] - gamma[i.ID])*(1000000), name="corner_{}_{}".format(i.ID, bin.ID))


left = {}
left2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                left[i.ID, j.ID, bin.ID] = model.addConstr(x[j.ID]-x[i.ID] <= 1000000 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B1[i.ID, j.ID]), name="left_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                left2[i.ID, j.ID, bin.ID] = model.addConstr(-x[j.ID]-j.length*(1-rho[j.ID]) - j.height*rho[j.ID] + x[i.ID] <= 1000000 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B1[i.ID, j.ID]), name="left2_{}_{}_{}".format(i.ID, j.ID, bin.ID))

right = {}
right2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                right[i.ID, j.ID, bin.ID] = model.addConstr(x[j.ID]-x[i.ID] - i.length*(1-rho[i.ID]) - i.height*rho[i.ID] <= 1000000 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B2[i.ID, j.ID]), name="right_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                right2[i.ID, j.ID, bin.ID] = model.addConstr(x[i.ID] + i.length*(1-rho[i.ID]) + i.height*rho[i.ID] - x[j.ID] - j.length*(1-rho[j.ID]) - j.height*rho[j.ID] <= 1000000 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B2[i.ID, j.ID]), name="right2_{}_{}_{}".format(i.ID, j.ID, bin.ID))

rotation = {}
for i in Items:
    if i.rotate == 0:
        rotation[i.ID] = model.addConstr(rho[i.ID] == 0, name="rotation_{}".format(i.ID))

fragile = {}
for j in Items:
    if j.fragile == 1:
        fragile[j.ID] = model.addConstr(gp.quicksum(B1[i.ID, j.ID] + B2[i.ID, j.ID] for i in Items if i.ID != j.ID) == 0, name="fragile_{}".format(j.ID))

radiation_activation = {}
radiation_limiter = {}
for bin in Bins:
    radiation_activation[bin.ID] = model.addConstr(gp.quicksum(i.radioactive*p[i.ID, bin.ID] for i in Items) <= 1000 * radioactive[bin.ID], name="radiation_activation_{}".format(bin.ID))
    radiation_limiter[bin.ID] = model.addConstr(gp.quicksum(i.perishable*p[i.ID, bin.ID] for i in Items) <= 1000 * (1-radioactive[bin.ID]), name="radiation_limiter_{}".format(bin.ID))

feasibility = {}
for i in Items:
    feasibility[i.ID] = model.addConstr(gamma[i.ID] + gp.quicksum(B1[i.ID,j.ID] + B2[i.ID,j.ID] for j in Items if j.ID != i.ID) + 2 * g[i.ID] >= 2, name="feasibility_{}".format(i.ID))

model.update()
model.setParam("LogFile", "log_file")
model.update()
model.write("model.lp")
model.optimize()

model.write("model.sol")
print("Optimal value: {}".format(model.objVal))
print("Optimal solution:")
for i in Items:
    for bin in Bins:
        if p[i.ID, bin.ID].X > 0.5:
            print("Item {} is assigned to Bin {}".format(i.ID, bin.ID))
            print("Item {}: x = {}, y = {}, rho = {}".format(i.ID, x[i.ID].X, y[i.ID].X, rho[i.ID].X))
            for j in Items:
                if i.ID != j.ID:
                    if B1[i.ID, j.ID].X > 0.5:
                        print("Item {} is above Item {}".format(i.ID, j.ID))
                    if B2[i.ID, j.ID].X > 0.5:
                        print("Item {} is above Item {}".format(i.ID, j.ID))



import matplotlib.pyplot as plt
import matplotlib.patches as patches

for bin in Bins:
    if z[bin.ID].X > 0.5:
        fig, ax = plt.subplots()
        ax.set_xlim(0, bin.length)
        ax.set_ylim(0, bin.height)
        for i in Items:
            if p[i.ID, bin.ID].X > 0.5:
                rect = patches.Rectangle((x[i.ID].X, y[i.ID].X), i.length * (1 - rho[i.ID].X) + i.height * rho[i.ID].X, i.height * (1 - rho[i.ID].X) + i.length * rho[i.ID].X, edgecolor='black', facecolor='blue', alpha=0.5)
                ax.add_patch(rect)
                plt.text(x[i.ID].X + (i.length * (1 - rho[i.ID].X) + i.height * rho[i.ID].X)/2, y[i.ID].X + (i.height * (1 - rho[i.ID].X) + i.length * rho[i.ID].X)/2, str(i.ID), color='white', ha='center', va='center')
        plt.title("Bin {}".format(bin.ID))
        plt.xlabel("Length")
        plt.ylabel("Height")
        plt.grid()
        plt.show()