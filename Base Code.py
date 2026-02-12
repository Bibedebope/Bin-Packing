import gurobipy as gp
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
Hmax = max([bin.height for bin in Bins])

I_values = list(I.values())
Items = []
for i in range(len(I_values)):
        Items.append(Item(i, I_values[i][0], I_values[i][1], I_values[i][2], I_values[i][3], I_values[i][4], I_values[i][5]))

print("Items:")
for item in Items:
    print("ID: {}, Length: {}, Height: {}, Rotate: {}, Fragile: {}, Perishable: {}, Radioactive: {}".format(item.ID, item.length, item.height, item.rotate, item.fragile, item.perishable, item.radioactive))

model = gp.Model("Bin Packing Problem")


z = {}
for b in Bins:
    z[b.ID] = model.addVar(obj= b.cost, vtype=gp.GRB.BINARY, name="z_{}".format(b.ID))

I = {}
for bn in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                I[i.ID, j.ID, bn.ID] = model.addVar(vtype=gp.GRB.BINARY, name="I_{}_{}_{}".format(i.ID, j.ID, bn.ID))
b = {}
for bn in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                b[i.ID, j.ID, bn.ID] = model.addVar(vtype=gp.GRB.BINARY, name="b_{}_{}_{}".format(i.ID, j.ID, bn.ID))

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
    for bin in Bins:
        gamma[i.ID, bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="gamma_{}_{}".format(i.ID, bin.ID))

radioactive = {}
for bin in Bins:
    radioactive[bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="radioactive_{}".format(bin.ID))

B1 = {}
B2 = {}
for bin in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                B1[i.ID, j.ID, bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="B1_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                B2[i.ID, j.ID, bin.ID] = model.addVar(vtype=gp.GRB.BINARY, name="B2_{}_{}_{}".format(i.ID, j.ID, bin.ID))

x = {}
y = {}
for i in Items:
    x[i.ID] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x_{}".format(i.ID))
    y[i.ID] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="y_{}".format(i.ID))




overlapping = {}
for bin in Bins:
    for i in Items:
        for j in Items:
            if i.ID < j.ID:
                overlapping[bin.ID, i.ID, j.ID] = model.addConstr(I[i.ID, j.ID, bin.ID] + I[j.ID, i.ID, bin.ID] + b[i.ID, j.ID, bin.ID] + b[j.ID, i.ID, bin.ID] >= p[i.ID, bin.ID] + p[j.ID, bin.ID] - 1, name="overlapping_{}_{}_{}".format(bin.ID, i.ID, j.ID))

horizontal1 = {}
horizontal2 = {}

for bn in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                horizontal1[i.ID, j.ID, bn.ID] = model.addConstr(x[j.ID] >= x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] - Lmax * (3 - p[i.ID, bn.ID] - p[j.ID, bn.ID] - I[i.ID, j.ID, bn.ID]), name="horizontal1_{}_{}_{}".format(i.ID, j.ID, bn.ID))
                horizontal2[i.ID, j.ID, bn.ID] = model.addConstr(x[j.ID] + 0.00001 <= x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] + Lmax * (2 - p[i.ID, bn.ID] - p[j.ID, bn.ID] + I[i.ID, j.ID, bn.ID]), name="horizontal2_{}_{}_{}".format(i.ID, j.ID, bn.ID))

vertical1 = {}
vertical2 = {}
for bn in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                vertical1[i.ID, j.ID, bn.ID] = model.addConstr(y[j.ID] >= y[i.ID] + i.height * (1 - rho[i.ID]) + i.length * rho[i.ID] - Hmax * (3- p[i.ID, bn.ID] - p[j.ID, bn.ID] - b[i.ID, j.ID, bn.ID]), name="vertical1_{}_{}_{}".format(i.ID, j.ID, bn.ID))
                vertical2[i.ID, j.ID, bn.ID] = model.addConstr(y[j.ID] <= y[i.ID] + i.height * (1 - rho[i.ID]) + i.length * rho[i.ID] + Hmax *  (2 - p[i.ID, bn.ID] - p[j.ID, bn.ID] + b[i.ID, j.ID, bn.ID]), name="vertical2_{}_{}_{}".format(i.ID, j.ID, bn.ID))

xbound = {}
ybound = {}
assigned = {}
for i in Items:
    xbound[i.ID] = model.addConstr(x[i.ID] + i.length * (1 - rho[i.ID]) + i.height * rho[i.ID] <= gp.quicksum(bin.length * p[i.ID, bin.ID] for bin in Bins), name="xbound_{}".format(i.ID))
    ybound[i.ID] = model.addConstr(y[i.ID] + i.height * (1 - rho[i.ID]) + i.length * rho[i.ID] <= gp.quicksum(bin.height * p[i.ID, bin.ID] for bin in Bins), name="ybound_{}".format(i.ID))
    assigned[i.ID] = model.addConstr(gp.quicksum(p[i.ID, bin.ID] for bin in Bins) == 1, name="assigned_{}".format(i.ID))

for bin in Bins:
    model.addConstr(gp.quicksum(p[i.ID, bin.ID] for i in Items) <= 24 * z[bin.ID], name="bin_usage_{}".format(bin.ID))

grounding = {}
for i in Items:
    grounding[i.ID] = model.addConstr(y[i.ID] <= 156 * (1-g[i.ID]), name="grounding_{}".format(i.ID))

bcon = {}
rcon = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                bcon[i.ID, j.ID, bin.ID] = model.addConstr(y[i.ID] - (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 156 * (3 - B1[i.ID, j.ID, bin.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="bcon_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                rcon[i.ID, j.ID, bin.ID] = model.addConstr(y[i.ID] - (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 156 * (3 - B2[i.ID, j.ID, bin.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="rcon_{}_{}_{}".format(i.ID, j.ID, bin.ID))

bcon2 = {}
rcon2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                bcon2[i.ID, j.ID, bin.ID] = model.addConstr(-y[i.ID] + (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 156 * (3 - B1[i.ID, j.ID, bin.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="bcon2_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                rcon2[i.ID, j.ID, bin.ID] = model.addConstr(-y[i.ID] + (y[j.ID] + j.height * (1 - rho[j.ID]) + j.length * rho[j.ID]) <= 156 * (3 - B2[i.ID, j.ID, bin.ID] - p[i.ID, bin.ID] - p[j.ID, bin.ID]), name="rcon2_{}_{}_{}".format(i.ID, j.ID, bin.ID))

cornerl = {}
corneru = {}
for i in Items:
    for bin in Bins:
        if bin.b>0:
            cornerl[i.ID, bin.ID] = model.addConstr(y[i.ID]>=-bin.b/bin.a * x[i.ID] + bin.b - (1-p[i.ID, bin.ID])*(156), name="corner_{}_{}".format(i.ID, bin.ID))
            corneru[i.ID, bin.ID] = model.addConstr(y[i.ID]<=-bin.b/bin.a * x[i.ID] + bin.b + (2-p[i.ID, bin.ID] - gamma[i.ID, bin.ID])*(156), name="corner_{}_{}".format(i.ID, bin.ID))
        else:
            model.addConstr(gamma[i.ID, bin.ID] == 0, name="gamma_{}_{}".format(i.ID, bin.ID))
        model.addConstr(gamma[i.ID, bin.ID] <= p[i.ID, bin.ID], name="gamma_p_{}_{}".format(i.ID, bin.ID))


left = {}
left2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                left[i.ID, j.ID, bin.ID] = model.addConstr(x[j.ID]-x[i.ID] <= 301 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B1[i.ID, j.ID, bin.ID]), name="left_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                left2[i.ID, j.ID, bin.ID] = model.addConstr(-x[j.ID]-j.length*(1-rho[j.ID]) - j.height*rho[j.ID] + x[i.ID]  + 0.00001 <= 301 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B1[i.ID, j.ID, bin.ID]), name="left2_{}_{}_{}".format(i.ID, j.ID, bin.ID))
    model.addConstr(gp.quicksum(gp.quicksum(B1[i.ID, j.ID, bin.ID] for j in Items if j.ID != i.ID) for bin in Bins) <= 1, name="left_b_{}".format(i.ID))

right = {}
right2 = {}
for i in Items:
    for j in Items:
        for bin in Bins:
            if i.ID != j.ID:
                right[i.ID, j.ID, bin.ID] = model.addConstr(x[j.ID]  + 0.00001 -x[i.ID] - i.length*(1-rho[i.ID]) - i.height*rho[i.ID] <= 301 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B2[i.ID, j.ID, bin.ID]), name="right_{}_{}_{}".format(i.ID, j.ID, bin.ID))
                right2[i.ID, j.ID, bin.ID] = model.addConstr(x[i.ID] + i.length*(1-rho[i.ID]) + i.height*rho[i.ID] - x[j.ID] - j.length*(1-rho[j.ID]) - j.height*rho[j.ID] <= 301 * (3-p[i.ID, bin.ID] - p[j.ID, bin.ID] - B2[i.ID, j.ID, bin.ID]), name="right2_{}_{}_{}".format(i.ID, j.ID, bin.ID))
    model.addConstr(gp.quicksum(gp.quicksum(B2[i.ID, j.ID, bin.ID] for j in Items if j.ID != i.ID) for bin in Bins) <= 1, name="right_b_{}".format(i.ID))

for bin in Bins:
    for i in Items:
        for j in Items:
            if i.ID != j.ID:
                model.addConstr(B1[i.ID, j.ID, bin.ID] <= p[i.ID, bin.ID])
                model.addConstr(B1[i.ID, j.ID, bin.ID] <= p[j.ID, bin.ID])
                model.addConstr(B2[i.ID, j.ID, bin.ID] <= p[i.ID, bin.ID])
                model.addConstr(B2[i.ID, j.ID, bin.ID] <= p[j.ID, bin.ID])



rotation = {}
for i in Items:
    if i.rotate == 0:
        rotation[i.ID] = model.addConstr(rho[i.ID] == 0, name="rotation_{}".format(i.ID))

fragile = {}
for j in Items:
    if j.fragile == 1:
        fragile[j.ID] = model.addConstr(gp.quicksum(gp.quicksum(B1[i.ID, j.ID, bin.ID] + B2[i.ID, j.ID, bin.ID] for i in Items if i.ID != j.ID) for bin in Bins) == 0, name="fragile_{}".format(j.ID))

radiation_activation = {}
radiation_limiter = {}
for bin in Bins:
    radiation_activation[bin.ID] = model.addConstr(gp.quicksum(i.radioactive*p[i.ID, bin.ID] for i in Items) <= 25 * radioactive[bin.ID], name="radiation_activation_{}".format(bin.ID))
    radiation_limiter[bin.ID] = model.addConstr(gp.quicksum(i.perishable*p[i.ID, bin.ID] for i in Items) <= 25 * (1-radioactive[bin.ID]), name="radiation_limiter_{}".format(bin.ID))

feasibility = {}
for i in Items:
    feasibility[i.ID] = model.addConstr(gp.quicksum(gamma[i.ID, bin.ID] for bin in Bins) + gp.quicksum(gp.quicksum(B1[i.ID,j.ID,bin.ID] + B2[i.ID,j.ID,bin.ID] for j in Items if j.ID != i.ID) for bin in Bins) + 2 * g[i.ID] >= 2, name="feasibility_{}".format(i.ID))

# for zz in range(2):
#     model.addConstr(z[zz] >= z[zz+1], name="bin_selection_{}".format(zz))



model.update()
model.setParam("LogFile", "log_file")
model.update()
model.write("model.lp")
model.setParam("TimeLimit", 60*60*2)
model.optimize()

model.write("model.sol")


output_dir = "bin_plots"
os.makedirs(output_dir, exist_ok=True)

def item_color(item_id):
    cmap = plt.get_cmap("tab20")
    return cmap(item_id % cmap.N)

for bin in Bins:
    if z[bin.ID].X > 0.5:
        fig, ax = plt.subplots()
        ax.set_xlim(0, bin.length)
        ax.set_ylim(0, bin.height)

        # ---- draw the cut-out line if present ----
        if getattr(bin, "a", 0) > 0:
            ax.plot([bin.a, 0], [0, bin.b], linewidth=2)

        for i in Items:
            if p[i.ID, bin.ID].X > 0.5:
                w = i.length * (1 - rho[i.ID].X) + i.height * rho[i.ID].X
                h = i.height * (1 - rho[i.ID].X) + i.length * rho[i.ID].X

                rect = patches.Rectangle(
                    (x[i.ID].X, y[i.ID].X),
                    w, h,
                    edgecolor='black',
                    facecolor=item_color(i.ID),
                    alpha=0.55
                )
                ax.add_patch(rect)

                ax.text(
                    x[i.ID].X + w / 2,
                    y[i.ID].X + h / 2,
                    str(i.ID),
                    color='white',
                    ha='center',
                    va='center'
                )

        ax.set_title(f"Bin {bin.ID}")
        ax.set_xlabel("Length")
        ax.set_ylabel("Height")
        ax.grid(True)

        # ---------- SAVE FIGURE ----------
        filename = os.path.join(output_dir, f"bin_{bin.ID}.png")
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)   # IMPORTANT: prevents memory leaks
        print(f"Saved {filename}")

plt.show()