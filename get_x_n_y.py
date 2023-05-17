
# X_list [0...len(X_list)] # all the state simulated 
# X_list[i].p[0] # Of i state trhough simulated, x-coord position
# X_list[i].p[1] # Of i state trhough simulated, y-coord position

x = []
y = []

def get_X_n_Y (X_list):
    for i in range(0,len(X_list)-1):
        x.append(X_list[i].p[0])
        y.append(X_list[i].p[1])
    return x,y


