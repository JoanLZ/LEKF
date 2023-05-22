
# X_list [0...len(X_list)] # all the state simulated 
# X_list[i].p[0] # Of i state trhough simulated, x-coord position
# X_list[i].p[1] # Of i state trhough simulated, y-coord position

def get_X_n_Y (X_list):
    x = [x_i.p[0] for x_i in X_list]
    y = [x_i.p[1] for x_i in X_list]
    return x,y

def get_R (X_list):
    R = [x_i.R.coeffs() for x_i in X_list]
    return R

