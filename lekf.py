class LEKF:
    # Constructor
    def __init__(self) -> None:
        pass

    # User function
    def predict(self, u):
        pass
    
    def correct(self, y):
        pass

    # Dynamics of system

    def f(self, x, u, r):

    def True_acceleration (X,a_m):
    
    g = np.array([0,0,9.81])

    da = a_m-X.a_b # da = a_m - a_b. Difference between the measured acceleration value from IMU and accel. bias. Its a R3Tangent().
    da = SO3Tangent(da) # Like this we define the diference as S03Tangent() group. Its useful to use the attribut .coeffs_copy
    
    J_da_am = np.identity([3,3]) # Jacobian of the differnce wrt a_m
    J_da_ab = -np.identity([3,3]) # Jacobian of the differnce wrt a_b

    a = X.R.act(da.coeffs_copy()) + g # a = R * ( da ) + g = a = R * ( a_m - a_b ) + g. The result is a R3 vector.

    J_a_R = -X.R.act(da.hat()) # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
    J_a_am = X.R # J_a_am = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_am
    J_a_ab = -J_a_am # J_a_ab = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_ab

def True_omega(X,w_m):
    
    w = w_m - X.wb # w = w_m - w_b defining wmeasured and wbias as R3, if we use -, we get a SO3Tangent().
    J_w_wm = np.identity((3,3))
    J_w_wb = -np.identity((3,3))

    # Observation system