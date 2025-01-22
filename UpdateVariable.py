from Generate_inital_environment import ResetFunction
from arg_parser import get_args

def updatevalue():
    args = get_args()
    counter = 0
    LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
    user_status, A_u, A_u_ori, A_resource = ResetFunction(args.N, args.L, args.T, args.U)

    def save_variable(new_LEO_status, new_HAPS_status, new_LEO_resource_status, new_HAPS_resource_status,
                      new_user_status, new_A_u, new_A_u_ori, new_A_resource, new_counter):
        nonlocal LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter
        LEO_status = new_LEO_status
        HAPS_status = new_HAPS_status
        LEO_resource_status = new_LEO_resource_status
        HAPS_resource_status = new_HAPS_resource_status
        user_status = new_user_status
        A_u = new_A_u
        A_u_ori  = new_A_u_ori
        A_resource = new_A_resource
        counter = new_counter
        return LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
               user_status, A_u, A_u_ori, A_resource, counter

    def load_variable():
        nonlocal LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter
        return LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter
    return save_variable, load_variable