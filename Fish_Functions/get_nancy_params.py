from collections import namedtuple

VNParams = namedtuple('VNParams', 
    ['N', 'L_total', 'L', 'm', 'Izz', 'g', 'ct', 'cn', 
     'joint_k', 'joint_b', 'G', 'A_group', 'freq', 'C_turn'])

def get_vnancy(N=6, L_total=1.0, m=0.03, ct=0.4, cn=2.0, 
               joint_k=5.0, joint_b=0.5, G=6, A_group=2.0, 
               freq=2.0, C_turn=0.0):
    L = L_total / N
    Izz = (1.0/12.0) * m * L**2
    g = 9.81
    return VNParams(N, L_total, L, m, Izz, g, ct, cn, 
                    joint_k, joint_b, G, A_group, freq, C_turn)
