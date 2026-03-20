def mach_represents_area(Ma,gamma=1.4):
    """通过马赫数计算面积比A/A*"""
    term1=1/Ma
    term2=2/(gamma+1)*(1+(gamma-1)/2*Ma**2)
    exponent=0.5*(gamma+1)/(gamma-1)
    return term1*term2**exponent

def mach_solved_from_area(A_ratio,gamma=1.4,subsonic=True):
    """通过面积比A/A*反解马赫数"""
    from scipy.optimize import fsolve#可以在哪里用到就在哪里引入
    if subsonic:#亚声速
        Ma_guess=0.5
        Ma=fsolve(lambda Ma:mach_represents_area(Ma,gamma)-A_ratio,Ma_guess)[0]#迭代
    else:#超声速
        Ma_guess=2.0
        Ma=fsolve(lambda Ma:mach_represents_area(Ma,gamma)-A_ratio,Ma_guess)[0]
    return Ma

def mach_represents_T_rho_p_ratio(Ma,gamma=1.4):
    """通过马赫数计算等熵流动关系式T/T0、ρ/ρ0、p/p0"""
    shared_term=1+(gamma-1)/2*Ma**2
    T_ratio=1/shared_term
    rho_ratio=shared_term**(-1/(gamma-1))
    p_ratio=shared_term**(-gamma/(gamma-1))
    return rho_ratio,T_ratio,p_ratio#接收顺序都反了！！！






