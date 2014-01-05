# Gotran generated code for: vanbeek_model_2007

from __future__ import division

def init_values(**values):
    """
    Init values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Init values
    # ATP_cyt=5912.77, ADP_cyt=64, PCr_cyt=5000, Cr_cyt=10500, Pi_cyt=913,
    # ATP_ims=5912.77, ADP_ims=39, PCr_ims=5000, Cr_ims=10500,
    # Pi_ims=910
    init_values = np.array([5912.77, 64, 5000, 10500, 913, 5912.77, 39, 5000,\
        10500, 910], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict(ATP_cyt=(0, Range()), ADP_cyt=(1, Range()), PCr_cyt=(2,\
        Range()), Cr_cyt=(3, Range()), Pi_cyt=(4, Range()), ATP_ims=(5,\
        Range()), ADP_ims=(6, Range()), PCr_ims=(7, Range()), Cr_ims=(8,\
        Range()), Pi_ims=(9, Range()))

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{{0}} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value

    return init_values

def default_parameters(**values):
    """
    Parameter values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Param values
    # Kb=15500.0, Kd=1670.0, Kia=900.0, Kib=34900.0, Kic=222.4, Kid=4730.0,
    # Vmax_MM_b=48040.0, Vmax_MM_f=11440.0, Kb_J=5200.0, Kd_J=500.0,
    # Kia_J=750.0, Kib_J=28800.0, Kic_J=204.8, Kid_J=1600.0,
    # Vmax_Mi_b=3704.0, Vmax_Mi_f=882.0, J_hyd_basis_1=486.5,
    # J_hyd_basis_2=627.6, freq_1=135, freq_2=220, nb_of_cycles_1=5,
    # KADP=25, KPi=800.0, V_max_syn=15040.0, PS_tot_ATP=13.3,
    # PS_tot_ADP=13.3, PS_tot_PCr=155.0, PS_tot_Cr=155.0,
    # PS_tot_Pi=194.0, V_cyt=0.75, V_ims=0.0625
    param_values = np.array([15500.0, 1670.0, 900.0, 34900.0, 222.4, 4730.0,\
        48040.0, 11440.0, 5200.0, 500.0, 750.0, 28800.0, 204.8, 1600.0,\
        3704.0, 882.0, 486.5, 627.6, 135, 220, 5, 25, 800.0, 15040.0, 13.3,\
        13.3, 155.0, 155.0, 194.0, 0.75, 0.0625], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict(Kb=(0, Range()), Kd=(1, Range()), Kia=(2, Range()),\
        Kib=(3, Range()), Kic=(4, Range()), Kid=(5, Range()), Vmax_MM_b=(6,\
        Range()), Vmax_MM_f=(7, Range()), Kb_J=(8, Range()), Kd_J=(9,\
        Range()), Kia_J=(10, Range()), Kib_J=(11, Range()), Kic_J=(12,\
        Range()), Kid_J=(13, Range()), Vmax_Mi_b=(14, Range()),\
        Vmax_Mi_f=(15, Range()), J_hyd_basis_1=(16, Range()),\
        J_hyd_basis_2=(17, Range()), freq_1=(18, Range()), freq_2=(19,\
        Range()), nb_of_cycles_1=(20, Range()), KADP=(21, Range()), KPi=(22,\
        Range()), V_max_syn=(23, Range()), PS_tot_ATP=(24, Range()),\
        PS_tot_ADP=(25, Range()), PS_tot_PCr=(26, Range()), PS_tot_Cr=(27,\
        Range()), PS_tot_Pi=(28, Range()), V_cyt=(29, Range()), V_ims=(30,\
        Range()))

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{{0}} is not a param".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,\
                range.format_not_in(value)))

        # Assign value
        param_values[ind] = value

    return param_values

def rhs(states, time, parameters, dy=None):
    """
    Compute right hand side
    """
    # Imports
    import numpy as np
    import math
    from math import pow, sqrt, log

    # Assign states
    assert(len(states) == 10)
    ATP_cyt, ADP_cyt, PCr_cyt, Cr_cyt, Pi_cyt, ATP_ims, ADP_ims, PCr_ims,\
        Cr_ims, Pi_ims = states

    # Assign parameters
    assert(len(parameters) == 31)
    Kb, Kd, Kia, Kib, Kic, Kid, Vmax_MM_b, Vmax_MM_f, Kb_J, Kd_J, Kia_J,\
        Kib_J, Kic_J, Kid_J, Vmax_Mi_b, Vmax_Mi_f, J_hyd_basis_1,\
        J_hyd_basis_2, freq_1, freq_2, nb_of_cycles_1, KADP, KPi, V_max_syn,\
        PS_tot_ATP, PS_tot_ADP, PS_tot_PCr, PS_tot_Cr, PS_tot_Pi, V_cyt,\
        V_ims = parameters

    # J ckmm
    Kc = Kd*Kic/Kid
    KIb = Kib
    Den_MMCK = 1.0 + (Cr_cyt/(KIb*Kic) + 1.0/Kic + PCr_cyt/(Kc*Kid))*ADP_cyt\
        + (Cr_cyt/(Kb*Kia) + 1.0/Kia)*ATP_cyt + PCr_cyt/Kid + Cr_cyt/Kib
    J_CKMM = (-ADP_cyt*PCr_cyt*Vmax_MM_b/(Kd*Kic) +\
        ATP_cyt*Cr_cyt*Vmax_MM_f/(Kb*Kia))/Den_MMCK

    # J ckmi
    Kc = Kd_J*Kic_J/Kid_J
    KIb = Kib_J
    Den_MiCK = 1.0 + (PCr_ims/(Kc*Kid_J) + Cr_ims/(KIb*Kic_J) +\
        1.0/Kic_J)*ADP_ims + Cr_ims/Kib_J + PCr_ims/Kid_J +\
        (Cr_ims/(Kb_J*Kia_J) + 1.0/Kia_J)*ATP_ims
    J_CKMi = (ATP_ims*Cr_ims*Vmax_Mi_f/(Kb_J*Kia_J) -\
        ADP_ims*PCr_ims*Vmax_Mi_b/(Kd_J*Kic_J))/Den_MiCK

    # J hyd
    t_cycle_1 = 60.0/freq_1
    t_cycle_2 = 60.0/freq_2
    duration_1 = nb_of_cycles_1*t_cycle_1
    ltime = (time - t_cycle_1*math.floor(time/t_cycle_1) if time <=\
        duration_1 else -t_cycle_2*math.floor((-duration_1 + time)/t_cycle_2)\
        - duration_1 + time)
    t_cycle = (t_cycle_1 if time <= duration_1 else t_cycle_2)
    H_ATPmax = (6.0*J_hyd_basis_1 if time <= duration_1 else 6.0*J_hyd_basis_2)
    J_hyd = (6.0*H_ATPmax*ltime/t_cycle if (ltime < t_cycle/6.0) and (ltime\
        >= 0.0) else ((2.0 - 6.0*ltime/t_cycle)*H_ATPmax if (ltime <\
        t_cycle/3.0) and (ltime >= t_cycle/6.0) else 0.0))

    # J syn
    Den_syn = 1.0 + ADP_ims*Pi_ims/(KADP*KPi) + Pi_ims/KPi + ADP_ims/KADP
    J_syn = ADP_ims*Pi_ims*V_max_syn/(Den_syn*KADP*KPi)

    # J diff atp
    J_diff_ATP = (ATP_ims - ATP_cyt)*PS_tot_ATP

    # J diff adp
    J_diff_ADP = (-ADP_cyt + ADP_ims)*PS_tot_ADP

    # J diff pcr
    J_diff_PCr = (-PCr_cyt + PCr_ims)*PS_tot_PCr

    # J diff cr
    J_diff_Cr = (-Cr_cyt + Cr_ims)*PS_tot_Cr

    # J diff pi
    J_diff_Pi = (-Pi_cyt + Pi_ims)*PS_tot_Pi

    # Atp cyt

    # Adp cyt

    # Pcr cyt

    # Cr cyt

    # Pi cyt

    # Atp ims

    # Adp ims

    # Pcr ims

    # Cr ims

    # Pi ims

    # The ODE system: 10 states

    # Init dy
    if dy is None:
        dy = np.zeros_like(states)
    dy[0] = (J_diff_ATP - J_CKMM - J_hyd)/V_cyt
    dy[1] = (J_CKMM + J_diff_ADP + J_hyd)/V_cyt
    dy[2] = (J_diff_PCr + J_CKMM)/V_cyt
    dy[3] = (J_diff_Cr - J_CKMM)/V_cyt
    dy[4] = (J_hyd + J_diff_Pi)/V_cyt
    dy[5] = (J_syn - J_diff_ATP - J_CKMi)/V_ims
    dy[6] = (J_CKMi - J_syn - J_diff_ADP)/V_ims
    dy[7] = (J_CKMi - J_diff_PCr)/V_ims
    dy[8] = (-J_diff_Cr - J_CKMi)/V_ims
    dy[9] = (-J_diff_Pi - J_syn)/V_ims

    # Return dy
    return dy

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict(ATP_cyt=0, ADP_cyt=1, PCr_cyt=2, Cr_cyt=3, Pi_cyt=4,\
        ATP_ims=5, ADP_ims=6, PCr_ims=7, Cr_ims=8, Pi_ims=9)

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    return indices if len(indices)>1 else indices[0]

def param_indices(*params):
    """
    Param indices
    """
    param_inds = dict(Kb=0, Kd=1, Kia=2, Kib=3, Kic=4, Kid=5, Vmax_MM_b=6,\
        Vmax_MM_f=7, Kb_J=8, Kd_J=9, Kia_J=10, Kib_J=11, Kic_J=12, Kid_J=13,\
        Vmax_Mi_b=14, Vmax_Mi_f=15, J_hyd_basis_1=16, J_hyd_basis_2=17,\
        freq_1=18, freq_2=19, nb_of_cycles_1=20, KADP=21, KPi=22,\
        V_max_syn=23, PS_tot_ATP=24, PS_tot_ADP=25, PS_tot_PCr=26,\
        PS_tot_Cr=27, PS_tot_Pi=28, V_cyt=29, V_ims=30)

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    return indices if len(indices)>1 else indices[0]

def monitor(states, time, parameters, monitored=None):
    """
    Compute monitored intermediates
    """
    # Imports
    import numpy as np
    import math
    from math import pow, sqrt, log

    # Assign states
    assert(len(states) == 10)
    ATP_cyt, ADP_cyt, PCr_cyt, Cr_cyt, Pi_cyt, ATP_ims, ADP_ims, PCr_ims,\
        Cr_ims, Pi_ims = states[0], states[1], states[2], states[3],\
        states[4], states[5], states[6], states[7], states[8], states[9]

    # Assign parameters
    assert(len(parameters) == 31)
    Kb, Kd, Kia, Kib, Kic, Kid, Vmax_MM_b, Vmax_MM_f, Kb_J, Kd_J, Kia_J,\
        Kib_J, Kic_J, Kid_J, Vmax_Mi_b, Vmax_Mi_f, J_hyd_basis_1,\
        J_hyd_basis_2, freq_1, freq_2, nb_of_cycles_1, KADP, KPi, V_max_syn,\
        PS_tot_ATP, PS_tot_ADP, PS_tot_PCr, PS_tot_Cr, PS_tot_Pi =\
        parameters[0], parameters[1], parameters[2], parameters[3],\
        parameters[4], parameters[5], parameters[6], parameters[7],\
        parameters[8], parameters[9], parameters[10], parameters[11],\
        parameters[12], parameters[13], parameters[14], parameters[15],\
        parameters[16], parameters[17], parameters[18], parameters[19],\
        parameters[20], parameters[21], parameters[22], parameters[23],\
        parameters[24], parameters[25], parameters[26], parameters[27],\
        parameters[28]

    # Common Sub Expressions for monitored intermediates
    cse_monitored_0 = -60.0*nb_of_cycles_1/freq_1 + time
    cse_monitored_1 = time <= 60.0*nb_of_cycles_1/freq_1
    cse_monitored_2 = (6.0*J_hyd_basis_1 if cse_monitored_1 else\
        6.0*J_hyd_basis_2)
    cse_monitored_3 = (60.0/freq_1 if cse_monitored_1 else 60.0/freq_2)
    cse_monitored_4 = (-60.0*math.floor(freq_1*time/60.0)/freq_1 + time if\
        cse_monitored_1 else cse_monitored_0 -\
        60.0*math.floor(cse_monitored_0*freq_2/60.0)/freq_2)
    cse_monitored_5 = -ADP_cyt
    cse_monitored_6 = ADP_ims/KADP
    cse_monitored_7 = Cr_cyt/Kib
    cse_monitored_8 = Cr_ims/Kib_J
    cse_monitored_9 = Pi_ims/KPi
    cse_monitored_10 = Cr_cyt/(Kb*Kia)
    cse_monitored_11 = Cr_ims/(Kb_J*Kia_J)
    cse_monitored_12 = PCr_cyt/(Kd*Kic)
    cse_monitored_13 = PCr_ims/(Kd_J*Kic_J)
    cse_monitored_14 = cse_monitored_3/6.0
    cse_monitored_15 = 6.0*cse_monitored_4*(freq_1/60.0 if cse_monitored_1 else\
        freq_2/60.0)

    # Init monitored
    if monitored is None:
        monitored = np.zeros(9, dtype=np.float_)

    # Monitored intermediates
    monitored[0] = (-cse_monitored_13*ADP_ims*Vmax_Mi_b +\
        cse_monitored_11*ATP_ims*Vmax_Mi_f)/(1.0 + (cse_monitored_13 +\
        1.0/Kic_J + cse_monitored_8/Kic_J)*ADP_ims + PCr_ims/Kid_J +\
        cse_monitored_8 + (cse_monitored_11 + 1.0/Kia_J)*ATP_ims)
    monitored[1] = cse_monitored_6*cse_monitored_9*V_max_syn/(1.0 +\
        cse_monitored_6*cse_monitored_9 + cse_monitored_9 + cse_monitored_6)
    monitored[2] = (-PCr_cyt + PCr_ims)*PS_tot_PCr
    monitored[3] = (-Cr_cyt + Cr_ims)*PS_tot_Cr
    monitored[4] = (-Pi_cyt + Pi_ims)*PS_tot_Pi
    monitored[5] = (cse_monitored_10*ATP_cyt*Vmax_MM_f +\
        cse_monitored_12*cse_monitored_5*Vmax_MM_b)/(1.0 + (cse_monitored_12 +\
        1.0/Kic + cse_monitored_7/Kic)*ADP_cyt + (cse_monitored_10 +\
        1.0/Kia)*ATP_cyt + cse_monitored_7 + PCr_cyt/Kid)
    monitored[6] = (ATP_ims - ATP_cyt)*PS_tot_ATP
    monitored[7] = (cse_monitored_5 + ADP_ims)*PS_tot_ADP
    monitored[8] = (cse_monitored_15*cse_monitored_2 if (cse_monitored_4 <\
        cse_monitored_14) and (cse_monitored_4 >= 0.0) else\
        (cse_monitored_2*(2.0 - cse_monitored_15) if (cse_monitored_4 <\
        cse_monitored_3/3.0) and (cse_monitored_4 >= cse_monitored_14) else\
        0.0))

    # Return monitored
    return monitored

