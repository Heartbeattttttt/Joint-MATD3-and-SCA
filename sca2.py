import numpy as np


def sca_power_control(env, discrete_actions, p_max_mw=200.0, p_init=None, n_iter=15):
    """
    Max-Min Fairness SCA (基于 TCOMM 逻辑)
    """
    active_idxs = np.where(discrete_actions > 0)[0]
    n_active = len(active_idxs)
    if n_active == 0: return np.zeros(env.n_veh)

    # 1. 构建干扰矩阵 (仅同 RSU)
    H = np.zeros((n_active, n_active))
    target_rsus = discrete_actions[active_idxs] - 1

    for i in range(n_active):
        u_tx = active_idxs[i]
        rsu_tx = target_rsus[i]
        for j in range(n_active):
            rsu_rx = target_rsus[j]
            if rsu_tx == rsu_rx:
                H[i, j] = env.g_v2i[u_tx, rsu_rx]
            else:
                H[i, j] = 0.0  # 异 RSU 无干扰

    noise_pwr = float(env.noise_watt)
    v_max = np.sqrt(p_max_mw / 1000.0)

    # 初始化
    if p_init is None:
        v = np.ones(n_active) * v_max * 0.5
    else:
        v = np.sqrt(p_init[active_idxs] / 1000.0)

    # 2. WMMSE 迭代
    for it in range(n_iter):
        p_curr = v ** 2

        # (A) 计算当前速率并动态调整权重 (核心!)
        rx_power = np.dot(p_curr, H) + noise_pwr
        weights = np.zeros(n_active)

        for k in range(n_active):
            sig = H[k, k] * p_curr[k]
            intf = rx_power[k] - sig
            sinr = sig / (intf + 1e-15)
            rate = np.log2(1 + sinr)

            # 【TCOMM 逻辑】: 速率越低，权重越大
            # 迫使算法照顾弱者
            # urgency = 1.0 / (env.vehicles[active_idxs[k]].T_n + 1e-6)
            # weights[k] = urgency * (1.0 / (rate + 0.1)) ** 2
            # [修改后] 只看紧迫程度，不要引入 rate 的反馈回路，防止震荡
            urgency = 1.0 / (env.vehicles[active_idxs[k]].T_n + 1e-6)
            weights[k] = urgency  # 越急的任务，权重越大，简单直接

        weights = weights / (np.mean(weights) + 1e-10)  # 归一化

        # (B) WMMSE 更新
        u = np.zeros(n_active)
        w = np.zeros(n_active)

        # Update u
        for k in range(n_active):
            u[k] = H[k, k] * v[k] / rx_power[k]

        # Update w (包含动态权重)
        for k in range(n_active):
            mse = 1.0 - u[k] * H[k, k] * v[k]
            w[k] = weights[k] / max(mse, 1e-10)

        # Update v
        v_new = np.zeros(n_active)
        for k in range(n_active):
            num = w[k] * u[k] * H[k, k]
            den = 0
            for j in range(n_active):
                den += w[j] * (u[j] ** 2) * H[k, j]
            v_new[k] = num / (den + 1e-15)

        v = np.clip(v_new, 0, v_max)

    p_full = np.zeros(env.n_veh)
    p_full[active_idxs] = (v ** 2) * 1000.0
    return p_full