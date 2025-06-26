import os
import time
import numpy as np
import pandapower as pp

from rl_adn.utility.grid import GridTensor
from rl_adn.utility.utils import create_pandapower_net


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rl_adn', 'data_sources'))
CONFIG = {
    'branch_info_file': os.path.join(ROOT_DIR, 'network_data/node_25', 'Lines_25.csv'),
    'bus_info_file': os.path.join(ROOT_DIR, 'network_data/node_25', 'Nodes_25.csv'),
    'vm_pu': 1.0,
    's_base': 1000,
}

P_FILE = np.array([
    387.09, 0., 387.09, 387.09, 0., 0., 387.09, 387.09,
    0., 387.09, 230.571, 121.176, 121.176, 121.176, 22.7205, 387.09,
    387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
])


def _run_powerflow(config, p_file):
    network = GridTensor(node_file_path=config['bus_info_file'],
                         lines_file_path=config['branch_info_file'])
    network.Q_file = np.zeros(len(p_file))

    start_laurent = time.time()
    solution_laurent = network.run_pf(active_power=p_file)
    time_laurent = time.time() - start_laurent

    net = create_pandapower_net(config)
    for bus_index in net.load.bus.index:
        if bus_index == 0:
            net.load.p_mw[bus_index] = 0
            net.load.q_mvar[bus_index] = 0
        else:
            net.load.p_mw[bus_index] = p_file[bus_index - 1] / 1000
            net.load.q_mvar[bus_index - 1] = 0

    start_panda = time.time()
    pp.runpp(net)
    time_panda = time.time() - start_panda

    v_laurent = solution_laurent["v"]
    v_laurent = np.insert(v_laurent, 0, 1)
    v_laurent_mag = np.abs(v_laurent)

    v_real_panda = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
    v_img_panda = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
    v_panda = v_real_panda + 1j * v_img_panda
    v_panda_mag = np.abs(v_panda)

    error = np.mean(np.abs(v_panda_mag - v_laurent_mag))
    return error, time_laurent, time_panda


def test_powerflow_25_node():
    error, t_laurent, t_panda = _run_powerflow(CONFIG, P_FILE)
    assert error < 1e-3
    assert t_laurent < t_panda
