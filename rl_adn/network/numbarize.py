"""Numerical kernels used by the Laurent and SAM power-flow solvers."""

from __future__ import annotations

import numpy as np
from numba import prange


def pre_power_flow_sam_sequential(active_power, reactive_power, s_base, alpha_z, alpha_i, yds, ydd, node_count):
    """Precompute SAM sequential matrices for ZIP-style load models."""
    active_power_pu = active_power / s_base
    reactive_power_pu = reactive_power / s_base
    nominal_power = (active_power_pu + 1j * reactive_power_pu).reshape(-1)

    if alpha_z.shape != nominal_power.shape:
        raise ValueError("alpha_z must match the flattened nominal power shape")
    if alpha_i.shape != nominal_power.shape:
        raise ValueError("alpha_i must match the flattened nominal power shape")

    if not np.any(alpha_z):
        inverse_matrix_b = np.linalg.inv(ydd)
    else:
        matrix_b = np.diag(np.multiply(alpha_z, np.conj(nominal_power))) + ydd
        inverse_matrix_b = np.linalg.inv(matrix_b)

    if not np.any(alpha_i):
        matrix_c = yds
    else:
        matrix_c = yds + np.multiply(alpha_i, np.conj(nominal_power)).reshape(node_count - 1, 1)

    return inverse_matrix_b, matrix_c, nominal_power


def power_flow_sam_sequential(
    inverse_matrix_b,
    matrix_c,
    voltage_guess,
    nominal_power,
    alpha_p,
    iterations,
    tolerance,
):
    """Run the sequential SAM fixed-point update for general ZIP loads."""
    iteration = 0
    voltage_delta = np.inf
    while (iteration < iterations) & (voltage_delta >= tolerance):
        matrix_a = np.diag(alpha_p * np.reciprocal(np.conj(voltage_guess) ** 2) * np.conj(nominal_power))
        vector_d = 2 * alpha_p * np.reciprocal(np.conj(voltage_guess)) * np.conj(nominal_power)

        voltage_solution = inverse_matrix_b @ (matrix_a @ np.conj(voltage_guess) - matrix_c - vector_d)
        voltage_delta = np.max(np.abs(np.abs(voltage_solution) - np.abs(voltage_guess)))
        voltage_guess = voltage_solution
        iteration += 1

    return voltage_guess, iteration


def power_flow_sam_sequential_constant_power_only(
    inverse_matrix_b,
    matrix_c,
    voltage_guess,
    nominal_power,
    iterations,
    tolerance,
):
    """Run the sequential SAM update for constant-power loads only."""
    iteration = 0
    voltage_delta = np.inf
    while (iteration < iterations) & (voltage_delta >= tolerance):
        matrix_a = np.diag(np.reciprocal(np.conj(voltage_guess) ** 2) * np.conj(nominal_power))
        vector_d = 2 * np.reciprocal(np.conj(voltage_guess)) * np.conj(nominal_power)

        voltage_solution = inverse_matrix_b @ (matrix_a @ np.conj(voltage_guess) - matrix_c - vector_d)
        voltage_delta = np.max(np.abs(np.abs(voltage_solution) - np.abs(voltage_guess)))
        voltage_guess = voltage_solution
        iteration += 1

    return voltage_guess, iteration


def pre_power_flow_tensor(
    all_constant_impedance_zero,
    all_constant_current_zero,
    all_constant_power_one,
    time_steps,
    node_count,
    nominal_power,
    alpha_z,
    alpha_i,
    alpha_p,
    yds,
    ydd,
):
    """Precompute tensor fixed-point factors for batched Laurent solves."""
    if not all_constant_impedance_zero:
        alpha_z_power = np.multiply(np.conj(nominal_power), alpha_z)
    else:
        alpha_z_power = np.zeros((time_steps, node_count - 1))

    if not all_constant_current_zero:
        alpha_i_power = np.multiply(np.conj(nominal_power), alpha_i)
    else:
        alpha_i_power = np.zeros((time_steps, node_count - 1))

    if all_constant_power_one:
        alpha_p_power = np.conj(nominal_power)
    else:
        alpha_p_power = np.multiply(np.conj(nominal_power), alpha_p)

    inverse_matrix_b = np.zeros((time_steps, node_count - 1, node_count - 1), dtype="complex128")
    tensor_factor_matrix = np.zeros((time_steps, node_count - 1, node_count - 1), dtype="complex128")
    tensor_bias_vector = np.zeros((time_steps, node_count - 1), dtype="complex128")
    matrix_c = alpha_i_power + yds.reshape(-1)

    for index in prange(time_steps):
        inverse_matrix_b[index] = np.linalg.inv(np.diag(alpha_z_power[index]) + ydd)
        tensor_factor_matrix[index] = -inverse_matrix_b[index] * alpha_p_power[index].reshape(1, -1)
        tensor_bias_vector[index] = (-inverse_matrix_b[index] @ matrix_c[index].reshape(-1, 1)).reshape(-1)

    return tensor_factor_matrix, tensor_bias_vector


def power_flow_tensor(
    tensor_factor_matrix,
    tensor_bias_vector,
    voltage_guess,
    time_steps,
    node_count,
    iterations,
    tolerance,
):
    """Run the batched Laurent tensor power-flow update."""
    iteration = 0
    voltage_delta = np.inf
    while (iteration < iterations) & (voltage_delta >= tolerance):
        reciprocal_voltage = np.reciprocal(np.conj(voltage_guess))
        residual_term = np.zeros((time_steps, node_count - 1), dtype="complex128")
        for index in prange(time_steps):
            residual_term[index] = tensor_factor_matrix[index] @ reciprocal_voltage[index]
        voltage_solution = tensor_bias_vector + residual_term
        voltage_delta = np.max(np.abs(np.abs(voltage_solution) - np.abs(voltage_guess)))
        voltage_guess = voltage_solution
        iteration += 1

    return voltage_guess, iteration


def power_flow_tensor_constant_power(kernel_matrix, slack_vector, nominal_power, voltage_guess, time_steps, node_count, iterations, tolerance):
    """Run the batched Laurent update for constant-power loads."""
    iteration = 0
    voltage_delta = np.inf
    transposed_power = nominal_power.T
    transposed_voltage = voltage_guess.T

    current_injection = np.zeros((node_count - 1, time_steps), dtype="complex128")
    voltage_drop = np.zeros((node_count - 1, time_steps), dtype="complex128")
    voltage_solution = np.zeros((node_count - 1, time_steps), dtype="complex128")

    while iteration < iterations and voltage_delta >= tolerance:
        current_injection = np.conj(transposed_power * np.reciprocal(transposed_voltage))
        voltage_drop = kernel_matrix @ current_injection
        voltage_solution = voltage_drop + slack_vector
        voltage_delta = np.max(np.abs(np.abs(voltage_solution) - np.abs(transposed_voltage)))
        transposed_voltage = voltage_solution
        iteration += 1

    return transposed_voltage.T, iteration
