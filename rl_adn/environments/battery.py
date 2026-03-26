import numpy as np

battery_parameters={
'capacity':300,# kW.h
'max_charge':50, # kW
'max_discharge':50, #kW
'efficiency':1,
'degradation':0, #euro/kw
'max_soc':0.8,
'min_soc':0.2,
'initial_soc':0.4}

DEFAULT_TIME_INTERVAL_MINUTES = 15.0

class Battery():
    """
    A simple battery model for energy storage and management.

    Attributes:
        capacity (float): The total energy capacity of the battery in kW.h.
        max_soc (float): The maximum state of charge (SOC) as a fraction of capacity.
        initial_soc (float): The initial state of charge as a fraction of capacity.
        min_soc (float): The minimum state of charge as a fraction of capacity.
        degradation (float): The cost of battery degradation per kW.
        max_charge (float): The maximum charging power of the battery in kW.
        max_discharge (float): The maximum discharging power of the battery in kW.
        efficiency (float): The efficiency of charging and discharging processes.
        current_soc (float): The current state of charge of the battery.

    Args:
        parameters (dict): A dictionary containing battery parameters.

    Description:
        This class simulates a simple battery for energy storage. It allows for charging and discharging
        operations while considering constraints like maximum/minimum state of charge, charging/discharging
        rates, and efficiency. It also calculates the cost associated with battery degradation.
    """

    def __init__(self, parameters):
        """
        Initializes the Battery object with given parameters.

        Args:
            parameters (dict): A dictionary containing the battery's parameters such as capacity, state of charge limits,
                               degradation cost, maximum charge/discharge rates, and efficiency.
        """
        self.capacity = parameters['capacity']  # 容量
        self.max_soc = parameters['max_soc']  # max soc 0.8
        self.initial_soc = parameters['initial_soc']  # initial soc 0.4
        self.min_soc = parameters['min_soc']  # 0.2
        self.degradation = parameters['degradation']  # degradation cost 0，
        self.max_charge = parameters['max_charge']  # max charge ability
        self.max_discharge = parameters['max_discharge']  # max discharge ability
        self.efficiency = parameters['efficiency']  # charge and discharge efficiency
        self.time_interval_minutes = float(parameters.get('time_interval_minutes', DEFAULT_TIME_INTERVAL_MINUTES))
        if self.time_interval_minutes <= 0:
            raise ValueError("time_interval_minutes must be positive")

    def step(self, action_battery):
        """
        Executes a step of battery operation based on the given action.

        Args:
            action_battery (float): The action to be taken, typically representing the amount of energy to charge or discharge.

        Description:
            This method updates the state of charge (SOC) of the battery based on the given action. It calculates the
            energy change and updates the SOC while ensuring it stays within the defined minimum and maximum limits.
            The energy change is also used to calculate the cost associated with battery operation.
        """
        action_array = np.asarray(action_battery).reshape(-1)
        if action_array.size != 1:
            raise ValueError("Battery.step expects a scalar action or a size-1 array")
        action_battery = float(action_array[0])
        rated_power = self.max_discharge if action_battery >= 0 else self.max_charge
        power_command = action_battery * rated_power
        interval_hours = self.time_interval_minutes / 60.0
        updated_soc = max(
            self.min_soc,
            min(
                self.max_soc,
                (self.current_soc * self.capacity + power_command * interval_hours) / self.capacity,
            ),
        )

        self.energy_change = (updated_soc - self.current_soc) * self.capacity / interval_hours  # if charge, positive, if discharge, negative
        self.current_soc = updated_soc  # update capacity to current codition

    def _get_cost(self, energy):  # calculate the cost depends on the energy change
        """
        Calculates the cost associated with a given energy change.

        Args:
            energy (float): The amount of energy change in the battery.

        Returns:
            float: The calculated cost based on the energy change.

        Description:
            This method calculates the cost of operating the battery, which is a function of the absolute value of the energy change.
            It is used internally to assess the cost implications of charging or discharging the battery.
        """
        cost = np.abs(energy)
        return cost

    def SOC(self):
        """
        Returns the current state of charge (SOC) of the battery.

        Returns:
            float: The current SOC of the battery.

        Description:
            This method provides the current state of charge of the battery as a fraction of its total capacity.
        """
        return self.current_soc

    def reset(self):
        """
        Resets the state of charge (SOC) of the battery to its initial value.

        Description:
            This method is used to reset the battery's state of charge to its initial value, typically used at the start of a new simulation or operational cycle.
        """
        self.current_soc = self.initial_soc
