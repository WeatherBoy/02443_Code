import numpy as np
from typing import Callable



################ Blocking system

# Define blocking system
m = 10              # service units
SERVICE_MEAN = 8    # 1/mean service time
ARRIVAL_MEAN = 1    # mean arrival time


class Event:
    def __init__(self, event_type : str, time : float, ) -> None:
        if event_type.lower() not in ['arrival', 'departure']:
            raise ValueError('event_type must be either arrival or departure')
        self.event_type = event_type
        self.time = time
       

def sample_arrival_poisson_process() -> float:
    """
        Sample the arrival time from a Poisson process.
        
        :return: arrival time (as a float)
    """
    return np.random.exponential(ARRIVAL_MEAN) # TODO: check if is it not 1/arrial_mean


def sample_service_time_exponential() -> float:
    """
        Sample the service time from an exponential distribution.
        
        :return: service time (as a float)
    """
    return np.random.exponential(SERVICE_MEAN) 


def check_available_service(service_units : list[bool]) -> tuple[int, bool]:
    """
    
    """
    for indx, unit_occupied in enumerate(service_units):
        if not unit_occupied:
            return indx, True
        
    return None, False


def apend_event(event_list : list[Event], event_to_append : Event) -> list[Event]:
    """

    """
    for indx, event in enumerate(event_list):
        
        if event.time > event_to_append.time:
            event_list.insert(indx, event_to_append)
            return event_list
        
    event_list.append(event_to_append)
    return event_list


## Simulation
def blocking_simulation(
    simulation_runs : int = 10,
    m : int = 10,
    N : int = 10000,
    sample_arrival : Callable[[], float] = sample_arrival_poisson_process,
    sample_service_time : Callable[[], float] = sample_service_time_exponential
    ) -> tuple[list[float], list[float]]:
    """
        A function for runinng multiple simulations of a blocking system.
        
        :param simulation_runs: number of simulations to run
        :param m: number of service units
        :param N: number of customers
        :param sample_arrival: function for sampling arrival time
        :param sample_service_time: function for sampling service time
        
        :return: list of blocked fractions and list of average arrival times
    """
    # NOTE: Maybe use burn in period...?
    
    blocked_fractions = []
    arrival_times = []
    for i in range(simulation_runs):
        print(f"run {i+1}")
        custmer_count = 0
        global_time = 0
        event_counter = 0
        block_count = 0
        arrivals = 0

        # lists
        event_list = []
        service_units_status = [False for _ in range(m)]    # <-- Indicates whether the service units are occupied or not
        
        # First arrival
        first_arrival = sample_arrival()
        event_list.append(Event('arrival', global_time + first_arrival))
        arrivals += first_arrival
        
        global_time += first_arrival
        event_list.append(Event('departure', global_time + sample_service_time()))
        service_units_status[0] = True # <-- unit 1 is occupied

        while custmer_count < N:
            
            current_event = event_list[event_counter]

            # Increment global time
            global_time = current_event.time

            if current_event.event_type == 'arrival':
                custmer_count += 1

                # Check for free service units
                indx, available = check_available_service(service_units_status)
                
                if available:
                    # Insert departure event and depend to eventlist
                    
            
                    departure_event = Event('departure', global_time + sample_service_time())
                    event_list = apend_event(event_list, departure_event)

                    # Take service unit
                    service_units_status[indx] = True # <-- unit indx is occupied

                if not available:
                    # Costumer blocked
                    block_count += 1
                
                # insert time for next arrival
                new_arrival = sample_arrival()
                arrival += new_arrival
                arrival_event = Event('arrival', global_time + new_arrival)
                event_list = apend_event(event_list, arrival_event)

            elif current_event.event_type == 'departure': 
                # Free the service unit for the current departure event
                for indx, unit_occupied in enumerate(service_units_status):
                    if unit_occupied:
                        service_units_status[indx] = False # <-- unit indx is free
                        break
                        
            # increment event counter
            event_counter += 1
        
        blocked_fractions.append(block_count / N)
        arrival_times.append(arrivals / N)
    
    return blocked_fractions, arrival_times





