import numpy as np


class GetData():
    def __init__(self, n_instance, n_customers, capacity=50.0):
        self.n_instance = n_instance
        self.n_customers = n_customers
        self.capacity = float(capacity)

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for idx in range(self.n_instance):
            coords = np.random.rand(self.n_customers + 1, 2).astype(np.float32)
            coords[0] = np.asarray([0.5, 0.5], dtype=np.float32)
            demands = np.zeros(self.n_customers + 1, dtype=np.float32)
            demands[1:] = np.random.randint(1, 10, size=self.n_customers).astype(np.float32)
            instance_data.append(
                {
                    "name": f"cvrp_{self.n_customers}_{idx}",
                    "coords": coords,
                    "demands": demands,
                    "capacity": self.capacity,
                }
            )
        return instance_data
