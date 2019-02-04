from ER_creator import ER
from detect_clique import DetectClique
from plant_clique import PlantClique


class CliqueInERDetector:

    def __init__(self):
        self._params = {
            'vertices': 100,
            'probability': 0.5,
            'clique_size': 20,
            'directed': False
        }
        self._g = self.create_er()
        self.plant_clique()
        self.detect_clique()

    def create_er(self):
        return ER(self._params)

    def plant_clique(self):
        pc = PlantClique(self._g, self._params)
        self._g = pc.graph()

    def detect_clique(self):
        DetectClique(self._g, self._params)
