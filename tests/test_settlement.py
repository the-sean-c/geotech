from geotech.config import Config
from geotech.loads import PointLoad
from geotech.soils import SoilLayer, SoilProfile, WaterTable

layer1 = SoilLayer("A", 100, 90, 20, 18, 0, 35, 0.33, 0.03, 1)
layer2 = SoilLayer("B", 90, 80, 20, 18, 0, 35, 0.33, 0.03, 1)

pwp = WaterTable(0, 0.0)

profile = SoilProfile([layer1, layer2], pwp)

load = PointLoad(100, 100, 0, 0)
