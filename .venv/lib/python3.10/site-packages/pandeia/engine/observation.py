# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .config import DefaultConfig
from .custom_exceptions import EngineInputError
from .scene import Scene
from .strategy import StrategyFactory


class Observation(DefaultConfig):

    """
    This class collects the pieces required to make an ETC simulated observation.
    """

    def __init__(self, scene=None, instrument=None, strategy=None, config={}, webapp=False, **kwargs):
        DefaultConfig.__init__(self, config=config, webapp=webapp, **kwargs)

        # make sure we're given a properly configured Instrument
        if instrument is not None:
            self.instrument = instrument
            self.mode = self.instrument.mode
        else:
            msg = "Must provide configured Instrument subclass."
            raise EngineInputError(value=msg)

        # Initialize the observation. This either uses a preconfigured scene, or 
        # requires webapp==False to load a default scene and source.
        if scene is None and not webapp:
            self.scene = Scene(instrument.telescope.tel_name, webapp=webapp)
        else:
            self.scene = scene

        # use strategy we're given or fall back to default for the given instrument if webapp==False
        if strategy is not None:
            self.strategy = strategy
        elif not webapp:
            self.strategy = StrategyFactory(self.instrument, webapp=webapp)
        else:
            msg = "Must provide valid Strategy."
            raise EngineInputError(value=msg)

    def get_random_seed(self):
        """
        Return the configured seed for the random number generator.
        """
        return self.random_seed
