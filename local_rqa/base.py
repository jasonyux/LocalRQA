from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class Component(ABC):
    """building block of a pipeline, which will call the `run` method iteratively
    """
    run_input_keys = []

    @abstractmethod
    def run(self, *args, **kwargs):
        """main entrypoint for the component. keywords in `run_input_keys` will be passed in as arguments for this method
        """
        raise NotImplementedError