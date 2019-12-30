import opfython.utils.logging as l
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)

class SupervisedOPF(OPF):
    """
    """

    def __init__(self, pre_computed_distance=False):
        """Initialization method.

        Args:
            pre_computed_distance (bool): Whether OPF should use pre-computed distances or not.
            
        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SupervisedOPF, self).__init__(pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')

    def fit(self, X, Y):
        """Fits new data in the classifier.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.

        """

        # Creating a subgraph
        self.g = Subgraph(X, Y)

        # Finding prototypes
        self._find_prototypes(self.g)


    def predict(self):
        """
        """
        
        pass
