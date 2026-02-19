"""Interactive Brokers data provider via ib_insync.

Re-exports from bds-data-providers shared package.
"""

from bds_data_providers.ib_market import (
    IBMarketProvider as IBProvider,
)
from bds_data_providers.ib_market import (
    is_available,
)

__all__ = ["IBProvider", "is_available"]
