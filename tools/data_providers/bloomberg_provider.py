"""Bloomberg Professional API data provider.

Re-exports from bds-data-providers shared package.
"""

from bds_data_providers.bloomberg_market import (
    BloombergMarketProvider as BloombergProvider,
)
from bds_data_providers.bloomberg_market import (
    is_available,
)

__all__ = ["BloombergProvider", "is_available"]
