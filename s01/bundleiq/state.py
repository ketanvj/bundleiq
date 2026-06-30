"""
bundleiq/state.py
-----------------
The shared state that flows through the LangGraph graph.

Every node reads from this state and writes back a partial update.
Only define the shape here -- no logic.
"""
from typing import TypedDict


# ---------------------------------------------------------------------------
# TODO 3 of 5 -- State definition
# ---------------------------------------------------------------------------
# Define BundleIQState as a TypedDict with exactly two fields:
#
#   customer_message : str   -- the question the customer typed
#   response         : str   -- the answer BundleIQ will return
#
# Pattern:
#   class BundleIQState(TypedDict):
#       field_name: type
#
# ---------------------------------------------------------------------------

class BundleIQState(TypedDict):
    pass  # TODO 3: replace this line with the two field definitions


# Guard: raises at import time if the fields haven't been defined yet.
if "customer_message" not in BundleIQState.__annotations__:
    raise NotImplementedError(
        "TODO 3: define 'customer_message: str' and 'response: str' "
        "in BundleIQState in bundleiq/state.py"
    )
