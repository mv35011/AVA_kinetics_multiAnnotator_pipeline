import random
from typing import List, Dict, Any, Optional


class AssignmentGenerator:
    """
    Selects annotators for a single batch task, handling overlap.
    """

    def __init__(self):
        pass

    def select_annotators_for_batch(
            self,
            all_annotators: List[str],
            num_required: int = 2
    ) -> List[str]:
        """
        Selects a random subset of annotators to work on a single batch task.

        Args:
            all_annotators: A list of all available annotator usernames.
            num_required: The number of annotators to assign to this batch (e.g., 2 for overlap).

        Returns:
            A list of selected annotator usernames.
        """
        if not all_annotators:
            raise ValueError("Annotators list cannot be empty.")

        num_to_select = min(num_required, len(all_annotators))

        selected_annotators = random.sample(all_annotators, k=num_to_select)

        return selected_annotators