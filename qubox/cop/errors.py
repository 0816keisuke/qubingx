class Errors:
    class ModelError(Exception):
        """Invalid model name."""

        pass

    class GroupError(Exception):
        """Invalid group name."""

        pass
