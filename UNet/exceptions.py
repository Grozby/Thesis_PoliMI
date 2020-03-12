class BadConfigurationFile(Exception):
    def __init__(self, msg=None):
        assert(msg is not None)


class IncorrectParserType(Exception):
    def __init__(self, msg=None):
        assert(msg is not None)
