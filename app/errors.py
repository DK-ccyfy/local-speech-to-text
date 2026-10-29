"""Application errors that map cleanly onto HTTP responses."""


class AppError(Exception):
    """An error with an HTTP status code and a user-readable message."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
