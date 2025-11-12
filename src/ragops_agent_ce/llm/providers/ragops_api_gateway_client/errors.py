class RagopsAPIGatewayError(Exception):
    """Base exception for ragops-api-gateway client errors."""


class RagopsAPIGatewayResponseError(RagopsAPIGatewayError):
    """Exception raised when API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class RagopsAPIGatewayMaxAttemptsExceededError(RagopsAPIGatewayError):
    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts
        super().__init__(
            f"Failed to connect to Donkit agent. Max attempts: {max_attempts}. Exiting."
        )


class RagopsAPIGatewayConnectionError(RagopsAPIGatewayError):
    """Connection error."""
