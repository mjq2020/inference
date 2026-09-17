"""Small, transport-independent errors for the RV1126B application."""


class EdgeError(Exception):
    def __init__(self, message, *, code="edge_error", status_code=400, details=None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}
