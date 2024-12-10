"""
This dependency is used to register dependencies 
that can be used across different modules.
"""

from src.services.service import Service

service = Service()


def get_service() -> Service:
    """
    Provide a singleton instance of the Service class.

    Returns:
        Service: The shared instance of the Service class
        for managing dependencies.
    """
    return service
