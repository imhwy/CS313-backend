"""
This module defines a FastAPI router for handling clip text retrieval requests.
"""

from fastapi import (
    status,
    Depends,
    APIRouter,
    HTTPException,
)
from fastapi.responses import JSONResponse

from src.services.service import Service
from src.dependencies.dependency import get_service


classification_router = APIRouter(
    tags=["Classification"],
    prefix="/api/v1",
)


@classification_router.post(
    "/classify-comment",
    status_code=status.HTTP_200_OK,
    response_description="Successful Response!!!"
)
async def classify_comment(
    text: str,
    service: Service = Depends(get_service)
) -> JSONResponse:
    """
    Endpoint for classifying a comment's sentiment.

    Args:

        text (str): The text of the comment to classify. Must be non-empty.

        service (Service): The Service instance for dependency injection, 
                           used to access the comment classification.

    Returns:

        JSONResponse: A JSON response containing the predicted sentiment.

    Raises:

        HTTPException:

            - If the `text` parameter is empty, raises a 404 Not Found error.

            - If an unexpected error occurs, raises a 500 Internal Server Error.
    """
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query is required"
        )

    try:
        result = await service.comment_classification.predict(text=text)

        return JSONResponse(
            content={
                "predict": result,
            },
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
