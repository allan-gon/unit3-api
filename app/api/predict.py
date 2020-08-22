import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

# for the real thing
import pickle

log = logging.getLogger(__name__)
router = APIRouter()


class RedditPost(BaseModel):
    """Use this data model to parse the request body JSON."""

    title: str = Field(..., example='This is a reddit title')
    body: str = Field(..., example='This is the text in a reddit post')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('title')
    def check_title(cls, value):
        """Validate that title is a string."""
        assert type(value) == str, f'Title == {value}, must be a string'
        return value


@router.post('/predict')
async def predict(title, body):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `title`: string
    - `body`: string

    ### Response
    - `prediction`: string, the subreddit the model thinks this post belongs to
    """

    data = RedditPost(title=title, body=body).to_df()
    log.info(data)
    prediction = random.choice(['r/AMA', 'r/Politics', 'r/PCMasterrace'])
    return {
        'subreddit prediction': prediction,
    }

