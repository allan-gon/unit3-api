from fastapi import APIRouter
from .predict import RedditPost, subs
import numpy as np
from random import sample
import plotly.express as px

router = APIRouter()


@router.post('/viz')
async def viz(item: RedditPost):
    """
    ### Request Body
    - `title`: string the title of the post
    - `body`: string the meat of the post
    - `n`: int number of subreddits you want back

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/)
    """
    data = item.to_df()

    # get predictions
    names = sample(subs, item.n)
    values = np.random.dirichlet(np.ones(len(names)), size=1)[0]  # replace with predict proba

    # Make Plotly figure
    fig = px.pie(values=values, names=names, title='Subreddits To Post To')
    fig.show()
    # Return Plotly figure as JSON string
    return fig.to_json()
