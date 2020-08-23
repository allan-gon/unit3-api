import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

subs = [
    'talesfromtechsupport', 'teenmom', 'Harley', 'ringdoorbell', 'intel', 'residentevil', 'BATProject', 'hockeyplayers',
    'asmr', 'rawdenim', 'steinsgate', 'DBZDokkanBattle', 'Nootropics', 'l5r', 'NameThatSong', 'homeless', 'antidepressants',
    'absolver', 'KissAnime', 'sissyhypno', 'oculusnsfw', 'dpdr', 'Garmin', 'AskLiteraryStudies', 'poetry_critics', 'skiing',
    'shrimptank', 'logorequests', 'Stargate', 'foreskin_restoration', 'sharepoint', 'synthesizers', 'gravityfalls', 'androiddev',
    'Grimdawn', 'driving', 'FORTnITE', 'dndnext', 'Magic', 'MtvChallenge', 'FoWtcg', 'harrypotter', 'TryingForABaby', 'sewing', 'foxholegame',
    'madmen', 'JUSTNOMIL', 'APStudents', 'sharditkeepit', 'amateurradio', 'sleeptrain', 'fatpeoplestories', 'GameStop', 'scuba', 'Firefighting',
    'Mustang', 'riverdale', 'flying', 'bartenders', 'scooters', 'trumpet', 'projecteternity', 'musictheory', 'factorio', 'SexToys', 'EternalCardGame',
    'PLC', 'sailing', 'Mattress', 'climbing', 'uberdrivers', 'Cloud9', 'csharp', 'communism101', 'windowsphone', 'AskAnthropology', 'secretsanta',
    'Volkswagen', 'BigBrother', 'osugame', 'spartanrace', 'needforspeed', 'Cruise', 'blackmirror', 'China', 'resumes', 'homeassistant', 'starcraft',
    'Cubers', 'Warframe', 'Professors', 'parrots', 'TOR', 'AvPD', 'Landlord', 'WhiteWolfRPG', 'DBS_CardGame', 'atheism', 'buffy', 'Shoplifting',
    'reddeadredemption', 'germany', 'Schizoid', 'Nanny', 'WWEChampions', 'MMA', 'MSLGame', 'French', 'cosplay', 'sugarlifestyleforum', 'PHPhelp',
    'WarhammerCompetitive', 'Iota', 'CryptoKitties', 'snakes', 'securityguards', 'Hue', 'Costco', 'IASIP', 'tacobell', 'jewelry', 'EmulationOnAndroid',
    'Rabbits', 'thesims', 'dresdenfiles', 'Hunting', 'MoviePassClub', 'TowerofGod', 'Allergies', 'snapchat', 'nanocurrency', 'Veterans', 'CaptainTsubasaDT',
    'Anarchism', 'indonesia', 'horror', 'malaysia', 'theydidthemath', 'fleshlight', 'AcademicPsychology', 'productivity', 'LinkinPark', 'fatestaynight', 'kucoin',
    'excel', 'tea', 'turning', 'UnresolvedMysteries', 'diabetes', 'eczema', 'whatsthisworth', 'westworld', 'thewalkingdead', 'docker', 'xxfitness', 'emojipasta',
    'synology', 'puppy101', 'Libraries', 'dji', 'survivor', 'muacjdiscussion', 'GMAT', 'DunderMifflin', 'bigboobproblems', 'LDESurvival', 'discgolf', 'Dreams',
    'headphones', 'StudentLoans', 'bourbon', 'Geosim', 'Plumbing', 'ptsd', 'lawschooladmissions', 'greysanatomy', 'PrettyLittleLiars', 'AlphaBayMarket', 'Snus',
    'TheExpanse', 'Miscarriage', 'Eve', 'uscg', 'fakeid', 'drumcorps', 'wacom', 'SonyXperia', 'vexillology', 'formula1', 'animation', 'digitalnomad', 'graphic_design',
    'VisitingIceland', 'widowers', 'tabletopgamedesign', 'cats', 'RocketLeague', 'GirlsXBattle', 'techsupport', 'woodworking', 'AutoModerator', 'yandere_simulator',
    'ABraThatFits', 'HotPeppers', 'yoga', 'hookah', 'guitars', 'weddingplanning', 'biology', 'PlasticSurgery', 'obs', 'GodofWar', 'AstralProjection', 'malehairadvice',
    'TalesFromThePizzaGuy', 'boxoffice', 'kingdomcome', 'TheSimpsons', 'beards', 'volleyball', 'tarot', 'Epilepsy', 'italy', 'SiliconValleyHBO', 'codes', 'TokyoGhoul'
]

# for the real thing
# import pickle
# with open('model', 'rb') as file:
#         model = pickle.load(file)

log = logging.getLogger(__name__)
router = APIRouter()


class RedditPost(BaseModel):
    """Use this data model to parse the request body JSON."""

    title: str = Field(..., example='This is a reddit title')
    body: str = Field(..., example='This is the text in a reddit post')
    n: int = Field(..., example=1)
    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([{'title': self.title, 'body': self.body}])

    @validator('title')
    def check_title(cls, value):
        """Validate that title is a string."""
        assert type(value) == str, f'Title == {value}, must be a string'
        return value
    
    @validator('body')
    def check_body(cls, value):
        """Validate that title is a string."""
        assert type(value) == str, f'Body == {value}, must be a string'
        return value
    
    @validator('n')
    def check_n(cls, value):
        """Validate that title is a string."""
        assert type(value) == int, f'n == {value}, must be an integer'
        assert value >= 1, f'n == {value}, must be greater than or equal to 1'
        return value


@router.post('/dummy-predict')
async def predict(item: RedditPost):
    """
    Randomly return a sub, hence dummy predict

    ### Request Body
    - `title`: string title of the post
    - `body`: string the content of the post
    - `n`: int how predictions you want back when using this route regardless of n you will get back 
    one result i did it this way so that you'd always request with the same schema. If you want multiple predictions
    use the n-dummy-predict route

    ### Response
    - `prediction`: string, the subreddit the model thinks this post belongs to
    """
    data = item.to_df()
    log.info(data)
    prediction = random.choice(subs)
    return {
        'subreddit prediction': prediction,
    }  # model.predict(data)


@router.post('/n-dummy-predict')
async def multiple_predict(item: RedditPost):
    """
    Return n subs

    ### Request Body
    - `title`: string the title of the post
    - `body`: string the meat of the post
    - `n`: int number of subreddits you want back
    ### Response
    - `prediction`: string, the subreddit the model thinks this post belongs to
    """
    data = item.to_df()
    log.info(data)
    predictions = random.sample(subs, item.n)
    return {
        'subreddit prediction': predictions,
    }  # model.predict(data)