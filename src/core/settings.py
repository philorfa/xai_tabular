from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = ""
    ENV: str = ""
    VERSION: str = ""
    LOCATION: str = ""

    model_config = ConfigDict(env_file=f".env.{os.getenv('ENV', 'dev')}",
                              extra="allow")


@lru_cache()
def get_settings():
    return Settings()
