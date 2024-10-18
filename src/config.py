from pydantic_settings import BaseSettings
import sys
import os

class Config(BaseSettings):

    PATH_DATA_INGESTIONS: str

    class Config:
        env_file ='.env'
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


CONFIG = Config()
