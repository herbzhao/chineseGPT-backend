@echo off
pipenv run uvicorn main:app --reload --port=8080 --host=localhost