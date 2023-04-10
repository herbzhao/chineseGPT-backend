# chineseGPT-frontend

# development

`pipenv install`
`pipenv run uvicorn main:app --reload --port=8080 --host=localhost`

# deployment

using fly.io to deploy:
`https://fly.io/docs/languages-and-frameworks/python/`

- procfile needs to specify the port to be 8080 and host to be 0.0.0.0
- convert pipenv to requirements.txt via `pipenv lock -r > requirements.txt`
- set the environment variable via `flyctl secrets set DATABASE_URL=postgres://example.com/mydb`

-automatic deployment via github actions
<https://fly.io/docs/app-guides/continuous-deployment-with-github-actions/>
