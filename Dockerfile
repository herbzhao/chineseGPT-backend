# 
FROM python:3.9-bullseye

# 
WORKDIR /code

# 
COPY ./ /code/

# 
# install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra
# use pipenv to install dependencies
RUN pip3 install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]