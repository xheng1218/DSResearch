# Docker file for the generator
FROM python:3.7

WORKDIR /usr/app

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
ADD ./ ./

CMD ["python3", "app.py"]