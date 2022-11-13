# FROM ubuntu:latest
FROM python:3.8.1

COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt

RUN pip3 install --no-cache-dir -r /exp/requirements.txt

WORKDIR /exp

ENV FLASK_APP="api.py"

EXPOSE 5000

# CMD ["python3", "plot_digits_classification.py"]
CMD ["flask", "run", "--host", "0.0.0.0"]